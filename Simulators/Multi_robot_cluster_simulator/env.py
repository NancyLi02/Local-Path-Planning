"""Multi-robot cluster local-replanning environment (RL training).

Each reset randomly builds a DENSE cluster scenario:
  * N in [1, 6] AMRs, each on a straight reference rail,
  * rail orientations are horizontal/vertical in random combination
    (=> parallel lanes and perpendicular crossings),
  * rails packed close together so the AMRs converge in a central region,
  * 1-2 human workers crossing that region.
This reproduces exactly the situation a Step-D cluster faces. The RL controls
ALL AMRs from the first step (no shield -- it must learn avoidance from the
reward). Per-AMR action is a body-frame local goal (fwd, lat) followed by a
pure-pursuit controller, matching the single-robot simulator.

SUCCESS = every AMR has returned to (reached the far end of) its reference path
with NO collision (AMR-AMR or AMR-human) during the episode.

Interface (for the attention PPO, variable agent count):
  reset(seed) -> obs (N_MAX, OBS_DIM) float32, with `active_mask` (N_MAX,)
  step(action (N_MAX, 2)) -> obs, reward (scalar), done, info
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N_MAX = 6
N_LOOKAHEAD = 4
OBS_DIM = 4 + 2 * N_LOOKAHEAD + 5 + 4      # own + lookahead + human + peer = 21


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class StraightRail:
    """Minimal straight reference path with arc-length / lateral queries."""
    def __init__(self, start, end):
        self.a = np.asarray(start, float)
        self.b = np.asarray(end, float)
        d = self.b - self.a
        self.length = float(np.linalg.norm(d))
        self.dir = d / (self.length + 1e-9)
        self.heading = float(math.atan2(self.dir[1], self.dir[0]))
        self.normal = np.array([-self.dir[1], self.dir[0]])

    def position(self, s):
        return self.a + self.dir * float(np.clip(s, 0.0, self.length))

    def project(self, x, y):
        """Return (s, signed_lateral)."""
        p = np.array([x, y]) - self.a
        s = float(np.dot(p, self.dir))
        lat = float(np.dot(np.array([x, y]) - self.position(s), self.normal))
        return float(np.clip(s, 0.0, self.length)), lat


@dataclass
class ClusterEnvConfig:
    map_size: float = 20.0
    dt: float = 0.1
    max_steps: int = 220
    robot_radius: float = 0.30
    human_radius: float = 0.30
    max_v: float = 1.0
    max_omega: float = 1.5
    rail_half: float = 6.0            # rail extends +-this around the region centre
    region_half: float = 2.2         # central conflict region half-size
    offset_range: float = 1.7        # perpendicular rail offset from centre (dense)
    n_humans_range: tuple = (1, 2)
    human_speed_range: tuple = (0.25, 0.5)
    amr_collision_dist: float = 0.66  # centre-centre AMR-AMR
    human_collision_dist: float = 0.55
    safety_dist: float = 1.3
    goal_tol: float = 0.6            # reached far end of rail
    success_lat: float = 0.5
    # action ranges (body frame local goal)
    fwd_range: tuple = (0.0, 3.0)
    lat_range: tuple = (-2.5, 2.5)
    oob_lat: float = 4.5             # |lateral| beyond which an AMR is "lost" (fail)
    # reward weights  (progress-dominant; deviation MILD + bounded so detours are ok)
    w_collision: float = -100.0
    w_oob: float = -60.0
    w_success: float = 200.0
    w_progress: float = 30.0
    w_deviation: float = -0.6        # × gated, linear, capped |lat|
    w_heading: float = -0.4
    w_return: float = 3.0
    w_safety: float = -3.0
    w_time: float = -0.3
    pen_min: float = 0.15
    pen_restore: float = 2.5


class MultiRobotClusterEnv:
    def __init__(self, cfg: ClusterEnvConfig | None = None):
        self.cfg = cfg or ClusterEnvConfig()
        self.N = N_MAX
        self._diag = self.cfg.map_size * 1.2
        self._lat_s = 2.0
        self._pos_s = N_LOOKAHEAD * 1.0 + 3.0
        self._vel_s = self.cfg.max_v + self.cfg.human_speed_range[1]

    # ---------------- scenario ----------------
    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        cx = cy = c.map_size / 2.0
        self.n = int(self.rng.integers(1, N_MAX + 1))

        self.rails: list[StraightRail] = []
        self.amr = []          # per-AMR dict
        min_sep = 1.4          # min distance between AMR start positions
        starts: list[np.ndarray] = []
        attempts = 0
        while len(self.rails) < self.n and attempts < 400:
            attempts += 1
            vertical = bool(self.rng.integers(0, 2))
            sgn = float(self.rng.choice([-1.0, 1.0]))
            off = float(self.rng.uniform(-c.offset_range, c.offset_range))
            if vertical:
                x = cx + off
                a = np.array([x, cy - sgn * c.rail_half]); b = np.array([x, cy + sgn * c.rail_half])
            else:
                y = cy + off
                a = np.array([cx - sgn * c.rail_half, y]); b = np.array([cx + sgn * c.rail_half, y])
            if any(float(np.linalg.norm(a - p)) < min_sep for p in starts):
                continue                      # reject overlapping start
            starts.append(a)
            rail = StraightRail(a, b)
            self.rails.append(rail)
            self.amr.append(dict(x=float(a[0]), y=float(a[1]), th=rail.heading,
                                 v=0.0, s=0.0, done=False, collided=False))
        self.n = len(self.rails)              # may be < requested if packed tight

        # humans crossing the central region, kept clear of AMR starts
        self.humans = []
        nh = int(self.rng.integers(c.n_humans_range[0], c.n_humans_range[1] + 1))
        for _ in range(nh):
            for _try in range(40):
                ang = float(self.rng.uniform(0, 2 * np.pi))
                d = np.array([math.cos(ang), math.sin(ang)])
                perp = np.array([-d[1], d[0]])
                start = np.array([cx, cy]) - d * (c.region_half + 2.5) + perp * self.rng.uniform(-1.0, 1.0)
                if all(float(np.linalg.norm(start - p)) > 1.6 for p in starts):
                    break
            sp = float(self.rng.uniform(*c.human_speed_range))
            self.humans.append(dict(x=float(start[0]), y=float(start[1]),
                                    vx=float(d[0] * sp), vy=float(d[1] * sp)))

        self.steps = 0
        self._min_haz = np.full(self.n, np.inf)
        self._prev_lat = np.zeros(self.n)
        self._traj = [[ (a["x"], a["y"]) ] for a in self.amr]
        self._htraj = [[ (h["x"], h["y"]) ] for h in self.humans]
        return self._obs()

    # ---------------- helpers ----------------
    def _nearest_human(self, ax, ay):
        best = None
        for h in self.humans:
            d = math.hypot(ax - h["x"], ay - h["y"])
            if best is None or d < best[0]:
                best = (d, h)
        return best

    def _nearest_peer(self, i):
        best = None
        for j in range(self.n):
            if j == i or self.amr[j]["done"] or self.amr[j]["collided"]:
                continue
            a, b = self.amr[i], self.amr[j]
            d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if best is None or d < best[0]:
                best = (d, b)
        return best

    def _obs(self):
        c = self.cfg
        obs = np.zeros((self.N, OBS_DIM), dtype=np.float32)
        mask = np.zeros(self.N, dtype=bool)
        for i in range(self.n):
            a = self.amr[i]
            if a["done"] or a["collided"]:
                continue
            mask[i] = True
            rail = self.rails[i]
            s, lat = rail.project(a["x"], a["y"])
            cr, sr = math.cos(a["th"]), math.sin(a["th"])
            hdg = _wrap(a["th"] - rail.heading)
            row = [a["v"] / c.max_v, s / rail.length, lat / self._lat_s, hdg / np.pi]
            for k in range(1, N_LOOKAHEAD + 1):
                p = rail.position(s + k * 1.0)
                dx, dy = p[0] - a["x"], p[1] - a["y"]
                row += [(cr * dx + sr * dy) / self._pos_s, (-sr * dx + cr * dy) / self._pos_s]
            nh = self._nearest_human(a["x"], a["y"])
            if nh is not None:
                d, h = nh
                dx, dy = h["x"] - a["x"], h["y"] - a["y"]
                hrx, hry = cr * dx + sr * dy, -sr * dx + cr * dy
                dvx, dvy = h["vx"] - a["v"] * cr, h["vy"] - a["v"] * sr
                hrvx, hrvy = cr * dvx + sr * dvy, -sr * dvx + cr * dvy
                risk = 1.0 if d < c.safety_dist * 1.5 else 0.0
                row += [hrx / self._diag, hry / self._diag, hrvx / self._vel_s, hrvy / self._vel_s, risk]
            else:
                row += [1.0, 0.0, 0.0, 0.0, 0.0]
            npr = self._nearest_peer(i)
            if npr is not None:
                d, b = npr
                dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                prx, pry = cr * dx + sr * dy, -sr * dx + cr * dy
                pvx, pvy = b["v"] * math.cos(b["th"]) - a["v"] * cr, b["v"] * math.sin(b["th"]) - a["v"] * sr
                prvx, prvy = cr * pvx + sr * pvy, -sr * pvx + cr * pvy
                row += [prx / self._diag, pry / self._diag, prvx / self._vel_s, prvy / self._vel_s]
            else:
                row += [1.0, 0.0, 0.0, 0.0]
            obs[i] = row
        self._mask = mask
        return obs

    @property
    def active_mask(self):
        return self._mask.copy()

    # ---------------- step ----------------
    def step(self, action):
        c = self.cfg
        self.steps += 1
        action = np.asarray(action, dtype=float)
        fwd_lo, fwd_hi = c.fwd_range; lat_lo, lat_hi = c.lat_range

        old_s = np.array([self.rails[i].project(self.amr[i]["x"], self.amr[i]["y"])[0]
                          for i in range(self.n)])

        # advance each active AMR via pure pursuit toward its body-frame local goal
        for i in range(self.n):
            a = self.amr[i]
            if a["done"] or a["collided"]:
                a["v"] = 0.0
                continue
            # policy outputs raw (-1,1) per dim; map to (fwd>=0, lat) ranges
            r0 = float(np.clip(action[i, 0], -1.0, 1.0))
            r1 = float(np.clip(action[i, 1], -1.0, 1.0))
            fwd = (r0 * 0.5 + 0.5) * fwd_hi
            lat = r1 * lat_hi
            cr, sr = math.cos(a["th"]), math.sin(a["th"])
            gx = a["x"] + fwd * cr - lat * sr
            gy = a["y"] + fwd * sr + lat * cr
            v, w = self._pursue(a["x"], a["y"], a["th"], gx, gy)
            if abs(fwd) < 0.05 and abs(lat) < 0.05:
                v = w = 0.0
            a["x"] += v * math.cos(a["th"]) * c.dt
            a["y"] += v * math.sin(a["th"]) * c.dt
            a["th"] = _wrap(a["th"] + w * c.dt)
            a["v"] = v
            self._traj[i].append((a["x"], a["y"]))

        for h in self.humans:
            h["x"] += h["vx"] * c.dt
            h["y"] += h["vy"] * c.dt
        for hi, h in enumerate(self.humans):
            self._htraj[hi].append((h["x"], h["y"]))

        # collisions
        collision = False
        for i in range(self.n):
            a = self.amr[i]
            if a["done"] or a["collided"]:
                continue
            for h in self.humans:
                if math.hypot(a["x"] - h["x"], a["y"] - h["y"]) < c.human_collision_dist:
                    a["collided"] = True; collision = True
            for j in range(i + 1, self.n):
                b = self.amr[j]
                if b["done"] or b["collided"]:
                    continue
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < c.amr_collision_dist:
                    a["collided"] = b["collided"] = True; collision = True

        # progress / goal / out-of-bounds
        new_s = np.zeros(self.n)
        oob = False
        for i in range(self.n):
            a = self.amr[i]
            s, lat = self.rails[i].project(a["x"], a["y"])
            new_s[i] = s
            if not a["done"] and not a["collided"]:
                if abs(lat) > c.oob_lat:
                    oob = True
                if s >= self.rails[i].length - c.goal_tol and abs(lat) < c.success_lat:
                    a["done"] = True; a["v"] = 0.0

        reward = self._reward(old_s, new_s, collision, oob)

        success = all(a["done"] for a in self.amr) and not any(a["collided"] for a in self.amr)
        done = collision or oob or success or self.steps >= c.max_steps
        obs = self._obs() if not done else np.zeros((self.N, OBS_DIM), dtype=np.float32)
        n_done = sum(a["done"] for a in self.amr)
        info = dict(success=bool(success), collision=bool(collision), oob=bool(oob),
                    completion=n_done / self.n, n=self.n,
                    n_collided=sum(a["collided"] for a in self.amr))
        return obs, float(reward), bool(done), info

    def _pursue(self, x, y, th, gx, gy):
        c = self.cfg
        dx, dy = gx - x, gy - y
        cs, sn = math.cos(th), math.sin(th)
        lx = cs * dx + sn * dy
        ly = -sn * dx + cs * dy
        L = max(math.hypot(lx, ly), 0.3)
        kappa = 2.0 * ly / (L * L)
        v = c.max_v * float(np.clip(L / 2.0, 0.3, 1.0))
        v *= float(np.clip(1.0 - 0.5 * abs(kappa), 0.2, 1.0))
        w = float(np.clip(v * kappa, -c.max_omega, c.max_omega))
        return v, w

    def _reward(self, old_s, new_s, collision, oob=False):
        c = self.cfg
        if collision:
            return c.w_collision
        if oob:
            return c.w_oob
        r = 0.0
        for i in range(self.n):
            a = self.amr[i]
            if a["collided"]:
                continue
            s, lat = self.rails[i].project(a["x"], a["y"])
            # nearest hazard distance (human or peer)
            hd = math.inf
            nh = self._nearest_human(a["x"], a["y"])
            if nh is not None:
                hd = min(hd, nh[0])
            npr = self._nearest_peer(i)
            if npr is not None:
                hd = min(hd, npr[0])
            self._min_haz[i] = min(self._min_haz[i], hd)
            # deviation gate: allow deviation near a hazard
            g = c.pen_min + (1 - c.pen_min) * min(hd / c.pen_restore, 1.0) if hd < math.inf else 1.0
            if a["done"]:
                continue
            r += c.w_progress * max(0.0, new_s[i] - old_s[i])           # progress-dominant
            r += c.w_deviation * g * min(abs(lat), 3.0)                 # MILD bounded deviation
            r += c.w_heading * g * abs(_wrap(a["th"] - self.rails[i].heading))
            if hd < c.safety_dist:
                r += c.w_safety * (c.safety_dist - hd) / c.safety_dist
            if g >= 0.9 and abs(lat) > c.success_lat:
                r += c.w_return * max(0.0, self._prev_lat[i] - abs(lat))
            r += c.w_time
            self._prev_lat[i] = abs(lat)
        if all(a["done"] for a in self.amr):
            r += c.w_success
        return r
