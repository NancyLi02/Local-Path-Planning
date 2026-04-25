"""
State-Machine DWA Baseline for Local Path Planning with Human Avoidance

Three-state navigation framework:
  PATH_FOLLOW  – Track the reference path at full speed via lookahead points.
  DWA_AVOID    – Human detected in the corridor; DWA local planner takes over.
  REJOIN_PATH  – Obstacle cleared; lateral-correction controller steers back
                 onto the reference path.

Transitions (with hysteresis to prevent chattering):
  PATH_FOLLOW → DWA_AVOID   : human visible, ahead, and within threat radius
  DWA_AVOID   → REJOIN_PATH : human behind robot OR far away (no longer a threat)
  REJOIN_PATH → PATH_FOLLOW : lateral offset < threshold AND heading aligned
  REJOIN_PATH → DWA_AVOID   : human reappears as threat during rejoin

Usage:
    python dwa_baseline.py                                   # 10 eps headless
    python dwa_baseline.py --episodes 20 --save-video DWA_SM # save gifs
    python dwa_baseline.py --episodes 5 --show-candidates    # DWA fan overlay
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import time
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from light_weight_simulator import (
    LocalPlannerEnv,
    PurePursuitController,
    wrap_angle,
    _obs_to_path_goal,
    _obs_normalization_scales,
    _result_tag,
    _show_result,
)


# ======================================================================
# Navigation state
# ======================================================================

class NavState(Enum):
    PATH_FOLLOW = auto()
    DWA_AVOID = auto()
    REJOIN_PATH = auto()


_STATE_COLORS = {
    NavState.PATH_FOLLOW: "#388e3c",
    NavState.DWA_AVOID: "#d32f2f",
    NavState.REJOIN_PATH: "#1565c0",
}
_STATE_LABELS = {
    NavState.PATH_FOLLOW: "PATH_FOLLOW",
    NavState.DWA_AVOID: "DWA_AVOID",
    NavState.REJOIN_PATH: "REJOIN_PATH",
}


# ======================================================================
# DWA local planner (action-space sampling)
# ======================================================================

@dataclass
class DWAConfig:
    predict_time: float = 5.0
    n_fwd: int = 10
    n_lat: int = 15

    w_obstacle: float = 8.0
    w_goal: float = 1.5
    w_path: float = 1.0
    w_heading: float = 1.0
    w_speed: float = 3.0

    goal_lookahead: float = 4.0


class DWAPlanner:
    """Action-space DWA that simulates the actual pure-pursuit dynamics."""

    def __init__(self, env_cfg: dict, dwa_cfg: DWAConfig | None = None):
        self.env_cfg = env_cfg
        self.c = dwa_cfg or DWAConfig()

        self.max_v = env_cfg["max_v"]
        self.max_omega = env_cfg["max_omega"]
        self.dt = env_cfg["dt"]
        self.collision_dist = env_cfg["collision_dist"]
        self.safety_dist = env_cfg["safety_dist"]

        fwd_lo, fwd_hi = env_cfg["goal_fwd_range"]
        lat_lo, lat_hi = env_cfg["goal_lat_range"]
        self.fwd_samples = np.linspace(max(fwd_lo, 0.3), fwd_hi, self.c.n_fwd)
        self.lat_samples = np.linspace(lat_lo, lat_hi, self.c.n_lat)

        self.pp = PurePursuitController(self.max_v, self.max_omega)

    def _simulate_pursuit(self, rx, ry, rtheta, goal, n_steps):
        traj = np.empty((n_steps + 1, 3))
        traj[0] = [rx, ry, rtheta]
        dt, mv, mw = self.dt, self.max_v, self.max_omega
        for i in range(n_steps):
            v, w = self.pp.compute(rx, ry, rtheta, goal)
            v = min(max(v, 0.0), mv)
            w = min(max(w, -mw), mw)
            rx += v * np.cos(rtheta) * dt
            ry += v * np.sin(rtheta) * dt
            rtheta = wrap_angle(rtheta + w * dt)
            traj[i + 1] = [rx, ry, rtheta]
        return traj

    @staticmethod
    def _predict_human(hx, hy, hvx, hvy, dt, n_steps):
        t = np.arange(n_steps + 1) * dt
        return np.column_stack([hx + hvx * t, hy + hvy * t])

    def _cost_obstacle(self, robot_traj, human_traj):
        n = min(len(robot_traj), len(human_traj))
        dx = robot_traj[:n, 0] - human_traj[:n, 0]
        dy = robot_traj[:n, 1] - human_traj[:n, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        min_d = float(dists.min())
        if min_d <= self.collision_dist + 0.05:
            return float("inf")
        cost = 0.0
        influence = self.safety_dist * 2.5
        if min_d < influence:
            gap = max(min_d - self.collision_dist, 0.01)
            cost = 3.0 / (gap * gap) if gap < 0.4 else 0.5 / gap
        d_first, d_last = float(dists[0]), float(dists[-1])
        if d_last < d_first and d_first < self.safety_dist * 5.0:
            closing = (d_first - d_last) / (n * self.dt + 1e-6)
            cost += closing * 5.0
        return cost

    @staticmethod
    def _cost_goal(xy, goal):
        return float(np.hypot(xy[0] - goal[0], xy[1] - goal[1]))

    @staticmethod
    def _cost_path(robot_traj, path_xy):
        sample = robot_traj[::6, :2]
        dx = sample[:, 0:1] - path_xy[:, 0]
        dy = sample[:, 1:2] - path_xy[:, 1]
        return float(np.sqrt((dx * dx + dy * dy).min(axis=1)).mean())

    @staticmethod
    def _cost_heading(theta, goal_heading):
        return abs(wrap_angle(theta - goal_heading))

    def _cost_speed(self, traj):
        d = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))
        return self.max_v - d.sum() / (len(traj) * self.dt + 1e-9)

    def plan(self, env: LocalPlannerEnv):
        rx, ry, rtheta = env.rx, env.ry, env.rtheta
        path, cur_s = env.path, env.cur_s
        c = self.c
        n_steps = int(c.predict_time / self.dt)
        cr, sr = np.cos(rtheta), np.sin(rtheta)

        eff_la = c.goal_lookahead
        h_traj = None
        if env._human_visible:
            h_traj = self._predict_human(
                env.hx, env.hy, env.hvx, env.hvy, self.dt, n_steps)
            h_rf_x = cr * (env.hx - rx) + sr * (env.hy - ry)
            if h_rf_x > 0:
                eff_la = max(c.goal_lookahead, h_rf_x + 3.0)

        goal_s = min(cur_s + eff_la, path.total_length - 0.5)
        goal_pos = path.position(goal_s)
        goal_hdg = path.heading(goal_s)

        path_px, path_py = path.get_all_xy()
        path_xy = np.column_stack([path_px, path_py])

        best_cost, best_traj, best_act = float("inf"), None, np.array([1.5, 0.0], dtype=np.float32)
        candidates: list[np.ndarray] = []

        for fwd in self.fwd_samples:
            for lat in self.lat_samples:
                gx = rx + fwd * cr - lat * sr
                gy = ry + fwd * sr + lat * cr
                traj = self._simulate_pursuit(rx, ry, rtheta, np.array([gx, gy]), n_steps)

                if h_traj is not None:
                    co = self._cost_obstacle(traj, h_traj)
                    if co == float("inf"):
                        continue
                else:
                    co = 0.0
                candidates.append(traj)

                cost = (c.w_obstacle * co
                        + c.w_goal * self._cost_goal(traj[-1, :2], goal_pos)
                        + c.w_path * self._cost_path(traj, path_xy)
                        + c.w_heading * self._cost_heading(traj[-1, 2], goal_hdg)
                        + c.w_speed * self._cost_speed(traj))
                if cost < best_cost:
                    best_cost, best_traj = cost, traj
                    best_act = np.array([fwd, lat], dtype=np.float32)

        if best_traj is None:
            return np.array([0.0, 0.0], dtype=np.float32), None, candidates
        return best_act, best_traj, candidates


# ======================================================================
# State-machine navigator
# ======================================================================

@dataclass
class SMConfig:
    """Transition thresholds for the state machine."""
    threat_radius: float = 5.0
    threat_exit_radius: float = 6.0        # > threat_radius for hysteresis
    rejoin_lat_thresh: float = 0.35
    rejoin_hdg_thresh: float = 0.25
    path_follow_lookahead_idx: int = 4     # which lookahead point (1-8)
    rejoin_lookahead_idx: int = 2          # closer point for tighter tracking
    rejoin_lat_gain: float = 2.5           # proportional gain on lateral error


class StateMachineNavigator:
    """Three-state navigator: PATH_FOLLOW → DWA_AVOID → REJOIN_PATH."""

    def __init__(self, env_cfg: dict,
                 sm_cfg: SMConfig | None = None,
                 dwa_cfg: DWAConfig | None = None):
        self.env_cfg = env_cfg
        self.sm = sm_cfg or SMConfig()
        self.dwa = DWAPlanner(env_cfg, dwa_cfg)
        self.state = NavState.PATH_FOLLOW

    def reset(self) -> None:
        self.state = NavState.PATH_FOLLOW

    # ----- threat detection -----

    def _human_threat(self, env: LocalPlannerEnv, radius: float) -> bool:
        """True if human is visible, ahead, and within *radius*."""
        if not env._human_visible:
            return False
        cr = np.cos(env.rtheta)
        sr = np.sin(env.rtheta)
        dx, dy = env.hx - env.rx, env.hy - env.ry
        h_rf_x = cr * dx + sr * dy
        if h_rf_x < 0:
            return False
        return np.hypot(dx, dy) < radius

    def _on_path(self, env: LocalPlannerEnv, obs: np.ndarray) -> bool:
        """True if robot is close to the reference path with aligned heading."""
        cfg = env.cfg
        lat_raw = float(obs[2])
        hdg_raw = float(obs[3])
        if cfg.get("normalize_obs", False):
            lat_s, _, _, _, _ = _obs_normalization_scales(cfg)
            lat_raw *= lat_s
            hdg_raw *= np.pi
        return (abs(lat_raw) < self.sm.rejoin_lat_thresh
                and abs(hdg_raw) < self.sm.rejoin_hdg_thresh)

    # ----- per-state action generators -----

    def _act_path_follow(self, obs: np.ndarray, env: LocalPlannerEnv):
        return _obs_to_path_goal(obs, env.cfg,
                                 lookahead_idx=self.sm.path_follow_lookahead_idx)

    def _act_rejoin(self, obs: np.ndarray, env: LocalPlannerEnv):
        action = _obs_to_path_goal(obs, env.cfg,
                                   lookahead_idx=self.sm.rejoin_lookahead_idx)
        lat_raw = float(obs[2])
        if env.cfg.get("normalize_obs", False):
            lat_s, _, _, _, _ = _obs_normalization_scales(env.cfg)
            lat_raw *= lat_s
        action[1] -= self.sm.rejoin_lat_gain * lat_raw
        action[1] = np.clip(action[1], *env.cfg["goal_lat_range"])
        return action

    # ----- main step -----

    def step(self, obs: np.ndarray, env: LocalPlannerEnv):
        """Compute action under the current state, then update state.

        Returns ``(action, state, best_traj, candidates)``.
        ``best_traj`` / ``candidates`` are non-None only in DWA_AVOID.
        """
        sm = self.sm
        best_traj = None
        candidates = None

        # --- transitions ---
        if self.state == NavState.PATH_FOLLOW:
            if self._human_threat(env, sm.threat_radius):
                self.state = NavState.DWA_AVOID

        elif self.state == NavState.DWA_AVOID:
            if not self._human_threat(env, sm.threat_exit_radius):
                self.state = NavState.REJOIN_PATH

        elif self.state == NavState.REJOIN_PATH:
            if self._human_threat(env, sm.threat_radius):
                self.state = NavState.DWA_AVOID
            elif self._on_path(env, obs):
                self.state = NavState.PATH_FOLLOW

        # --- action ---
        if self.state == NavState.PATH_FOLLOW:
            action = self._act_path_follow(obs, env)
        elif self.state == NavState.DWA_AVOID:
            action, best_traj, candidates = self.dwa.plan(env)
        else:
            action = self._act_rejoin(obs, env)

        return action, self.state, best_traj, candidates


# ======================================================================
# Visualization
# ======================================================================

def _overlay(env: LocalPlannerEnv,
             state: NavState,
             best_traj: np.ndarray | None,
             candidates: list[np.ndarray] | None,
             show_candidates: bool = False) -> None:
    if env._ax is None:
        return
    ax = env._ax

    if state == NavState.DWA_AVOID:
        if show_candidates and candidates:
            step = max(1, len(candidates) // 30)
            for t in candidates[::step]:
                ax.plot(t[:, 0], t[:, 1], "c-", lw=0.4, alpha=0.18)
        if best_traj is not None:
            ax.plot(best_traj[:, 0], best_traj[:, 1], "m-", lw=2.5, alpha=0.85)
            ax.plot(best_traj[-1, 0], best_traj[-1, 1], "m*", ms=8)

    label = _STATE_LABELS[state]
    color = _STATE_COLORS[state]
    ax.text(0.02, 0.96, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.85))
    ax.legend(loc="upper right", fontsize=8)

    if env._recording:
        env._fig.canvas.draw()
        frame = np.asarray(env._fig.canvas.buffer_rgba())[..., :3]
        if env._frames:
            env._frames[-1] = frame.copy()


# ======================================================================
# Episode runner
# ======================================================================

@dataclass
class EpisodeResult:
    seed: int = 0
    result: str = ""
    behavior: str = ""
    steps: int = 0
    episode_return: float = 0.0
    collision: bool = False
    success: bool = False
    min_human_dist: float = -1.0
    final_lateral: float = -1.0
    on_path_at_end: bool = False
    human_clear_at_end: bool = False
    state_counts: dict = field(default_factory=dict)


def run_episode(
    env: LocalPlannerEnv,
    nav: StateMachineNavigator,
    seed: int,
    *,
    render: bool = False,
    show_candidates: bool = False,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    nav.reset()
    state_counts = {s: 0 for s in NavState}

    ret, done = 0.0, False
    while not done:
        action, state, best_traj, candidates = nav.step(obs, env)
        state_counts[state] += 1
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc

        if render or env._recording:
            env.render()
            _overlay(env, state, best_traj, candidates, show_candidates)

    tag = _result_tag(info)
    es = info.get("episode_stats", {})

    return EpisodeResult(
        seed=seed,
        result=tag,
        behavior=info.get("behavior", ""),
        steps=info.get("step", 0),
        episode_return=ret,
        collision=es.get("collision", False),
        success=es.get("success", False),
        min_human_dist=es.get("min_human_dist", -1.0),
        final_lateral=es.get("final_abs_lateral", -1.0),
        on_path_at_end=es.get("on_path_at_end", False),
        human_clear_at_end=es.get("human_clear_at_end", False),
        state_counts={_STATE_LABELS[k]: v for k, v in state_counts.items()},
    )


# ======================================================================
# Multi-episode evaluation
# ======================================================================

@dataclass
class EvalSummary:
    num_episodes: int = 0
    collision_rate: float = 0.0
    success_rate: float = 0.0
    timeout_rate: float = 0.0
    other_rate: float = 0.0
    avg_return: float = 0.0
    avg_steps: float = 0.0
    avg_min_human_dist: float = 0.0
    avg_final_lateral: float = 0.0
    episodes: list[EpisodeResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "collision_rate": self.collision_rate,
            "success_rate": self.success_rate,
            "timeout_rate": self.timeout_rate,
            "other_rate": self.other_rate,
            "avg_return": self.avg_return,
            "avg_steps": self.avg_steps,
            "avg_min_human_dist": self.avg_min_human_dist,
            "avg_final_lateral": self.avg_final_lateral,
            "episodes": [
                {
                    "seed": ep.seed,
                    "result": ep.result,
                    "behavior": ep.behavior,
                    "steps": ep.steps,
                    "return": ep.episode_return,
                    "collision": ep.collision,
                    "success": ep.success,
                    "min_human_dist": ep.min_human_dist,
                    "final_lateral": ep.final_lateral,
                    "state_counts": ep.state_counts,
                }
                for ep in self.episodes
            ],
        }


EVAL_CFG = {
    "p_ambient_human": 0.0,
    "encounter_t_range": (2.0, 3.5),
    "encounter_jitter": (0.92, 1.05),
    "human_delay": 1.0,
    "human_from_below_prob": 0.5,
}


def evaluate(
    episodes: int = 10,
    seed_offset: int = 3000,
    render: bool = False,
    save_video: str | None = None,
    show_candidates: bool = False,
    dwa_cfg: DWAConfig | None = None,
    sm_cfg: SMConfig | None = None,
    env_cfg: dict | None = None,
) -> EvalSummary:
    cfg = dict(EVAL_CFG)
    if env_cfg:
        cfg.update(env_cfg)

    need_frames = render or (save_video is not None)
    env = LocalPlannerEnv(
        config=cfg,
        render_mode="human" if need_frames else None,
    )
    nav = StateMachineNavigator(env.cfg, sm_cfg, dwa_cfg)

    results: list[EpisodeResult] = []
    t0 = time.time()

    for ep in range(episodes):
        seed = seed_offset + ep
        if save_video:
            env.start_recording()

        res = run_episode(env, nav, seed,
                          render=render, show_candidates=show_candidates)
        results.append(res)

        sc = res.state_counts
        extra = ""
        if res.min_human_dist >= 0:
            extra = (
                f"  min_d={res.min_human_dist:.2f}"
                f"  |lat|={res.final_lateral:.2f}"
                f"  on_path={res.on_path_at_end}"
                f"  h_clear={res.human_clear_at_end}"
            )
        print(
            f"  ep {ep + 1:>3d}/{episodes}:  {res.result:<12s}"
            f"  steps={res.steps:>3d}  return={res.episode_return:>7.1f}"
            f"  [{res.behavior}]"
            f"  F={sc.get('PATH_FOLLOW',0)} A={sc.get('DWA_AVOID',0)}"
            f" R={sc.get('REJOIN_PATH',0)}{extra}"
        )

        if save_video:
            vid_dir = f"./Evaluation_video/{save_video}"
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = f"{vid_dir}/eval_ep{ep+1}_{res.result.lower().replace(' ','_')}.gif"
            env.stop_recording(vid_path)

    env.close()
    elapsed = time.time() - t0

    n = len(results)
    n_coll = sum(r.collision for r in results)
    n_succ = sum(r.success for r in results)
    n_timeout = sum(r.result == "TIMEOUT" for r in results)
    n_other = n - n_coll - n_succ - n_timeout
    valid_d = [r.min_human_dist for r in results if r.min_human_dist >= 0]
    valid_l = [r.final_lateral for r in results if r.final_lateral >= 0]

    summary = EvalSummary(
        num_episodes=n,
        collision_rate=n_coll / n,
        success_rate=n_succ / n,
        timeout_rate=n_timeout / n,
        other_rate=n_other / n,
        avg_return=float(np.mean([r.episode_return for r in results])),
        avg_steps=float(np.mean([r.steps for r in results])),
        avg_min_human_dist=float(np.mean(valid_d)) if valid_d else -1.0,
        avg_final_lateral=float(np.mean(valid_l)) if valid_l else -1.0,
        episodes=results,
    )

    print(f"\n{'=' * 65}")
    print(f"State-Machine DWA Evaluation  ({n} episodes, {elapsed:.1f}s)")
    print(f"{'=' * 65}")
    print(f"  Success rate    : {summary.success_rate*100:5.1f}%  ({n_succ}/{n})")
    print(f"  Collision rate  : {summary.collision_rate*100:5.1f}%  ({n_coll}/{n})")
    print(f"  Timeout rate    : {summary.timeout_rate*100:5.1f}%  ({n_timeout}/{n})")
    print(f"  Other rate      : {summary.other_rate*100:5.1f}%  ({n_other}/{n})")
    print(f"  Avg return      : {summary.avg_return:7.1f}")
    print(f"  Avg steps       : {summary.avg_steps:7.1f}")
    if summary.avg_min_human_dist >= 0:
        print(f"  Avg min human d : {summary.avg_min_human_dist:7.3f}")
    if summary.avg_final_lateral >= 0:
        print(f"  Avg final |lat| : {summary.avg_final_lateral:7.3f}")
    print(f"{'=' * 65}")

    return summary


# ======================================================================
# CLI
# ======================================================================

def main():
    pa = argparse.ArgumentParser(description="State-machine DWA baseline")
    pa.add_argument("--episodes", type=int, default=10)
    pa.add_argument("--seed-offset", type=int, default=3000)
    pa.add_argument("--render", action="store_true")
    pa.add_argument("--save-video", type=str, default=None, metavar="NAME")
    pa.add_argument("--show-candidates", action="store_true")

    pa.add_argument("--predict-time", type=float, default=3.0)
    pa.add_argument("--n-fwd", type=int, default=10)
    pa.add_argument("--n-lat", type=int, default=15)
    pa.add_argument("--w-obstacle", type=float, default=20.0)
    pa.add_argument("--w-goal", type=float, default=1.5)
    pa.add_argument("--w-path", type=float, default=1.0)
    pa.add_argument("--w-heading", type=float, default=1.0)
    pa.add_argument("--w-speed", type=float, default=1.0)
    pa.add_argument("--goal-lookahead", type=float, default=4.0)
    pa.add_argument("--threat-radius", type=float, default=5.0)

    args = pa.parse_args()

    dwa_cfg = DWAConfig(
        predict_time=args.predict_time, n_fwd=args.n_fwd, n_lat=args.n_lat,
        w_obstacle=args.w_obstacle, w_goal=args.w_goal, w_path=args.w_path,
        w_heading=args.w_heading, w_speed=args.w_speed,
        goal_lookahead=args.goal_lookahead,
    )
    sm_cfg = SMConfig(threat_radius=args.threat_radius,
                      threat_exit_radius=args.threat_radius + 1.0)

    return evaluate(
        episodes=args.episodes, seed_offset=args.seed_offset,
        render=args.render, save_video=args.save_video,
        show_candidates=args.show_candidates,
        dwa_cfg=dwa_cfg, sm_cfg=sm_cfg,
    )


if __name__ == "__main__":
    main()
