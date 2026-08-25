"""Closed-loop runtime: Steps A-D + Step E (V0 or V1) + command execution.

Per simulation frame:

    A prediction -> B safety inflation -> C affected AMR -> D conflict cluster
        -> E local replanning for every cluster (this package)
        -> command tracking (the AMRs actually move)

Cluster members are handed to Step E and leave the QR-cell protocol until they
rejoin their reference path; non-member AMRs keep running the centralized QR
planner and hold outside a locked busy area, exactly as in the rule baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from Centralized_Local_Planner.tools.affected_amr import (
    AMR, CentralizedPlanner, ConflictChecker, ConflictResult,
    T_REPLAN_DEFAULT, V_AMR_TYPICAL_DEFAULT, amr_human_collision,
)
from Centralized_Local_Planner.tools.conflict_cluster import ConflictClusterBuilder
from Centralized_Local_Planner.tools.factory_map import GOALS, MAP_BOUNDS, OBSTACLES
from Centralized_Local_Planner.tools.geometry import point_in_polygon, safety_tube_polygon
from Centralized_Local_Planner.tools.prediction import IntentParticlePredictor, PredictorConfig
from Centralized_Local_Planner.tools.safety_inflation import (
    SafetyInflationConfig, SafetyInflationModel,
)
from Centralized_Local_Planner.tools.scenario import make_amrs, make_workers

from .cluster_state import ClusterAgent, MapData, project_onto_path, worker_predictions_from_step_b
from .commands import MODE_STOP


@dataclass
class ControlState:
    """Step-E kinematic state of one AMR while the local planner owns it."""
    s: float
    lat: float
    speed: float
    heading: float
    accel: float = 0.0
    omega: float = 0.0
    goal_s: float = 0.0
    start_frame: int = 0
    cluster_id: int = -1


@dataclass
class FrameLog:
    frame: int
    n_clusters: int = 0
    n_controlled: int = 0
    plan_time_ms: float = 0.0
    shield_modified: int = 0
    candidates: int = 0
    stops: int = 0
    unsafe: int = 0
    min_worker_clearance: float = float("inf")
    min_amr_distance: float = float("inf")
    route_deviation: float = 0.0
    commands: dict = field(default_factory=dict)


class StepERuntime:
    """A->D pipeline with a pluggable Step-E replanner and command execution."""

    def __init__(self, planner, config, num_frames: int = 420, num_workers: int = 2,
                 num_amrs: int = 6, seed: int = 7, worker_frames=None,
                 amr_safety_dist: float = 1.10, human_collision_dist: float = 0.55):
        self.planner = planner
        self.cfg = config
        self.num_frames = int(num_frames)
        self.human_collision_dist = float(human_collision_dist)

        self.pcfg = PredictorConfig(seed=seed)
        self.dt = self.pcfg.dt
        self.horizon_T = self.pcfg.horizon_steps
        self.safety = SafetyInflationModel(SafetyInflationConfig())
        self.workers = make_workers(num_frames, self.dt, num_workers)
        self.predictors = [IntentParticlePredictor(GOALS, OBSTACLES, self.pcfg,
                                                   rng=np.random.default_rng(seed + 17 * i))
                           for i in range(len(self.workers))]
        self.amrs = make_amrs(num_amrs)
        self.qr_planner = CentralizedPlanner(amr_safety_dist=amr_safety_dist, dt=self.dt)
        self.cluster_builder = ConflictClusterBuilder()
        self.map_data = MapData.from_factory(OBSTACLES, MAP_BOUNDS)

        self._wf = worker_frames                 # optional precomputed Step A/B
        self.control: dict[str, ControlState] = {}
        self.locks: list[dict] = []
        self.logs: list[FrameLog] = []
        self.newly_formed: list[dict] = []
        self.last_results: dict[str, ConflictResult] = {}
        self.last_worker_predictions = None
        self._next_cluster_id = 0

    # -- Steps A + B --------------------------------------------------------
    def worker_data(self, frame: int) -> list[dict]:
        if self._wf is not None:
            return self._wf[frame]
        out = []
        for w, predictor in zip(self.workers, self.predictors):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            pred = predictor.rollout(obs)
            hard, soft, _ = self.safety.inflate_all(
                pred["ellipses"], self.dt, pred["belief"], mean_traj=pred["mean"])
            out.append(dict(name=w["name"], color=w["color"], inflated=hard, soft=soft,
                            centers=pred["ellipses"][:, :2], ellipses=pred["ellipses"],
                            tube=safety_tube_polygon(hard)))
        return out

    def worker_positions(self, frame: int) -> list[np.ndarray]:
        return [(w["truth"][frame] if frame < len(w["truth"]) else w["truth"][-1])
                for w in self.workers]

    # -- Step E bookkeeping -------------------------------------------------
    def _exit_s(self, amr: AMR, hull: np.ndarray) -> float:
        """First arc-length downstream where the reference path leaves the hull."""
        s = amr.progress
        last_inside = s
        while s <= amr.total_length:
            if point_in_polygon(amr.position_at(s), hull):
                last_inside = s
                s += 0.10
            else:
                break
        return float(min(last_inside + self.cfg.goal_lookahead, amr.total_length))

    def _agent(self, amr: AMR, result: ConflictResult | None, st: ControlState) -> ClusterAgent:
        ttc = result.ttc if (result is not None) else float("inf")
        affected = bool(result is not None and result.status in ("REPLAN", "SLOWDOWN"))
        # Task priority: how far the AMR still is from finishing its mission
        # (a nearly finished transport is worth protecting). Braking risk grows
        # with speed -- stopping a fast AMR costs the most throughput.
        task_priority = float(np.clip(amr.progress / max(amr.total_length, 1e-9), 0.0, 1.0))
        braking_risk = float(np.clip(st.speed / max(self.cfg.v_max, 1e-9), 0.0, 1.0))
        return ClusterAgent(id=amr.name, ref=amr, s=st.s, lat=st.lat, speed=st.speed,
                            heading=st.heading, accel=st.accel, omega=st.omega,
                            goal_s=st.goal_s, ttc=ttc, task_priority=task_priority,
                            braking_risk=braking_risk, affected=affected,
                            radius=self.cfg.amr_radius, v_cap=amr.commanded_speed,
                            meta=dict(amr=amr))

    def _enter_control(self, amr: AMR, hull: np.ndarray, frame: int, cluster_id: int) -> None:
        s = float(amr.progress)
        self.control[amr.name] = ControlState(
            s=s, lat=0.0, speed=float(amr.actual_speed), heading=amr.heading_at(s),
            goal_s=self._exit_s(amr, hull), start_frame=frame, cluster_id=cluster_id)
        amr.enter_local(self.control[amr.name].goal_s)

    def _release(self, amr: AMR) -> None:
        st = self.control.pop(amr.name, None)
        if st is not None:
            amr.exit_s = st.goal_s
        amr.resume_rail()

    # -- command execution ---------------------------------------------------
    def _execute(self, amr: AMR, st: ControlState, cmd, dt: float) -> None:
        """Track the commanded trajectory for one simulation frame."""
        traj = cmd.trajectory
        pos, heading, speed, s_new = traj.state_at(dt)
        i = int(np.clip(np.searchsorted(traj.times, dt), 0, len(traj.times) - 1))
        lat_new = float(traj.lateral[i])
        # The tracked speed must keep its SIGN: it seeds the next re-plan, and
        # a magnitude-only state makes every reverse command restart from a
        # forward speed, so the AMR never actually backs off.
        v_long = (float(s_new) - st.s) / max(dt, 1e-9)
        st.accel = (v_long - st.speed) / max(dt, 1e-9)
        dh = (heading - st.heading + math.pi) % (2 * math.pi) - math.pi
        st.omega = dh / max(dt, 1e-9)
        st.speed, st.heading, st.s, st.lat = v_long, float(heading), float(s_new), lat_new
        amr.xy = np.asarray(pos, dtype=float)
        amr.actual_speed = abs(float(speed))
        amr.granted_s = max(amr.granted_s, st.s)

    # -- main loop -----------------------------------------------------------
    def step(self, frame: int) -> dict:
        cfg = self.cfg
        wd = self.worker_data(frame)
        wp = worker_predictions_from_step_b(wd, dt=self.dt)
        self.last_worker_predictions = wp
        wpos = self.worker_positions(frame)
        log = FrameLog(frame=frame)

        rail = [a for a in self.amrs if a.is_active(frame) and a.name not in self.control]

        # ---- Steps C + D on the rail-following AMRs ----
        self.qr_planner.resolve(rail, frame)
        results: dict[str, ConflictResult] = {}
        for a in rail:
            results[a.name] = ConflictChecker.check(
                a, self.dt, self.horizon_T, wd, t_replan=T_REPLAN_DEFAULT,
                v_amr_typical=V_AMR_TYPICAL_DEFAULT)
        clusters = self.cluster_builder.build(rail, results, frame, self.dt, self.horizon_T)
        self.last_results = results

        # ---- new clusters take over their members ----
        self.newly_formed = []
        locked = {n for lk in self.locks for n in lk["members"]}
        for cl in clusters.clusters:
            if any(n in locked for n in cl.member_names):
                continue
            members = [a for a in self.amrs if a.name in cl.member_names]
            cid = self._next_cluster_id
            self._next_cluster_id += 1
            for a in members:
                self._enter_control(a, cl.hull, frame, cid)
            self.locks.append(dict(id=cid, members=set(cl.member_names),
                                   hull=cl.hull, color=cl.color))
            locked.update(cl.member_names)
            self.newly_formed.append(dict(frame=frame, cluster_id=cid, hull=cl.hull,
                                          members=[a.name for a in members]))

        # ---- Step E: plan + execute, one cluster at a time ----
        log.n_clusters = len(self.locks)
        for lk in self.locks:
            members = [a for a in self.amrs
                       if a.name in lk["members"] and a.name in self.control and not a.collided]
            if not members:
                continue
            cluster = [self._agent(a, results.get(a.name), self.control[a.name])
                       for a in members]
            commands = self.planner.plan(cluster, wp, self.map_data, dt=cfg.dt)
            st = self.planner.stats
            log.n_controlled += len(members)
            log.plan_time_ms += st.plan_time_ms
            log.shield_modified += st.n_shield_modified
            log.candidates += st.n_candidates
            log.unsafe += st.n_unsafe
            log.min_worker_clearance = min(log.min_worker_clearance, st.min_worker_clearance)
            log.min_amr_distance = min(log.min_amr_distance, st.min_amr_distance)
            for a in members:
                cmd = commands[a.name]
                log.commands[a.name] = cmd
                log.stops += int(cmd.mode == MODE_STOP)
                self._execute(a, self.control[a.name], cmd, self.dt)
                log.route_deviation += abs(self.control[a.name].lat)

        # ---- collisions + rejoin ----
        for a in list(self.amrs):
            if a.name not in self.control or a.collided:
                continue
            st = self.control[a.name]
            p = a.current_xy()
            for w, q in zip(self.workers, wpos):
                if float(np.linalg.norm(p - q)) < self.human_collision_dist:
                    a.mark_collision(frame, w["name"])
                    self.control.pop(a.name, None)
                    break
            if a.collided:
                continue
            reached = (st.s >= st.goal_s - cfg.reach_tol and abs(st.lat) < 0.25
                       and st.speed >= 0.0)
            timeout = frame - st.start_frame > cfg.max_control_frames
            if reached or timeout:
                a.progress = float(min(st.s, a.total_length))
                self._release(a)

        # ---- release finished busy areas ----
        self.locks = [lk for lk in self.locks
                      if any(n in self.control for n in lk["members"])]

        # ---- rail AMRs advance, holding outside locked busy areas ----
        for a in rail:
            if not a.is_active(frame) or a.name in self.control:
                continue
            nxt = a.position_at(min(a.progress + a.actual_speed * self.dt, a.granted_s))
            if self._blocked_by_lock(nxt, a.name):
                a.actual_speed = 0.0
                a.waiting_for = "busy-area"
                continue
            hit, hw = amr_human_collision(a, frame, self.workers, self.human_collision_dist)
            if hit:
                a.mark_collision(frame, hw)
            else:
                a.step(self.dt)

        self.logs.append(log)
        return dict(worker_data=wd, results=results, clusters=clusters,
                    locks=list(self.locks), log=log)

    def current_cluster_agents(self) -> list:
        """ClusterAgent view of every AMR currently under Step-E control."""
        wp = getattr(self, "last_worker_predictions", None)
        if wp is None:
            self.last_worker_predictions = worker_predictions_from_step_b(
                self.worker_data(0), dt=self.dt)
        out = []
        for amr in self.amrs:
            st = self.control.get(amr.name)
            if st is None or amr.collided:
                continue
            out.append(self._agent(amr, self.last_results.get(amr.name), st))
        return out

    def _blocked_by_lock(self, xy: np.ndarray, name: str) -> bool:
        for lk in self.locks:
            if name in lk["members"]:
                continue
            if point_in_polygon(xy, lk["hull"]):
                return True
        return False

    # -- metrics -------------------------------------------------------------
    def run(self, frames: int | None = None) -> dict:
        n = self.num_frames if frames is None else int(frames)
        min_clear = float("inf")
        for f in range(n):
            self.step(f)
            for a in self.amrs:
                if a.is_spawned(f) and not a.collided and not a.is_done():
                    p = a.current_xy()
                    for q in self.worker_positions(f):
                        min_clear = min(min_clear, float(np.linalg.norm(p - q)))
        return self.metrics(min_clear)

    def metrics(self, min_clear: float = float("inf")) -> dict:
        controlled = [l for l in self.logs if l.n_controlled > 0]
        n_ctrl = sum(l.n_controlled for l in controlled)
        return dict(
            completion=float(np.mean([a.is_done() for a in self.amrs])),
            worker_collisions=int(sum(a.collided for a in self.amrs)),
            progress=float(np.mean([a.progress / a.total_length for a in self.amrs])),
            min_clearance=float(min_clear if np.isfinite(min_clear) else 0.0),
            control_frames=int(n_ctrl),
            n_clusters=int(self._next_cluster_id),
            stop_ratio=float(sum(l.stops for l in controlled) / max(n_ctrl, 1)),
            shield_rate=float(sum(l.shield_modified for l in controlled) / max(n_ctrl, 1)),
            unsafe_rate=float(sum(l.unsafe for l in controlled) / max(n_ctrl, 1)),
            route_deviation=float(sum(l.route_deviation for l in controlled) / max(n_ctrl, 1)),
            plan_ms=float(np.mean([l.plan_time_ms for l in controlled]) if controlled else 0.0),
            candidates_per_amr=float(sum(l.candidates for l in controlled) / max(n_ctrl, 1)),
            min_worker_clearance=float(min([l.min_worker_clearance for l in controlled],
                                           default=float("inf"))),
        )
