"""Trunk / orchestrator for the Centralized Local Path Planning framework.

This is the single entry point that wires the ``tools`` (pure algorithm logic)
together and dispatches the ``viz`` demos.

Two ways to use it
------------------
1. Render a demo (CLI)::

       python -m Centralized_Local_Planner.main predict   [--preview] [--frames N] [--workers K]
       python -m Centralized_Local_Planner.main safety
       python -m Centralized_Local_Planner.main affected  [--amrs M] [--inject-stray]
       python -m Centralized_Local_Planner.main cluster   [--amrs M]
       python -m Centralized_Local_Planner.main pipeline            # render all four

2. Run the pipeline programmatically (the clean hook for Step E -- RL + local
   replanning). ``Pipeline.step(frame)`` runs A->B->C->D for one frame and
   returns the structured results (worker safety tubes, per-AMR conflict
   status, conflict clusters + local replanning region) without any rendering::

       from Centralized_Local_Planner.main import Pipeline
       pipe = Pipeline(num_frames=280, num_workers=2, num_amrs=6)
       for f in range(pipe.num_frames):
           out = pipe.step(f)            # -> dict(worker_data, results, clusters)
           # Step E (V0/V1) consumes out["clusters"] + out["results"] here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .tools.prediction import PredictorConfig, IntentParticlePredictor
from .tools.safety_inflation import SafetyInflationConfig, SafetyInflationModel
from .tools.geometry import safety_tube_polygon
from .tools.affected_amr import (
    CentralizedPlanner,
    ConflictChecker,
    ConflictResult,
    amr_human_collision,
    T_REPLAN_DEFAULT,
    V_AMR_TYPICAL_DEFAULT,
)
from .tools.conflict_cluster import ClusterResult, ConflictClusterBuilder
from .tools.factory_map import GOALS, OBSTACLES
from .tools.scenario import make_workers, make_amrs


def _outputs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


# ===========================================================================
# Programmatic pipeline  (A -> B -> C -> D)   --  reusable hook for Step E
# ===========================================================================


class Pipeline:
    """Headless A->B->C->D pipeline driver.

    Mirrors exactly the per-frame wiring used by the demos, but returns plain
    data structures instead of drawing. Intended as the integration point for
    Step E (TTC-priority sequential replanning shield / attention policy).
    """

    def __init__(
        self,
        num_frames: int = 280,
        num_workers: int = 2,
        num_amrs: int = 6,
        seed: int = 7,
        amr_safety_dist: float = 1.10,
        human_collision_dist: float = 0.55,
        t_replan: float = T_REPLAN_DEFAULT,
        v_amr_typical: float = V_AMR_TYPICAL_DEFAULT,
        cluster_spatial_dist: float = 2.5,
        cascade_dist: float = 2.0,
        replan_region_buffer: float = 1.8,
        replanner=None,
    ):
        self.replanner = replanner
        self.num_frames = num_frames
        self.human_collision_dist = human_collision_dist
        self.t_replan = t_replan
        self.v_amr_typical = v_amr_typical

        self.cfg = PredictorConfig(seed=seed)
        self.dt = self.cfg.dt
        self.horizon_T = self.cfg.horizon_steps

        safety_cfg = SafetyInflationConfig()
        self.safety = SafetyInflationModel(safety_cfg)

        self.workers = make_workers(num_frames, self.dt, num_workers)
        self.predictors = [
            IntentParticlePredictor(
                GOALS, OBSTACLES, self.cfg,
                rng=np.random.default_rng(seed + 17 * i),
            )
            for i in range(len(self.workers))
        ]
        self.amrs = make_amrs(num_amrs)
        self.planner = CentralizedPlanner(amr_safety_dist=amr_safety_dist, dt=self.dt)
        self.cluster_builder = ConflictClusterBuilder(
            cluster_spatial_dist=cluster_spatial_dist,
            cascade_dist=cascade_dist,
            replan_region_buffer=replan_region_buffer,
        )

    def step(self, frame: int, advance: bool = True) -> dict:
        """Run one frame of A->B->C->D. Returns worker_data, results, clusters."""
        # ----- Step A: worker prediction  +  Step B: safety inflation -----
        worker_data: list[dict] = []
        for w, predictor in zip(self.workers, self.predictors):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            hard_lobes, soft_lobes, _ = self.safety.inflate_all(
                out["ellipses"], self.dt, out["belief"], mean_traj=out["mean"],
            )
            worker_data.append(dict(
                name=w["name"], color=w["color"],
                inflated=hard_lobes, soft=soft_lobes,
                centers=out["ellipses"][:, :2],
                tube=safety_tube_polygon(hard_lobes),
            ))

        # ----- Step C: planner + per-AMR conflict check -----
        self.planner.resolve(self.amrs, frame)
        results: dict[str, ConflictResult] = {}
        for amr in self.amrs:
            if not amr.is_active(frame):
                continue
            hit, hw = amr_human_collision(
                amr, frame, self.workers, self.human_collision_dist,
            )
            if hit:
                amr.mark_collision(frame, hw)
                continue
            results[amr.name] = ConflictChecker.check(
                amr, self.dt, self.horizon_T, worker_data,
                t_replan=self.t_replan, v_amr_typical=self.v_amr_typical,
            )

        # ----- Step D: conflict cluster construction -----
        clusters: ClusterResult = self.cluster_builder.build(
            self.amrs, results, frame, self.dt, self.horizon_T,
        )

        # ----- Step E (V0): TTC-priority replanning + space-time shield -----
        if self.replanner is not None:
            self.replanner.plan(
                self.amrs, results, worker_data, frame, self.dt, self.horizon_T,
            )

        # ----- advance AMRs along their rails -----
        if advance:
            for amr in self.amrs:
                if amr.is_active(frame):
                    amr.step(self.dt)

        return dict(worker_data=worker_data, results=results, clusters=clusters)


# ===========================================================================
# CLI dispatch
# ===========================================================================


def _run_demo(step: str, args) -> None:
    preview = args.preview or args.no_video
    out = None if preview else Path(args.output)

    if step == "predict":
        from .viz.render_prediction import build_animation
        saved = build_animation(
            output_path=out, num_frames=args.frames,
            num_workers=args.workers, preview=preview, seed=args.seed,
        )
    elif step == "safety":
        from .viz.render_safety import build_safety_animation
        saved = build_safety_animation(
            output_path=out, num_frames=args.frames,
            num_workers=args.workers, preview=preview, seed=args.seed,
        )
    elif step == "affected":
        from .viz.render_affected import build_step_c_animation
        saved = build_step_c_animation(
            output_path=out, num_frames=args.frames, num_workers=args.workers,
            num_amrs=args.amrs, preview=preview, seed=args.seed,
            inject_stray=args.inject_stray,
        )
    elif step == "cluster":
        from .viz.render_clusters import build_step_d_animation
        saved = build_step_d_animation(
            output_path=out, num_frames=args.frames, num_workers=args.workers,
            num_amrs=args.amrs, preview=preview, seed=args.seed,
        )
    elif step == "replan":
        from .viz.render_replanning import build_replanning_animation
        saved = build_replanning_animation(
            output_path=out, num_frames=args.frames, num_workers=args.workers,
            num_amrs=args.amrs, preview=preview, seed=args.seed,
        )
    else:
        raise ValueError(step)

    if saved is not None:
        print(f"Saved {step} demo to {saved}")


_DEFAULT_OUTPUT = {
    "predict":  "worker_prediction_demo.mp4",
    "safety":   "safety_inflation_demo.mp4",
    "affected": "step_c_affected_amr_demo.mp4",
    "cluster":  "step_d_conflict_cluster_demo.mp4",
    "replan":   "step_e_v0_replanning_demo.mp4",
}
_DEFAULT_FRAMES = {"predict": 80, "safety": 280, "affected": 280,
                   "cluster": 280, "replan": 360}


def _add_common(sp, step: str) -> None:
    sp.add_argument("--output", type=str,
                    default=str(_outputs_dir() / _DEFAULT_OUTPUT[step]))
    sp.add_argument("--frames", type=int, default=_DEFAULT_FRAMES[step])
    sp.add_argument("--workers", type=int, default=2, choices=[1, 2, 3, 4])
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--preview", action="store_true",
                    help="open a live matplotlib window instead of rendering a file")
    sp.add_argument("--no-video", action="store_true", help="alias for --preview")
    if step in ("affected", "cluster", "replan"):
        sp.add_argument("--amrs", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])
    if step == "affected":
        sp.add_argument("--inject-stray", action="store_true",
                        help="enable the engineered 'Loader' collision demo")


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = pa.add_subparsers(dest="step", required=True)
    for step in ("predict", "safety", "affected", "cluster", "replan"):
        _add_common(sub.add_parser(step, help=f"render the {step} demo"), step)
    sub.add_parser("pipeline", help="render all four step demos in sequence")

    args = pa.parse_args(argv)

    if args.step == "pipeline":
        for step in ("predict", "safety", "affected", "cluster"):
            ns = argparse.Namespace(
                output=str(_outputs_dir() / _DEFAULT_OUTPUT[step]),
                frames=_DEFAULT_FRAMES[step], workers=2, seed=7,
                preview=False, no_video=False, amrs=6, inject_stray=False,
            )
            print(f"--- rendering {step} ---")
            _run_demo(step, ns)
        return

    _run_demo(args.step, args)


if __name__ == "__main__":
    main()
