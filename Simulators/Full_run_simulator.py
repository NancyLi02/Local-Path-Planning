"""
Full Run Simulator

Use this file to test a trained PPO / SAC policy on a long route.

Logic:
    - Robot follows reference path with pure path-following.
    - Human may become visible in observation earlier.
    - RL policy takes over only when the human enters a tighter trigger range.
    - Corridor membership is used as risk flag (risk=1 in corridor).
    - After the human leaves risk region and robot returns to path, path-following resumes.
    - Ends when robot reaches goal.

Examples:

1. No RL model:
    python3 Simulators/Full_run_simulator.py

2. Test PPO:
    python3 Simulators/Full_run_simulator.py --model models/ppo.zip --algo ppo

3. Test SAC:
    python3 Simulators/Full_run_simulator.py --model models/sac.zip --algo sac

4. Save video + report (creates Evaluation_video/full_run/<run_name>/ with gif + report.md):
    python3 Simulators/Full_run_simulator.py --model models/sac.zip --algo sac --save-video run.gif
    python3 Simulators/Full_run_simulator.py --model logs/SAC/sac_v2/sac_v2.zip --algo sac --save-video run.gif

    Random seed generation:
    python3 Simulators/Full_run_simulator.py \
  --model logs/SAC/sac_v7/sac_v7.zip \
  --algo sac \
  --seed $RANDOM \
  --save-video fr_sacv7_1.gif

5. Longer route:
    python3 Simulators/Full_run_simulator.py --model logs/SAC/sac_v4/sac_v4.zip --algo sac --path-length 80 --save-video fr_sacv4_t2.gif

6. Validate blocking encounter test suite (25 cases):
    python3 Simulators/Full_run_simulator.py --validate-encounters
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Simulators.Single_robot_simulator.path import (
    ReferencePath,
    obs_normalization_scales,
    wrap_angle,
)
from Simulators.Single_robot_simulator.controller import PurePursuitController
from Simulators.Single_robot_simulator.policies import obs_to_path_goal


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_full_run_video_dir() -> str:
    return os.path.join(_project_root(), "Evaluation_video", "full_run")


def _resolve_run_output_paths(save_video: str) -> tuple[str, str, str]:
    """Return (run_dir, video_path, report_path) for a full-run output.

    Each run gets its own subfolder under Evaluation_video/full_run/ (or the
    parent of an absolute save_video path) containing the video and report.md.
    """
    base_name = os.path.basename(save_video) or "full_run.gif"
    run_name = os.path.splitext(base_name)[0] or "full_run"

    if os.path.isabs(save_video):
        parent = os.path.dirname(os.path.abspath(save_video)) or os.getcwd()
    else:
        parent = os.path.abspath(_default_full_run_video_dir())

    run_dir = os.path.join(parent, run_name)
    if os.path.exists(run_dir):
        n = 1
        while os.path.exists(f"{run_dir}_{n:03d}"):
            n += 1
        run_dir = f"{run_dir}_{n:03d}"

    video_path = os.path.join(run_dir, base_name)
    report_path = os.path.join(run_dir, "report.md")
    return run_dir, video_path, report_path


def _elapsed_seconds(steps: int, dt: float) -> float:
    return float(steps) * float(dt)


_FULL_RUN_DEFAULT_CFG: dict = dict(
    path_length=50.0,
    path_y=5.0,
    robot_radius=0.3,
    max_v=1.0,
    max_omega=1.0,
    dt=0.1,
    init_v=0.6,
    max_steps=1200,
    human_radius=0.3,
    # Kept at the single-robot training simulator's upper bound so the obs velocity
    # normalization stays identical to training (vel_s = max_v + human_speed_range[1]
    # = 2.0). Do NOT lower this just to slow humans down -- it would rescale the
    # human-velocity observation channel away from what the policy was trained on.
    # It only drives normalization now; actual speeds come from the walk/jog bands below.
    human_speed_range=(0.3, 1.0),
    # Actual pedestrian crossing speed. Most humans walk (below the robot's ~0.7 cruise);
    # a minority jog (up to the robot's max_v = 1.0). So a human is usually slower than
    # the AMR and only occasionally as fast as it. Decoupled from human_speed_range so
    # obs normalization is unaffected.
    human_walk_speed_range=(0.3, 0.6),
    human_jog_speed_range=(0.7, 1.0),
    human_jog_prob=0.2,
    collision_dist=0.7,
    safety_dist=1.5,
    corridor_len=8.0,
    corridor_w=1.8,
    # Primary, distance-based replanning trigger shared with the single-robot
    # training simulator: SAC takes over once the human is within this straight-line
    # distance and hands back once it leaves. Keep this equal to the single-robot
    # env's human_detect_radius so kick-in behaviour is consistent across both sims.
    human_detect_radius=7.0,
    # Legacy forward/lateral trigger box, used only as a fallback if
    # human_detect_radius is set to None.
    rl_trigger_forward_dist=7.0,
    rl_trigger_lateral_dist=2.2,
    rl_trigger_behind_dist=0.5,
    goal_fwd_range=(0.0, 3.0),
    goal_lat_range=(-2.0, 2.0),
    n_lookahead=8,
    lookahead_spacing=1.0,
    normalize_obs=True,
    success_lat_thresh=0.05,
    success_hdg_thresh=0.1,
    oob_margin=5.0,
    encounters=None,
    scenario_robot_speed=0.7,
    cross_appear_dist_default=7.0,
    cross_time_scale=0.78,
    cross_start_dist_min=2.0,
    cross_angle_jitter=0.45,
    encounter_spawn_min_ahead=5.0,
    encounter_s_start_margin=10.0,
    encounter_s_end_margin=8.0,
    encounter_min_spacing=8.0,
    encounter_count=6,
    random_encounters=True,
    human_despawn_delay=15,
    hallway_half_width=3.0,
    view_ahead=10.0,
    view_behind=6.0,
    view_half_height=6.0,
    map_size=20.0,
)


def _reference_robot_speed(cfg: dict) -> float:
    """Nominal cruise speed used to time pedestrian crossings."""
    if "scenario_robot_speed" in cfg:
        return float(cfg["scenario_robot_speed"])
    return float((cfg.get("init_v", 0.6) + cfg.get("max_v", 1.0)) * 0.5)


def build_blocking_encounter(
    s: float,
    speed: float,
    side: float,
    cfg: dict | None = None,
    *,
    cross_angle: float | None = None,
    appear_distance: float | None = None,
    time_scale: float | None = None,
) -> dict:
    """Build a pedestrian crossing timed to meet the robot on the reference path."""
    c = dict(_FULL_RUN_DEFAULT_CFG)
    if cfg:
        c.update(cfg)

    appear = float(
        appear_distance
        if appear_distance is not None
        else c.get("cross_appear_dist_default", 7.0)
    )

    enc: dict = {
        "s": float(s),
        "behavior": "cross",
        "speed": float(speed),
        "side": float(side),
        "appear_distance": appear,
    }
    if cross_angle is not None:
        enc["cross_angle"] = float(cross_angle)
    if time_scale is not None:
        enc["time_scale"] = float(time_scale)
    return enc


def simulate_path_only_encounter(
    encounter: dict,
    config: dict | None = None,
    *,
    seed: int = 0,
    lookahead_idx: int = 3,
) -> dict:
    """Run one encounter with pure path-following to check if it blocks the robot."""
    cfg = dict(_FULL_RUN_DEFAULT_CFG)
    if config:
        cfg.update(config)
    cfg["random_encounters"] = False
    cfg["encounters"] = [encounter]
    cfg["cross_angle_jitter"] = 0.0

    env = FullRunEnv(config=cfg)
    obs, _ = env.reset(seed=seed)
    done = False
    min_dist_corridor = float("inf")
    human_in_corridor = False
    crosses_in_front = False

    while not done:
        action = obs_to_path_goal(obs, env.cfg, lookahead_idx=lookahead_idx)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if env._human_visible and env._in_corridor(env.hx, env.hy):
            human_in_corridor = True
            dist = float(np.hypot(env.rx - env.hx, env.ry - env.hy))
            min_dist_corridor = min(min_dist_corridor, dist)
            cr, sr = np.cos(env.rtheta), np.sin(env.rtheta)
            h_ahead = cr * (env.hx - env.rx) + sr * (env.hy - env.ry)
            if h_ahead > 0.0:
                crosses_in_front = True

        enc_s = float(encounter["s"])
        if (
            info.get("encounter_collision")
            or info.get("encounter_success")
            or info.get("encounter_skipped")
            or info.get("encounter_despawned")
            or env.cur_s > enc_s + 6.0
        ):
            done = True

    collision = any(r["result"] == "collision" for r in env._encounter_results)
    collision_dist = float(cfg["collision_dist"])
    blocking = collision or (
        human_in_corridor and min_dist_corridor < collision_dist
    )
    return {
        "blocking": blocking,
        "collision": collision,
        "human_in_corridor": human_in_corridor,
        "crosses_in_front": crosses_in_front,
        "min_dist_corridor": min_dist_corridor,
        "encounter_results": list(env._encounter_results),
    }


def validate_blocking_encounter_case(case: dict) -> dict:
    """Validate a single blocking-encounter test case."""
    cfg = {"path_length": float(case.get("path_length", 50.0))}
    if "config" in case:
        cfg.update(case["config"])

    enc = case.get("encounter")
    if enc is None:
        draft = build_blocking_encounter(
            case["s"],
            case["speed"],
            case["side"],
            cfg,
            cross_angle=case.get("cross_angle"),
        )
        enc = _tune_blocking_encounter(
            draft, cfg, seed=int(case.get("seed", 0)),
        )
        if enc is None:
            enc = draft

    result = simulate_path_only_encounter(
        enc,
        cfg,
        seed=int(case.get("seed", 0)),
    )
    passed = bool(
        result["blocking"]
        and result["human_in_corridor"]
        and result["crosses_in_front"]
        and result["min_dist_corridor"] < float(cfg.get("collision_dist", 0.7))
    )
    return {
        "id": case.get("id"),
        "name": case.get("name", f"case_{case.get('id', '?')}"),
        "passed": passed,
        "encounter": enc,
        **result,
    }


# 25 benchmark cases with varied position, speed, side, and crossing angle.
BLOCKING_ENCOUNTER_TEST_CASES: list[dict] = [
    {"id": 1, "name": "mid_slow_left", "s": 14.0, "speed": 0.20, "side": 1.0, "cross_angle": -0.25, "seed": 1},
    {"id": 2, "name": "mid_slow_right", "s": 14.0, "speed": 0.20, "side": -1.0, "cross_angle": 0.30, "seed": 2},
    {"id": 3, "name": "mid_med_left", "s": 16.0, "speed": 0.25, "side": 1.0, "cross_angle": 0.10, "seed": 3},
    {"id": 4, "name": "mid_med_right", "s": 16.0, "speed": 0.25, "side": -1.0, "cross_angle": -0.15, "seed": 4},
    {"id": 5, "name": "mid_fast_left", "s": 18.0, "speed": 0.30, "side": 1.0, "cross_angle": 0.35, "seed": 5},
    {"id": 6, "name": "mid_fast_right", "s": 18.0, "speed": 0.30, "side": -1.0, "cross_angle": -0.35, "seed": 6},
    {"id": 7, "name": "early_slow_left", "s": 12.0, "speed": 0.18, "side": 1.0, "cross_angle": 0.20, "seed": 7},
    {"id": 8, "name": "early_slow_right", "s": 12.0, "speed": 0.18, "side": -1.0, "cross_angle": -0.20, "seed": 8},
    {"id": 9, "name": "late_slow_left", "s": 22.0, "speed": 0.18, "side": 1.0, "cross_angle": -0.10, "seed": 9},
    {"id": 10, "name": "late_slow_right", "s": 22.0, "speed": 0.18, "side": -1.0, "cross_angle": 0.25, "seed": 10},
    {"id": 11, "name": "late_med_left", "s": 24.0, "speed": 0.25, "side": 1.0, "cross_angle": 0.15, "seed": 11},
    {"id": 12, "name": "late_med_right", "s": 24.0, "speed": 0.25, "side": -1.0, "cross_angle": -0.30, "seed": 12},
    {"id": 13, "name": "late_fast_left", "s": 26.0, "speed": 0.32, "side": 1.0, "cross_angle": 0.40, "seed": 13},
    {"id": 14, "name": "late_fast_right", "s": 26.0, "speed": 0.32, "side": -1.0, "cross_angle": -0.40, "seed": 14},
    {"id": 15, "name": "deep_slow_left", "s": 30.0, "speed": 0.20, "side": 1.0, "cross_angle": 0.05, "seed": 15},
    {"id": 16, "name": "deep_slow_right", "s": 30.0, "speed": 0.20, "side": -1.0, "cross_angle": -0.05, "seed": 16},
    {"id": 17, "name": "deep_med_left", "s": 32.0, "speed": 0.27, "side": 1.0, "cross_angle": 0.28, "seed": 17},
    {"id": 18, "name": "deep_med_right", "s": 32.0, "speed": 0.27, "side": -1.0, "cross_angle": -0.28, "seed": 18},
    {"id": 19, "name": "deep_fast_left", "s": 34.0, "speed": 0.33, "side": 1.0, "cross_angle": 0.12, "seed": 19},
    {"id": 20, "name": "deep_fast_right", "s": 34.0, "speed": 0.33, "side": -1.0, "cross_angle": -0.12, "seed": 20},
    {"id": 21, "name": "far_slow_left", "s": 38.0, "speed": 0.22, "side": 1.0, "cross_angle": -0.22, "seed": 21},
    {"id": 22, "name": "far_slow_right", "s": 38.0, "speed": 0.22, "side": -1.0, "cross_angle": 0.22, "seed": 22},
    {"id": 23, "name": "far_med_left", "s": 40.0, "speed": 0.28, "side": 1.0, "cross_angle": 0.18, "seed": 23},
    {"id": 24, "name": "far_med_right", "s": 40.0, "speed": 0.28, "side": -1.0, "cross_angle": -0.18, "seed": 24},
    {"id": 25, "name": "far_fast_right", "s": 42.0, "speed": 0.35, "side": -1.0, "cross_angle": 0.32, "seed": 25},
]


def run_blocking_encounter_tests(
    cases: list[dict] | None = None,
    *,
    verbose: bool = True,
) -> tuple[int, int, list[dict]]:
    """Run blocking-encounter validation suite. Returns (passed, total, results)."""
    cases = cases or BLOCKING_ENCOUNTER_TEST_CASES
    results = [validate_blocking_encounter_case(case) for case in cases]
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    if verbose:
        print(f"Blocking encounter tests: {passed}/{total} passed")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            md = r["min_dist_corridor"]
            md_str = f"{md:.2f}m" if md != float("inf") else "inf"
            print(
                f"  [{status}] #{r['id']:02d} {r['name']}: "
                f"blocking={r['blocking']} corridor={r['human_in_corridor']} "
                f"collision={r['collision']} min_dist={md_str}"
            )
    return passed, total, results


def validate_random_encounter_suite(
    *,
    path_length: float = 80.0,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """Validate randomized encounter generation across multiple seeds."""
    seeds = seeds or [0, 1, 7, 13, 42, 99, 123, 256]
    passed = total = 0
    collision_dist = float(_FULL_RUN_DEFAULT_CFG["collision_dist"])
    for seed in seeds:
        cfg = {"path_length": path_length, "scenario_seed": seed}
        rng = np.random.default_rng(seed)
        try:
            encounters = build_random_encounters(path_length, cfg, rng)
        except RuntimeError as exc:
            if verbose:
                print(f"  [FAIL] seed={seed}: {exc}")
            total += int(_FULL_RUN_DEFAULT_CFG.get("encounter_count", 6))
            continue
        for i, enc in enumerate(encounters):
            total += 1
            result = simulate_path_only_encounter(
                enc, cfg, seed=seed + i * 31,
            )
            ok = (
                result["blocking"]
                and result["human_in_corridor"]
                and result["crosses_in_front"]
                and result["min_dist_corridor"] < collision_dist
            )
            if ok:
                passed += 1
            elif verbose:
                print(
                    f"  [FAIL] seed={seed} enc={i + 1}: "
                    f"min={result['min_dist_corridor']:.2f}"
                )
    if verbose:
        print(f"Random encounter suite: {passed}/{total} passed")
    return passed, total


def _spaced_crossing_positions(
    path_length: float,
    cfg: dict,
    n: int = 6,
) -> list[float]:
    """Evenly space n crossing positions along the path with minimum spacing."""
    start = float(cfg.get("encounter_s_start_margin", 10.0))
    end = float(cfg.get("encounter_s_end_margin", 8.0))
    min_spacing = float(cfg.get("encounter_min_spacing", 8.0))
    s_lo, s_hi = start, path_length - end
    usable = s_hi - s_lo
    if usable < min_spacing or n < 1:
        return []

    max_fit = max(1, int(usable // min_spacing) + 1)
    count = min(n, max_fit)
    if count == 1:
        return [s_lo + usable * 0.5]

    inner = usable - (count - 1) * min_spacing
    segment = max(inner / count, 0.0)
    positions: list[float] = []
    cursor = s_lo
    for i in range(count):
        offset = segment * (0.35 + 0.3 * ((i * 5 + 2) % 7) / 6.0)
        positions.append(cursor + offset)
        cursor += offset + min_spacing
    return positions


def _tune_blocking_encounter(
    enc: dict,
    cfg: dict,
    *,
    seed: int = 0,
) -> dict | None:
    """Search timing parameters so path-following would collide with the human."""
    collision_dist = float(cfg.get("collision_dist", 0.7))
    for appear in np.arange(4.5, 10.5, 0.5):
        for scale in np.arange(0.68, 0.88, 0.02):
            candidate = {
                **enc,
                "appear_distance": float(appear),
                "time_scale": float(scale),
            }
            result = simulate_path_only_encounter(candidate, cfg, seed=seed)
            if (
                result["blocking"]
                and result["human_in_corridor"]
                and result["crosses_in_front"]
                and result["min_dist_corridor"] < collision_dist
            ):
                return candidate
    return None


def _random_crossing_positions(
    rng,
    path_length: float,
    cfg: dict,
    n: int,
) -> list[float]:
    """Spaced crossing positions with random jitter inside each segment."""
    start = float(cfg.get("encounter_s_start_margin", 10.0))
    end = float(cfg.get("encounter_s_end_margin", 8.0))
    min_spacing = float(cfg.get("encounter_min_spacing", 8.0))
    s_lo, s_hi = start, path_length - end
    usable = s_hi - s_lo
    if usable < min_spacing or n < 1:
        return []

    max_fit = max(1, int(usable // min_spacing) + 1)
    count = min(n, max_fit)
    if count == 1:
        return [float(s_lo + usable * 0.5)]

    inner = usable - (count - 1) * min_spacing
    segment = max(inner / count, 0.0)
    positions: list[float] = []
    cursor = s_lo
    for _ in range(count):
        offset = float(rng.uniform(0.15, 0.85) * segment) if segment > 0 else 0.0
        positions.append(cursor + offset)
        cursor += offset + min_spacing
    return positions


def _sample_walkjog_speed(rng, cfg: dict) -> float:
    """Sample a pedestrian speed: usually a walk, occasionally a jog.

    With probability ``human_jog_prob`` the speed is drawn from ``human_jog_speed_range``
    (the faster, can-match-the-AMR case); otherwise from ``human_walk_speed_range`` (the
    common, slower-than-the-AMR case). Falls back to ``human_speed_range`` if the walk/jog
    bands are absent.
    """
    walk = cfg.get("human_walk_speed_range")
    jog = cfg.get("human_jog_speed_range")
    if walk is None or jog is None:
        lo, hi = cfg.get("human_speed_range", (0.1, 0.3))
        return float(rng.uniform(lo, hi))
    jog_prob = float(cfg.get("human_jog_prob", 0.2))
    band = jog if rng.random() < jog_prob else walk
    return float(rng.uniform(float(band[0]), float(band[1])))


def build_random_encounters(
    path_length: float,
    cfg: dict | None,
    rng,
    *,
    n: int | None = None,
    validate: bool = True,
) -> list[dict]:
    """Build varied, seed-driven encounters that all block path-following."""
    c = dict(_FULL_RUN_DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    c["path_length"] = path_length

    count = int(n if n is not None else c.get("encounter_count", 6))
    angle_max = float(c.get("cross_angle_jitter", 0.45))
    collision_dist = float(c.get("collision_dist", 0.7))

    positions = _random_crossing_positions(rng, path_length, c, count)
    encounters: list[dict] = []
    base_seed = int(rng.integers(0, 2**31 - 1)) if hasattr(rng, "integers") else 0

    for i, s in enumerate(positions):
        tuned: dict | None = None
        for attempt in range(24):
            speed = _sample_walkjog_speed(rng, c)
            side = float(rng.choice([1.0, -1.0]))
            cross_angle = float(rng.uniform(-angle_max, angle_max))
            draft = build_blocking_encounter(
                s, speed, side, c,
                cross_angle=cross_angle,
            )
            if validate:
                tuned = _tune_blocking_encounter(
                    draft, c, seed=base_seed + i * 97 + attempt,
                )
                if tuned is not None:
                    break
            else:
                tuned = draft
                break

        if tuned is None:
            raise RuntimeError(
                f"Could not tune blocking encounter {i + 1} at s={s:.1f} "
                f"(path_length={path_length})"
            )
        encounters.append(tuned)

    return encounters


def _format_encounter_summary(encounters: list[dict]) -> str:
    lines = [f"Encounters: {len(encounters)} (randomized crossings)"]
    for i, enc in enumerate(encounters):
        ang = enc.get("cross_angle", 0.0)
        lines.append(
            f"  Human {i + 1}: s={enc['s']:.1f}m  speed={enc['speed']:.2f}  "
            f"side={'L' if enc.get('side', 1) > 0 else 'R'}  "
            f"angle={np.degrees(ang):+.0f}°  "
            f"appear={enc.get('appear_distance', 0):.1f}m"
        )
    return "\n".join(lines)


def build_default_encounters(path_length: float, cfg: dict | None = None) -> list[dict]:
    """Build encounters for a full run (randomized when random_encounters=True)."""
    c = dict(_FULL_RUN_DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    c["path_length"] = path_length

    if c.get("random_encounters", True):
        seed = int(c.get("scenario_seed", 42))
        rng = np.random.default_rng(seed)
        return build_random_encounters(path_length, c, rng)

    angle_max = float(c.get("cross_angle_jitter", 0.45))
    positions = _spaced_crossing_positions(path_length, c, n=int(c.get("encounter_count", 6)))
    encounters: list[dict] = []
    rng = np.random.default_rng(int(c.get("scenario_seed", 42)))
    for s in positions:
        draft = build_blocking_encounter(
            s,
            float(rng.uniform(0.18, 0.32)),
            float(rng.choice([1.0, -1.0])),
            c,
            cross_angle=float(rng.uniform(-angle_max, angle_max)),
        )
        tuned = _tune_blocking_encounter(draft, c, seed=int(rng.integers(0, 10000)))
        if tuned is None:
            tuned = {**draft, "appear_distance": 7.0, "time_scale": 0.78}
        encounters.append(tuned)
    return encounters


def _format_min_dist(value: float) -> str:
    if value == float("inf"):
        return "N/A"
    return f"{value:.2f}"


def _write_run_report(
    report_path: str,
    *,
    steps: int,
    dt: float,
    goal_reached: bool,
    encounter_results: list[dict],
    path_length: float,
    policy_mode: str,
    seed: int,
    video_path: str | None = None,
) -> str:
    elapsed_s = _elapsed_seconds(steps, dt)
    n_col = sum(1 for r in encounter_results if r["result"] == "collision")
    n_suc = sum(1 for r in encounter_results if r["result"] == "success")
    status = "Goal reached" if goal_reached else "Incomplete"

    lines = [
        "# Full Run Report",
        "",
        "## Summary",
        "",
        f"- **Status:** {status}",
        f"- **Total time elapsed:** {elapsed_s:.1f} s",
        f"- **Steps:** {steps}",
        f"- **Path length:** {path_length:.1f} m",
        f"- **Policy:** {policy_mode}",
        f"- **Seed:** {seed}",
        f"- **Encounters:** {len(encounter_results)}",
        f"- **Collisions:** {n_col}",
        f"- **Avoided:** {n_suc}",
        "",
    ]
    if video_path:
        lines.extend([
            f"- **Video:** `{os.path.basename(video_path)}`",
            "",
        ])

    lines.extend([
        "## Encounter Results",
        "",
        "| Encounter | Result | Min distance (m) | Path-follow min (m) |",
        "|-----------|--------|------------------|---------------------|",
    ])
    for r in encounter_results:
        idx = r["idx"] + 1
        result = r["result"].replace("_", " ").title()
        pf_min = r.get(
            "min_dist_path_following",
            r.get("min_dist_to_path", float("inf")),
        )
        lines.append(
            f"| {idx} | {result} | {_format_min_dist(r['min_dist'])} | "
            f"{_format_min_dist(pf_min)} |"
        )

    if not encounter_results:
        lines.append("| — | — | — | — |")

    lines.extend([
        "",
        "*Path-follow min* is the closest human–robot distance if the robot had "
        "not replanned (distance to the ghost that continues on the reference path).",
        "",
    ])
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return report_path


# ======================================================================
# Environment
# ======================================================================


class FullRunEnv(gym.Env):
    """Gymnasium environment for a continuous start-to-goal run with encounters.

    The robot follows a long straight reference path. Predefined human
    encounters trigger avoidance episodes along the way. Each episode ends
    on collision or successful return to path, then the robot continues.
    The environment terminates only when the robot reaches the goal.

    Observation and action formats match LocalPlannerEnv for model compatibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DEFAULT_CFG: dict = _FULL_RUN_DEFAULT_CFG

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = dict(self.DEFAULT_CFG)
        if config:
            self.cfg.update(config)
        if (
            not self.cfg.get("random_encounters", True)
            and self.cfg.get("encounters") is None
        ):
            self.cfg["encounters"] = build_default_encounters(
                float(self.cfg["path_length"]),
                self.cfg,
            )
        self.render_mode = render_mode

        c = self.cfg
        n_lk = c["n_lookahead"]
        obs_dim = 1 + 3 + 2 * n_lk + 4 + 1

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([c["goal_fwd_range"][0], c["goal_lat_range"][0]], dtype=np.float32),
            high=np.array([c["goal_fwd_range"][1], c["goal_lat_range"][1]], dtype=np.float32),
        )

        self.controller = PurePursuitController(c["max_v"], c["max_omega"])

        self.path: ReferencePath | None = None
        self.rx = self.ry = self.rtheta = self.rv = 0.0
        self.hx = self.hy = self.hvx = self.hvy = 0.0
        self.cur_s = 0.0
        self._ghost_rx = self._ghost_ry = self._ghost_rtheta = self._ghost_rv = 0.0
        self._ghost_cur_s = 0.0
        self._ghost_visible = False
        self.steps = 0

        self._human_visible = False
        self._encounter_idx = 0
        self._encounter_active = False
        self._encounter_resolved = False
        self._encounter_results: list[dict] = []
        self._despawn_counter = 0
        self._last_cross_s = 0.0
        self._active_cross_s = 0.0
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._h_behav = ""
        self._prev_abs_lat = 0.0

        self._rtraj: list[np.ndarray] = []
        self._gtraj: list[np.ndarray] = []
        self._htraj: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []

        # Overlay text for encounter results in video
        self._overlay_text = ""
        self._overlay_color = "#333333"
        self._overlay_ttl = 0

        self._fig = None
        self._ax = None
        self._recording = False
        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def stop_recording(self, path: str = "full_run.mp4", fps: int = 10) -> str | None:
        self._recording = False
        if not self._frames:
            print("No frames to save.")
            return None

        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis("off")
        im = ax.imshow(self._frames[0])

        def _update(i):
            im.set_data(self._frames[i])
            return [im]

        ani = FuncAnimation(
            fig, _update, frames=len(self._frames),
            interval=1000 // fps, blit=True,
        )

        saved_path: str | None = None
        if path.endswith(".gif"):
            from matplotlib.animation import PillowWriter

            ani.save(path, writer=PillowWriter(fps=fps))
            saved_path = path
        else:
            try:
                from matplotlib.animation import FFMpegWriter

                ani.save(path, writer=FFMpegWriter(fps=fps))
                saved_path = path
            except Exception:
                gif_path = path.rsplit(".", 1)[0] + ".gif"
                try:
                    from matplotlib.animation import PillowWriter

                    ani.save(gif_path, writer=PillowWriter(fps=fps))
                    saved_path = gif_path
                    print(f"ffmpeg unavailable, saved as {saved_path}")
                except Exception as e:
                    print(f"Failed to save video: {e}")

        n_frames = len(self._frames)
        plt.close(fig)
        self._frames = []

        if saved_path:
            print(f"Video saved -> {saved_path}  ({n_frames} frames)")
        return saved_path

    # ------------------------------------------------------------------
    # Path creation
    # ------------------------------------------------------------------

    def _make_straight_path(self) -> None:
        c = self.cfg
        length = c["path_length"]
        y = c["path_y"]
        n_pts = max(10, int(length / 2))
        xs = np.linspace(0, length, n_pts)
        ys = np.full_like(xs, y)
        self.path = ReferencePath(
            np.column_stack([xs, ys]),
            num_samples=max(1000, int(length * 20)),
        )

    # ------------------------------------------------------------------
    # Human encounter spawning
    # ------------------------------------------------------------------

    def _spawn_encounter_human(self, enc: dict, cross_s: float | None = None) -> None:
        """Spawn a pedestrian timed to cross the reference path ahead of the robot."""
        c = self.cfg
        s_enc = float(cross_s if cross_s is not None else enc["s"])
        speed = float(enc.get("speed", 0.2))
        side = float(enc.get("side", 1.0))

        enc_pos = self.path.position(s_enc)
        nrm = self.path.normal(s_enc)

        if "cross_angle" in enc:
            ang = float(enc["cross_angle"])
        else:
            jitter_max = float(c.get("cross_angle_jitter", 0.45))
            ang = float(self.np_random.uniform(-jitter_max, jitter_max))

        gap = max(s_enc - self.cur_s, 0.5)
        v_r = max(self.rv, c.get("init_v", 0.6))
        time_scale = float(enc.get("time_scale", c.get("cross_time_scale", 0.78)))
        t_enc = gap / max(v_r, 0.1) * time_scale
        start_min = float(c.get("cross_start_dist_min", 2.0))
        start_dist = max(speed * t_enc, start_min)

        ca, sa = np.cos(ang), np.sin(ang)
        d_base = -side * nrm
        d = np.array([
            d_base[0] * ca - d_base[1] * sa,
            d_base[0] * sa + d_base[1] * ca,
        ])
        d = d / max(float(np.linalg.norm(d)), 1e-6)

        start = enc_pos - d * start_dist
        self.hvx, self.hvy = float(d[0] * speed), float(d[1] * speed)
        self.hx, self.hy = float(start[0]), float(start[1])
        self._active_cross_s = s_enc
        self._h_behav = enc.get("behavior", "cross")
        self._human_visible = True
        self._encounter_active = False
        self._encounter_resolved = False
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._ghost_visible = False
        self._htraj = []

    def _maybe_spawn_next_human(self) -> None:
        c = self.cfg
        encounters = c.get("encounters", [])
        if self._human_visible or self._encounter_idx >= len(encounters):
            return

        min_ahead = float(c.get("encounter_spawn_min_ahead", 5.0))
        end_margin = float(c.get("encounter_s_end_margin", 8.0))
        min_spacing = float(c.get("encounter_min_spacing", 8.0))
        start_margin = float(c.get("encounter_s_start_margin", 10.0))

        if self.cur_s + min_ahead > self.path.total_length - end_margin:
            return

        enc = encounters[self._encounter_idx]
        s_nominal = float(enc["s"])

        if self._encounter_idx == 0:
            min_cross_s = start_margin
        else:
            min_cross_s = self._last_cross_s + min_spacing

        cross_s = max(s_nominal, self.cur_s + min_ahead, min_cross_s)
        cross_s = min(cross_s, self.path.total_length - end_margin)

        appear_dist = float(
            enc.get("appear_distance", c.get("cross_appear_dist_default", 7.0))
        )

        if self.cur_s >= cross_s - appear_dist:
            self._spawn_encounter_human(enc, cross_s=cross_s)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _in_corridor(self, px: float, py: float) -> bool:
        c = self.cfg
        s_lo = self.cur_s
        s_hi = min(self.cur_s + c["corridor_len"], self.path.total_length)
        mid = (s_lo + s_hi) / 2
        rad = (s_hi - s_lo) / 2 + 1.0
        s_cl, _, dist = self.path.closest_point(px, py, s_hint=mid, search_radius=rad)
        if s_cl < s_lo - 0.3 or s_cl > s_hi + 0.3:
            return False
        return dist < c["corridor_w"]

    def _update_progress(self) -> None:
        s_new, _, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        self.cur_s = max(self.cur_s, s_new)

    # ------------------------------------------------------------------
    # Observation (same format as LocalPlannerEnv)
    # ------------------------------------------------------------------

    def _obs_at(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        rv: float,
        cur_s: float,
    ) -> np.ndarray:
        """Observation as if the robot were at the given state (shared human state)."""
        saved = (self.rx, self.ry, self.rtheta, self.rv, self.cur_s)
        self.rx, self.ry, self.rtheta, self.rv, self.cur_s = (
            rx, ry, rtheta, rv, cur_s,
        )
        obs = self._obs()
        self.rx, self.ry, self.rtheta, self.rv, self.cur_s = saved
        return obs

    def _step_path_follower(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        rv: float,
        cur_s: float,
    ) -> tuple[float, float, float, float, float]:
        """Advance a virtual robot that only follows the reference path."""
        c = self.cfg
        obs = self._obs_at(rx, ry, rtheta, rv, cur_s)
        action = obs_to_path_goal(obs, c, lookahead_idx=3)
        fwd, lat = float(action[0]), float(action[1])

        cr, sr = np.cos(rtheta), np.sin(rtheta)
        goal = np.array([
            rx + fwd * cr - lat * sr,
            ry + fwd * sr + lat * cr,
        ])

        if abs(fwd) < 0.05 and abs(lat) < 0.05:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd = self.controller.compute(rx, ry, rtheta, goal)

        dt = c["dt"]
        v = float(np.clip(v_cmd, 0, c["max_v"]))
        w = float(np.clip(w_cmd, -c["max_omega"], c["max_omega"]))
        rx = rx + v * np.cos(rtheta) * dt
        ry = ry + v * np.sin(rtheta) * dt
        rtheta = wrap_angle(rtheta + w * dt)
        s_new, _, _ = self.path.closest_point(
            rx, ry, s_hint=cur_s, search_radius=5.0,
        )
        cur_s = max(cur_s, s_new)
        return rx, ry, rtheta, v, cur_s

    def _is_on_path(self) -> bool:
        c = self.cfg
        _, lat, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        h_err = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
        return (
            abs(lat) < c["success_lat_thresh"]
            and h_err < c["success_hdg_thresh"]
        )

    def _rl_engaged(self) -> bool:
        """True when the human is detected within sensing range (SAC takes over).

        This is the same condition the demo runner uses to hand control to the
        SAC policy, so it marks the exact moment the model comes into action --
        before the robot starts to deviate from the path or slow down.

        Replanning is gated purely on the straight-line distance to the human
        (``human_detect_radius``), identical to the single-robot training
        simulator, so the kick-in behaviour is consistent across both. The legacy
        forward/lateral trigger box is kept as a fallback when no radius is set.
        """
        if not self._human_visible:
            return False

        c = self.cfg
        radius = c.get("human_detect_radius")
        if radius is not None:
            d_rh = float(np.hypot(self.hx - self.rx, self.hy - self.ry))
            return d_rh <= float(radius)

        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
        dx = self.hx - self.rx
        dy = self.hy - self.ry
        h_forward = cr * dx + sr * dy
        h_lateral = -sr * dx + cr * dy

        return (
            h_forward > -c.get("rl_trigger_behind_dist", 0.5)
            and h_forward < c.get("rl_trigger_forward_dist", 7.0)
            and abs(h_lateral) < c.get("rl_trigger_lateral_dist", 2.2)
        )

    def _spawn_ghost_on_path(self) -> None:
        """Spawn ghost at the on-path position where the SAC policy took over."""
        p = self.path.position(self.cur_s)
        h = self.path.heading(self.cur_s)
        self._ghost_rx = float(p[0])
        self._ghost_ry = float(p[1])
        self._ghost_rtheta = h
        self._ghost_rv = self.rv
        self._ghost_cur_s = self.cur_s
        self._ghost_visible = True
        self._gtraj = [np.array([self._ghost_rx, self._ghost_ry])]

    def _hide_ghost(self) -> None:
        self._ghost_visible = False

    def _update_ghost(self) -> None:
        """Show the no-replan ghost whenever the SAC policy is in control.

        The ghost is the baseline robot that keeps following the straight-line
        path (no replanning). It appears the moment the SAC model comes into
        action -- before the robot starts to deviate or slow down -- so the two
        can be compared, and disappears once SAC hands control back.
        """
        if not self._rl_engaged():
            if self._ghost_visible:
                self._hide_ghost()
            return

        if not self._ghost_visible:
            self._spawn_ghost_on_path()

        self._ghost_rx, self._ghost_ry, self._ghost_rtheta, self._ghost_rv, self._ghost_cur_s = (
            self._step_path_follower(
                self._ghost_rx,
                self._ghost_ry,
                self._ghost_rtheta,
                self._ghost_rv,
                self._ghost_cur_s,
            )
        )
        self._gtraj.append(np.array([self._ghost_rx, self._ghost_ry]))
        d_ghost = float(np.hypot(
            self._ghost_rx - self.hx,
            self._ghost_ry - self.hy,
        ))
        self._ep_min_d_path_following = min(
            self._ep_min_d_path_following, d_ghost,
        )

    def _obs(self) -> np.ndarray:
        c = self.cfg
        s = self.cur_s
        _, lat, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=s, search_radius=5.0,
        )
        h_err = wrap_angle(self.rtheta - self.path.heading(s))
        progress = s / self.path.total_length

        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)

        look: list[float] = []
        for i in range(1, c["n_lookahead"] + 1):
            sa = min(s + i * c["lookahead_spacing"], self.path.total_length)
            p = self.path.position(sa)
            dx, dy = p[0] - self.rx, p[1] - self.ry
            look.extend([cr * dx + sr * dy, -sr * dx + cr * dy])

        if self._human_visible:
            dx, dy = self.hx - self.rx, self.hy - self.ry
            hrx = cr * dx + sr * dy
            hry = -sr * dx + cr * dy
            dvx = self.hvx - self.rv * np.cos(self.rtheta)
            dvy = self.hvy - self.rv * np.sin(self.rtheta)
            hrvx = cr * dvx + sr * dvy
            hrvy = -sr * dvx + cr * dvy
            risk = 1.0 if self._in_corridor(self.hx, self.hy) else 0.0
        else:
            hrx, hry, hrvx, hrvy, risk = 10.0, 0.0, 0.0, 0.0, 0.0

        vec = [self.rv, progress, lat, h_err] + look + [hrx, hry, hrvx, hrvy, risk]
        if c.get("normalize_obs", False):
            lat_s, pos_s, vel_s, ms, mv = obs_normalization_scales(c)
            vec[0] = float(vec[0]) / mv
            vec[2] = float(vec[2]) / lat_s
            vec[3] = float(vec[3]) / np.pi
            for i in range(4, 4 + 2 * c["n_lookahead"], 2):
                vec[i] = float(vec[i]) / pos_s
                vec[i + 1] = float(vec[i + 1]) / pos_s
            hb = 4 + 2 * c["n_lookahead"]
            vec[hb] = float(vec[hb]) / ms
            vec[hb + 1] = float(vec[hb + 1]) / ms
            vec[hb + 2] = float(vec[hb + 2]) / vel_s
            vec[hb + 3] = float(vec[hb + 3]) / vel_s

        return np.asarray(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Encounter management
    # ------------------------------------------------------------------

    def _resolve_encounter(self, result: str) -> dict:
        self._encounter_active = False
        self._encounter_resolved = True
        self._despawn_counter = self.cfg.get("human_despawn_delay", 15)
        rec = {
            "idx": self._encounter_idx,
            "result": result,
            "min_dist": float(self._ep_min_d_human),
            "min_dist_path_following": float(self._ep_min_d_path_following),
        }
        self._encounter_results.append(rec)

        if result == "collision":
            self._overlay_text = "COLLISION!"
            self._overlay_color = "#d32f2f"
            self._overlay_ttl = 20
        elif result == "success":
            self._overlay_text = "AVOIDED"
            self._overlay_color = "#388e3c"
            self._overlay_ttl = 15

        return rec

    def _check_encounter_events(self) -> dict:
        c = self.cfg
        info: dict = {}

        if not self._human_visible:
            return info

        # Handle despawn countdown after encounter resolved
        if self._encounter_resolved:
            self._despawn_counter -= 1
            if self._despawn_counter <= 0:
                self._human_visible = False
                self._last_cross_s = self._active_cross_s
                self._encounter_idx += 1
                info["encounter_despawned"] = True
            return info

        dh = float(np.hypot(self.rx - self.hx, self.ry - self.hy))

        # 1. Collision check (always)
        if dh < c["collision_dist"]:
            rec = self._resolve_encounter("collision")
            info["encounter_collision"] = True
            info["encounter_result"] = rec
            return info

        # 2. Encounter activation: human enters corridor
        if not self._encounter_active:
            if self._in_corridor(self.hx, self.hy):
                self._encounter_active = True
                info["encounter_start"] = True

        # 3. Success check: robot back on path + human clear
        if self._encounter_active:
            on_path = self._is_on_path()

            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            human_behind = h_ahead < 0
            human_far = dh > c["safety_dist"] * 2
            human_clear = (not self._in_corridor(self.hx, self.hy)) and (
                human_far or human_behind
            )

            if on_path and human_clear:
                rec = self._resolve_encounter("success")
                info["encounter_success"] = True
                info["encounter_result"] = rec

        # 4. Skip: human passed without ever entering corridor
        if not self._encounter_active and not self._encounter_resolved:
            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead_dist = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            if h_ahead_dist < -5.0 or dh > 15.0:
                self._human_visible = False
                self._last_cross_s = self._active_cross_s
                self._encounter_idx += 1
                self._encounter_results.append({
                    "idx": self._encounter_idx - 1,
                    "result": "no_conflict",
                    "min_dist": float(self._ep_min_d_human),
                    "min_dist_path_following": float(self._ep_min_d_path_following),
                })
                info["encounter_skipped"] = True

        return info

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._make_straight_path()

        if self.cfg.get("random_encounters", True):
            scfg = dict(self.cfg)
            if seed is not None:
                scfg["scenario_seed"] = int(seed)
            self.cfg["encounters"] = build_random_encounters(
                float(self.cfg["path_length"]),
                scfg,
                self.np_random,
            )

        self.cur_s = 0.5

        p = self.path.position(self.cur_s)
        h = self.path.heading(self.cur_s)
        self.rx, self.ry, self.rtheta = float(p[0]), float(p[1]), h
        self.rv = 0.0

        self._human_visible = False
        self._encounter_idx = 0
        self._encounter_active = False
        self._encounter_resolved = False
        self._encounter_results = []
        self._despawn_counter = 0
        self._last_cross_s = 0.0
        self._active_cross_s = 0.0
        self._h_behav = ""
        self.hx = self.hy = self.hvx = self.hvy = 0.0

        self.steps = 0
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._ghost_visible = False
        self._prev_abs_lat = 0.0
        self._rtraj = [np.array([self.rx, self.ry])]
        self._gtraj = []
        self._htraj = []
        self._goals = []

        self._overlay_text = ""
        self._overlay_ttl = 0

        obs = self._obs()
        info = {
            "phase": "cruise",
            "encounters": list(self.cfg.get("encounters", [])),
        }
        return obs, info

    def step(self, action):
        c = self.cfg
        self.steps += 1

        self._maybe_spawn_next_human()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        fwd, lat = float(action[0]), float(action[1])

        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
        goal = np.array([
            self.rx + fwd * cr - lat * sr,
            self.ry + fwd * sr + lat * cr,
        ])
        self._goals.append(goal.copy())

        if abs(fwd) < 0.05 and abs(lat) < 0.05:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd = self.controller.compute(self.rx, self.ry, self.rtheta, goal)

        dt = c["dt"]
        v = float(np.clip(v_cmd, 0, c["max_v"]))
        w = float(np.clip(w_cmd, -c["max_omega"], c["max_omega"]))
        self.rx += v * np.cos(self.rtheta) * dt
        self.ry += v * np.sin(self.rtheta) * dt
        self.rtheta = wrap_angle(self.rtheta + w * dt)
        self.rv = v

        if self._human_visible:
            self.hx += self.hvx * dt
            self.hy += self.hvy * dt
            self._htraj.append(np.array([self.hx, self.hy]))
            dh_step = float(np.hypot(self.rx - self.hx, self.ry - self.hy))
            self._ep_min_d_human = min(self._ep_min_d_human, dh_step)

        self._rtraj.append(np.array([self.rx, self.ry]))
        self._update_progress()

        self._update_ghost()

        enc_info = self._check_encounter_events()

        terminated = truncated = False
        info = dict(enc_info)
        info["step"] = self.steps

        if self.cur_s >= self.path.total_length - 1.0:
            terminated = True
            info["goal_reached"] = True

        if self.steps >= c["max_steps"]:
            truncated = True
            info["timeout"] = True

        margin = c.get("oob_margin", 5.0)
        path_y = c["path_y"]
        if (
            self.rx < -margin
            or self.rx > c["path_length"] + margin
            or self.ry < path_y - margin * 3
            or self.ry > path_y + margin * 3
        ):
            truncated = True
            info["out_of_bounds"] = True

        if self._encounter_active:
            info["phase"] = "encounter"
        elif self._human_visible:
            info["phase"] = "human_visible"
        else:
            info["phase"] = "cruise"

        if terminated or truncated:
            info["encounter_results"] = list(self._encounter_results)

        reward = 0.0
        obs = self._obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None and not self._recording:
            return None
        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(14, 6))

        ax = self._ax
        ax.clear()
        c = self.cfg

        va = c.get("view_ahead", 10.0)
        vb = c.get("view_behind", 6.0)
        vh = c.get("view_half_height", 6.0)

        x_lo = self.rx - vb
        x_hi = self.rx + va
        y_lo = c["path_y"] - vh
        y_hi = c["path_y"] + vh

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        # Hallway walls and floor
        hw = c.get("hallway_half_width", 3.0)
        wall_top = c["path_y"] + hw
        wall_bot = c["path_y"] - hw
        ax.fill_between(
            [x_lo, x_hi], wall_bot, wall_top,
            color="#f5f0e1", alpha=0.3, zorder=0,
        )
        ax.hlines(
            [wall_top, wall_bot], x_lo, x_hi,
            colors="#8B4513", linewidths=2.5, alpha=0.6, zorder=1,
        )

        # Reference path
        px, py = self.path.get_all_xy()
        ax.plot(px, py, "b--", lw=1.5, alpha=0.4, label="Reference path")

        # Start and goal markers
        p_start = self.path.position(0)
        p_goal = self.path.position(self.path.total_length)
        ax.plot(p_start[0], p_start[1], "s", color="green", ms=12, label="Start", zorder=5)
        ax.plot(p_goal[0], p_goal[1], "*", color="red", ms=16, label="Goal", zorder=5)

        # Corridor detection zone
        s_lo_c = self.cur_s
        s_hi_c = min(self.cur_s + c["corridor_len"], self.path.total_length)
        s_arr = np.linspace(s_lo_c, s_hi_c, 40)
        left, right = [], []
        for sv in s_arr:
            p = self.path.position(sv)
            n = self.path.normal(sv)
            left.append(p + c["corridor_w"] * n)
            right.append(p - c["corridor_w"] * n)
        poly = np.array(left + right[::-1])
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color="orange", label="Corridor")

        # Robot trajectory
        if len(self._rtraj) > 1:
            rt = np.array(self._rtraj)
            ax.plot(rt[:, 0], rt[:, 1], "g-", lw=2, alpha=0.7, label="Robot traj")

        # Ghost (path-following) trajectory — drawn while SAC is in control
        if self._ghost_visible and len(self._gtraj) > 1:
            gt = np.array(self._gtraj)
            ax.plot(
                gt[:, 0], gt[:, 1], color="#1e88e5", ls="--", lw=1.5,
                alpha=0.55, label="No-replan ghost",
            )
        elif len(self._gtraj) > 1:
            gt = np.array(self._gtraj)
            ax.plot(
                gt[:, 0], gt[:, 1], color="#1e88e5", ls="--", lw=1.5,
                alpha=0.35,
            )

        # Human trajectory
        if self._human_visible and len(self._htraj) > 1:
            ht = np.array(self._htraj)
            ax.plot(ht[:, 0], ht[:, 1], "r-", lw=1.5, alpha=0.5, label="Human traj")

        # Robot
        ax.add_patch(plt.Circle(
            (self.rx, self.ry), c["robot_radius"],
            color="green", alpha=0.7, zorder=10,
        ))
        al = 0.5
        ax.arrow(
            self.rx, self.ry,
            al * np.cos(self.rtheta), al * np.sin(self.rtheta),
            head_width=0.15, head_length=0.1, fc="darkgreen", ec="darkgreen",
            zorder=11,
        )
        ax.add_patch(plt.Circle(
            (self.rx, self.ry), c["safety_dist"],
            fill=False, ls="--", color="gold", alpha=0.4, zorder=9,
        ))

        # Ghost robot — visible the whole time the SAC policy is in control
        if self._ghost_visible:
            ax.add_patch(plt.Circle(
                (self._ghost_rx, self._ghost_ry), c["robot_radius"],
                fill=False, ls="-", color="#1e88e5", lw=2.0, alpha=0.85, zorder=9,
            ))
            ax.add_patch(plt.Circle(
                (self._ghost_rx, self._ghost_ry), c["robot_radius"] * 0.55,
                color="#64b5f6", alpha=0.45, zorder=9,
            ))
            ax.arrow(
                self._ghost_rx, self._ghost_ry,
                al * np.cos(self._ghost_rtheta), al * np.sin(self._ghost_rtheta),
                head_width=0.12, head_length=0.08, fc="#1565c0", ec="#1565c0",
                alpha=0.75, zorder=10,
            )

        # Human
        if self._human_visible:
            ax.add_patch(plt.Circle(
                (self.hx, self.hy), c["human_radius"],
                color="red", alpha=0.7, zorder=10,
            ))
            ax.arrow(
                self.hx, self.hy, self.hvx * 0.8, self.hvy * 0.8,
                head_width=0.1, head_length=0.08, fc="darkred", ec="darkred",
                zorder=11,
            )

        # Local goal
        if self._goals:
            g = self._goals[-1]
            ax.plot(g[0], g[1], "mx", ms=12, mew=3, label="Local goal", zorder=8)

        # Encounter position markers
        encounters = c.get("encounters", [])
        for i, enc in enumerate(encounters):
            ep = self.path.position(enc["s"])
            color = "#cccccc"
            if i < len(self._encounter_results):
                res = self._encounter_results[i]["result"]
                color = "#d32f2f" if res == "collision" else (
                    "#388e3c" if res == "success" else "#999999"
                )
            elif i == self._encounter_idx and self._encounter_active:
                color = "#ff9800"
            ax.plot(ep[0], ep[1], "D", ms=8, color=color, alpha=0.7, zorder=6)

        # Overlay text (collision / avoided banners)
        if self._overlay_ttl > 0:
            alpha = min(1.0, self._overlay_ttl / 8.0)
            ax.text(
                0.5, 0.88, self._overlay_text,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=22, fontweight="bold", color="white",
                bbox=dict(
                    boxstyle="round,pad=0.4", fc=self._overlay_color,
                    alpha=0.85 * alpha,
                ),
                zorder=20,
            )
            self._overlay_ttl -= 1

        # Title bar
        phase = "ENCOUNTER" if self._encounter_active else "CRUISE"
        if self._encounter_resolved and self._human_visible:
            phase = "RESOLVED"
        progress_pct = (self.cur_s / self.path.total_length) * 100
        n_done = len(self._encounter_results)
        n_total = len(encounters)
        elapsed_s = self.steps * c["dt"]
        title = (
            f"t={elapsed_s:.1f}s  |  Step {self.steps}  |  v={self.rv:.2f} m/s  |  "
            f"{phase}  |  Progress: {progress_pct:.0f}%  |  "
            f"Encounters: {n_done}/{n_total}"
        )
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

        if self.render_mode == "human":
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            try:
                plt.pause(0.01)
            except Exception:
                pass

        if self._recording or self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            frame = np.asarray(self._fig.canvas.buffer_rgba())[..., :3]
            if self._recording:
                self._frames.append(frame.copy())
            if self.render_mode == "rgb_array":
                return frame

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = None


# ======================================================================
# Demo runner
# ======================================================================


def run_full_demo(
    rl_model=None,
    config: dict | None = None,
    seed: int = 42,
    render: bool = True,
    save_video: str | None = None,
    policy_mode: str = "hybrid",
) -> dict:
    """Run a complete start-to-goal demonstration.

    Args:
        rl_model: Trained RL model (e.g. from stable-baselines3).
                  If None, uses pure path-following.
        config: Optional config dict overrides.
        seed: Random seed.
        render: Whether to display live rendering.
        save_video: Path to save video file (e.g. "run.gif"). Creates a per-run
            subfolder under Evaluation_video/full_run/<run_name>/ containing
            the video and report.md; absolute paths use the same layout under
            the given path's parent directory.
        policy_mode: "hybrid" | "path" | "stop_corridor".
            - hybrid: HybridPolicy (requires rl_model)
            - path: pure path following
            - stop_corridor: stop when human is in corridor, resume after leaving

    Returns:
        Dict with run summary (steps, goal_reached, encounter_results).
    """
    env = FullRunEnv(config=config, render_mode="human" if render else None)
    report_path: str | None = None
    if save_video:
        _run_dir, save_video, report_path = _resolve_run_output_paths(save_video)
        os.makedirs(_run_dir, exist_ok=True)
        env.start_recording()

    obs, info = env.reset(seed=seed)

    policy = None
    mode = str(policy_mode).lower()
    if mode == "hybrid":
        if rl_model is not None:
            policy = rl_model
        else:
            print("policy=hybrid requires --model; fallback to policy=path")
            mode = "path"
    elif mode not in {"path", "stop_corridor"}:
        raise ValueError(f"Unknown policy_mode: {policy_mode}")

    n_enc = len(env.cfg.get("encounters", []))
    print(
        f"Starting full run: path_length={env.cfg['path_length']:.0f}m, "
        f"encounters={n_enc}"
    )
    print(_format_encounter_summary(env.cfg.get("encounters", [])))

    def should_use_rl(env) -> bool:
        return env._rl_engaged()

    done = False
    stopped_last_step = False
    while not done:
        if policy is not None:
            if should_use_rl(env):
                action, state = rl_model.predict(obs, deterministic=True)
            else:
                action = obs_to_path_goal(obs, env.cfg, lookahead_idx=3)
        elif mode == "stop_corridor":
            human_in_corridor = (
                env._human_visible and env._in_corridor(env.hx, env.hy)
            )
            if human_in_corridor:
                action = np.array([0.0, 0.0], dtype=np.float32)
                if not stopped_last_step:
                    print("  Stop policy: human entered corridor -> robot stop")
                stopped_last_step = True
            else:
                action = obs_to_path_goal(obs, env.cfg, lookahead_idx=3)
                if stopped_last_step:
                    print("  Stop policy: human left corridor -> robot resume")
                stopped_last_step = False
        else:
            action = obs_to_path_goal(obs, env.cfg, lookahead_idx=3)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

        if "encounter_start" in info:
            idx = env._encounter_idx
            print(f"  Encounter {idx + 1} started! ({env._h_behav})")
        if "encounter_collision" in info:
            print(f"  Encounter ended: COLLISION")
        if "encounter_success" in info:
            print(f"  Encounter ended: AVOIDED")
        if "encounter_skipped" in info:
            print(f"  Encounter skipped: no conflict")

    # Final result
    elapsed_s = _elapsed_seconds(info["step"], env.cfg["dt"])
    if info.get("goal_reached"):
        print(f"\nGoal reached in {info['step']} steps ({elapsed_s:.1f} s)")
    elif info.get("timeout"):
        print(f"\nTimeout at {info['step']} steps ({elapsed_s:.1f} s)")
    elif info.get("out_of_bounds"):
        print(f"\nOut of bounds at {info['step']} steps ({elapsed_s:.1f} s)")

    results = info.get("encounter_results", [])
    print(f"\n--- Final Report (total time: {elapsed_s:.1f} s) ---")
    for r in results:
        status = "COLLISION" if r["result"] == "collision" else r["result"].upper()
        min_d = r["min_dist"]
        pf_min = r.get("min_dist_path_following", r.get("min_dist_to_path", float("inf")))
        min_str = _format_min_dist(min_d)
        path_str = _format_min_dist(pf_min)
        print(
            f"  Encounter {r['idx'] + 1}: {status} "
            f"(min_dist={min_str}, path_follow_min={path_str})"
        )

    # Final overlay on the rendering
    if render or env._recording:
        env.render()
        ax = env._ax
        if ax is not None:
            tag = "GOAL REACHED" if info.get("goal_reached") else "INCOMPLETE"
            color = "#388e3c" if info.get("goal_reached") else "#f57c00"
            ax.text(
                0.5, 0.92, tag,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.85),
                zorder=30,
            )
            n_col = sum(1 for r in results if r["result"] == "collision")
            n_suc = sum(1 for r in results if r["result"] == "success")
            summary = (
                f"Time: {elapsed_s:.1f} s  |  Steps: {info['step']}  |  "
                f"Collisions: {n_col}  |  Avoided: {n_suc}"
            )
            ax.text(
                0.5, 0.82, summary,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=12, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
                zorder=30,
            )
            env._fig.canvas.draw_idle()
            env._fig.canvas.flush_events()
            if env._recording:
                env._fig.canvas.draw()
                result_frame = np.asarray(
                    env._fig.canvas.buffer_rgba()
                )[..., :3].copy()
                for _ in range(15):
                    env._frames.append(result_frame)

    if save_video:
        env.stop_recording(save_video)

    if report_path:
        saved_report = _write_run_report(
            report_path,
            steps=info["step"],
            dt=env.cfg["dt"],
            goal_reached=info.get("goal_reached", False),
            encounter_results=results,
            path_length=env.cfg["path_length"],
            policy_mode=mode,
            seed=seed,
            video_path=save_video,
        )
        print(f"Report saved -> {saved_report}")

    env.close()
    return {
        "steps": info["step"],
        "elapsed_s": elapsed_s,
        "goal_reached": info.get("goal_reached", False),
        "encounter_results": results,
        "video_path": save_video,
        "report_path": report_path,
    }


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Full run simulator demo")
    pa.add_argument("--no-render", action="store_true")
    pa.add_argument("--save-video", type=str, default=None, metavar="PATH",
                    help="Save video + report under Evaluation_video/full_run/<run_name>/")
    pa.add_argument("--path-length", type=float, default=50.0)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--model", type=str, default=None,
                    help="Path to trained RL model (.zip)")
    pa.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"],
                    help="RL algorithm used for the model")
    pa.add_argument(
        "--policy",
        type=str,
        default="hybrid",
        choices=["hybrid", "path", "stop_corridor"],
        help="Control policy: hybrid | path | stop_corridor",
    )
    pa.add_argument(
        "--validate-encounters",
        action="store_true",
        help="Run the 25-case blocking encounter validation suite and exit",
    )
    args = pa.parse_args()

    if args.validate_encounters:
        p1, t1, _ = run_blocking_encounter_tests()
        p2, t2 = validate_random_encounter_suite(
            path_length=float(args.path_length),
        )
        ok = (p1 == t1) and (p2 == t2)
        raise SystemExit(0 if ok else 1)

    cfg = {
        "path_length": args.path_length,
        "scenario_seed": args.seed,
    }

    rl = None
    if args.model:
        from stable_baselines3 import PPO, SAC
        AlgoCls = {"ppo": PPO, "sac": SAC}[args.algo.lower()]
        rl = AlgoCls.load(args.model)

    run_full_demo(
        rl_model=rl,
        config=cfg,
        seed=args.seed,
        render=not args.no_render,
        save_video=args.save_video,
        policy_mode=args.policy,
    )