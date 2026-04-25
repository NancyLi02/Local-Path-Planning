"""
PPO training with Stable-Baselines3.
Clean run-based structure with metadata snapshot for reproducibility.

====================================================
USAGE EXAMPLES
====================================================

1. Train a new PPO model:
    python3 training_files/PPO_train.py --name ppo_v1

This creates:
    logs/PPO/ppo_v1/
        ppo_v1.zip
        best/
        eval/
        tb/
        metadata/
            reward_config.json
            train_config.json
            env_config.json
            reward_function.py.txt
            git_commit.txt

2. Train with custom timesteps:
    python3 training_files/PPO_train.py --name ppo_v1 --timesteps 1000000

3. Train with more parallel environments:
    python3 training_files/PPO_train.py --name ppo_v1 --n-envs 16


====================================================
EVALUATION
====================================================

4. Evaluate final saved model:
    python3 training_files/PPO_train.py --name ppo_v1 --eval

5. Evaluate best checkpoint:
    python3 training_files/PPO_train.py --name ppo_v1 --eval --use-best

6. Evaluate with on-screen rendering:
    python3 training_files/PPO_train.py --name ppo_v1 --eval --render

7. Evaluate and save GIF videos:
    python3 training_files/PPO_train.py --name ppo_v1 --eval --save-video

Videos saved to:
    Evaluation_video/PPO/ppo_v1/

8. Evaluate best model and save videos:
    python3 training_files/PPO_train.py --name ppo_v1 --eval --use-best --save-video


====================================================
REWARD DEBUGGING
====================================================

9. Run reward audit before training:
    python3 training_files/PPO_train.py --name ppo_v1 --reward-audit


====================================================
TENSORBOARD
====================================================

10. Launch TensorBoard:
    tensorboard --logdir /logs/PPO/ppo_v1/tb

Then open browser:
    http://localhost:6006


====================================================
NOTES
====================================================

1. All outputs are grouped by --name.
2. Reusing the same --name will overwrite/update that run.
3. Use different names for different reward settings or hyperparameters.
4. GPU will be used automatically if available.
5. Each training run saves a metadata snapshot for reproducibility.

"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from Simulators.Single_robot_simulator import LocalPlannerEnv, HybridPolicy

PPO_ROOT = _REPO_ROOT / "logs" / "PPO"
PPO_VIDEO_ROOT = _REPO_ROOT / "Evaluation_video" / "PPO"


# ---------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------

# Scene / scenario settings shared by train and eval unless overridden.
BASE_ENV_CFG = {
    "normalize_obs": True,
}

# Reward version and all reward-related parameters.
# Change reward_version whenever reward logic or reward weights change.
REWARD_CFG = {
    "reward_version": "r_v1",
    "w_collision": -200.0,
    "w_safety": -10.0,
    "w_deviation": -5.0,
    "w_heading": -2.0,
    "w_progress": 20.0,
    "w_speed": 2.0,
    "w_time": -1.0,
    "w_success": 100.0,
    "w_return_path": 10.0,
    "path_pen_min": 0.1,
    "path_pen_restore_dist": 2.0,
}

# Train-only scenario config
TRAIN_SCENE_CFG = {
    "p_ambient_human": 0.12,
    "encounter_t_range": (2.0, 4.5),
    "encounter_jitter": (0.90, 1.08),
    "human_delay": 1.5,
    "human_from_below_prob": 0.5,
}

# Eval-only scenario config
EVAL_SCENE_CFG = {
    "p_ambient_human": 0.0,
    "encounter_t_range": (2.0, 3.5),
    "encounter_jitter": (0.92, 1.05),
    "human_delay": 1.0,
    "human_from_below_prob": 0.5,
}


def _compose_env_cfg(*parts: dict) -> dict:
    cfg: dict = {}
    for p in parts:
        cfg.update(p)
    return cfg


TRAIN_ENV_CFG = _compose_env_cfg(BASE_ENV_CFG, REWARD_CFG, TRAIN_SCENE_CFG)
EVAL_ENV_CFG = _compose_env_cfg(BASE_ENV_CFG, REWARD_CFG, EVAL_SCENE_CFG)


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def _run_dir(name: str) -> Path:
    return PPO_ROOT / name


def _model_path(name: str) -> Path:
    return _run_dir(name) / name


def _best_model_path(name: str) -> Path:
    return _run_dir(name) / "best" / "best_model"


def _metadata_dir(name: str) -> Path:
    return _run_dir(name) / "metadata"


# ---------------------------------------------------------------------
# Reproducibility / metadata helpers
# ---------------------------------------------------------------------

def _json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "UNKNOWN"


def _snapshot_metadata(args, run_dir: Path) -> None:
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "algo": "PPO",
        "run_name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(_REPO_ROOT.resolve()),
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "eval_episodes": args.eval_episodes,
        "audit_episodes": args.audit_episodes,
        "device": "auto",
        "ppo_hyperparameters": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "cli_args": vars(args),
    }

    env_config = {
        "base_env_config": BASE_ENV_CFG,
        "reward_config": REWARD_CFG,
        "train_scene_config": TRAIN_SCENE_CFG,
        "eval_scene_config": EVAL_SCENE_CFG,
        "train_env_config": TRAIN_ENV_CFG,
        "eval_env_config": EVAL_ENV_CFG,
    }

    reward_source = inspect.getsource(LocalPlannerEnv._reward_terms)
    git_commit = _get_git_commit() + "\n"

    _json_dump(meta_dir / "reward_config.json", REWARD_CFG)
    _json_dump(meta_dir / "train_config.json", train_config)
    _json_dump(meta_dir / "env_config.json", env_config)
    _write_text(meta_dir / "reward_function.py.txt", reward_source)
    _write_text(meta_dir / "git_commit.txt", git_commit)


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------

def make_env(seed: int, cfg: dict | None = None):
    def _init():
        env = LocalPlannerEnv(config=cfg)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    name = args.name
    run_dir = _run_dir(name)
    tb_dir = run_dir / "tb"
    eval_dir = run_dir / "eval"
    best_dir = run_dir / "best"
    meta_dir = run_dir / "metadata"

    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata snapshot before training starts
    _snapshot_metadata(args, run_dir)

    n_envs = args.n_envs
    env = VecMonitor(SubprocVecEnv([make_env(i, TRAIN_ENV_CFG) for i in range(n_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(1000 + i, EVAL_ENV_CFG) for i in range(4)]))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="auto",
        tensorboard_log=str(tb_dir),
    )

    print(f"Training PPO ({name!r}) for {args.timesteps} timesteps with {n_envs} parallel envs ...")
    print(f"  run directory : {run_dir.resolve()}")
    print(f"  model path    : {_model_path(name).resolve()}.zip")
    print(f"  best model dir: {best_dir.resolve()}")
    print(f"  eval log dir  : {eval_dir.resolve()}")
    print(f"  tb log dir    : {tb_dir.resolve()}")
    print(f"  metadata dir  : {meta_dir.resolve()}")
    print(f"  reward version: {REWARD_CFG['reward_version']}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        tb_log_name=name,
    )

    save_stem = _model_path(name)
    model.save(str(save_stem))
    print(f"Model saved to {save_stem}.zip")

    env.close()
    eval_env.close()


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

def evaluate(args):
    from stable_baselines3 import PPO

    name = args.name
    render = args.render
    save_video = args.save_video

    if args.use_best:
        load_stem = _best_model_path(name)
    else:
        load_stem = _model_path(name)

    if not Path(str(load_stem) + ".zip").exists():
        raise FileNotFoundError(f"Model not found: {load_stem}.zip")

    env = LocalPlannerEnv(
        config=EVAL_ENV_CFG,
        render_mode="human" if (render or save_video) else None,
    )
    model = PPO.load(str(load_stem))

    n_ep = args.eval_episodes
    results = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0, "OTHER": 0}

    vid_dir = PPO_VIDEO_ROOT / name
    if save_video:
        vid_dir.mkdir(parents=True, exist_ok=True)
        print(f"Video save dir : {vid_dir.resolve()}")

    print(f"Evaluating model: {Path(str(load_stem) + '.zip').resolve()}")

    for ep in range(n_ep):
        obs, info = env.reset(seed=2000 + ep)

        if save_video:
            env.start_recording()

        done = False
        ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc

            if render or save_video:
                env.render()

        tag = (
            "COLLISION" if info.get("collision") else
            "SUCCESS" if info.get("success") else
            "TIMEOUT" if info.get("timeout") else
            "OTHER"
        )
        results[tag] += 1

        es = info.get("episode_stats", {})
        extra = ""
        if es:
            extra = (
                f"  min_d={es.get('min_human_dist', -1):.2f} "
                f"|lat|={es.get('final_abs_lateral', -1):.2f} "
                f"on_path={es.get('on_path_at_end')} "
                f"h_clear={es.get('human_clear_at_end')}"
            )

        print(f"  ep {ep + 1}: {tag}  steps={info['step']}  return={ret:.1f}  {extra}")

        if save_video:
            vid_path = vid_dir / f"eval_ep{ep + 1}_{tag.lower()}.gif"
            env.stop_recording(str(vid_path))

    env.close()

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}/{n_ep} ({100 * v / n_ep:.0f}%)")


# ---------------------------------------------------------------------
# Reward audit
# ---------------------------------------------------------------------

def reward_audit(args):
    """Run a simple rule-based policy and print reward term magnitudes (per-step means)."""
    audit_cfg = dict(TRAIN_ENV_CFG)
    audit_cfg["return_reward_breakdown"] = True

    env = LocalPlannerEnv(config=audit_cfg)

    term_sums: dict[str, float] = defaultdict(float)
    term_counts: dict[str, int] = defaultdict(int)
    n_ep = args.audit_episodes
    coll = 0
    succ = 0
    min_d_list: list[float] = []

    for ep in range(n_ep):
        env.reset(seed=4000 + ep)
        done = False
        info: dict = {}

        while not done:
            action = np.array([1.2, 0.0], dtype=np.float32)
            _obs, _r, term, trunc, info = env.step(action)
            done = term or trunc

            rt = info.get("reward_terms")
            if rt:
                for k, v in rt.items():
                    term_sums[k] += float(v)
                    term_counts[k] += 1

        st = info.get("episode_stats", {})
        if st.get("collision"):
            coll += 1
        if st.get("success"):
            succ += 1

        md = st.get("min_human_dist", -1.0)
        if md >= 0:
            min_d_list.append(md)

    print(f"Reward audit: {n_ep} episodes, rule policy action=(1.2, 0.0)\n")
    print(f"Reward version: {REWARD_CFG['reward_version']}")
    print("Per-step mean contribution (approx. scale PPO will see):")
    for k in sorted(term_sums.keys()):
        n = max(term_counts[k], 1)
        print(f"  {k:16s}  mean={term_sums[k] / n:8.4f}  (sum over {term_counts[k]} steps)")

    print(f"\nEpisode-level: collision_rate={coll / n_ep:.2f}, success_rate={succ / n_ep:.2f}")

    if min_d_list:
        print(
            f"min_human_dist over episodes (when human appeared): "
            f"mean={np.mean(min_d_list):.3f}  min={np.min(min_d_list):.3f}",
        )

    env.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train or evaluate PPO (Stable-Baselines3).")
    pa.add_argument(
        "--name",
        type=str,
        default="ppo_run",
        help="Run name. All model / tb / eval / best / video outputs follow this name.",
    )
    pa.add_argument("--eval", action="store_true", help="Evaluate saved model")
    pa.add_argument(
        "--use-best",
        action="store_true",
        help="When used with --eval, load best/best_model.zip instead of the final <name>.zip",
    )
    pa.add_argument(
        "--reward-audit",
        action="store_true",
        help="Run rule-based rollouts and print reward term magnitudes before training",
    )
    pa.add_argument("--render", action="store_true", help="Show visualization during eval")
    pa.add_argument("--save-video", action="store_true", help="Save each eval episode as gif")
    pa.add_argument("--timesteps", type=int, default=500_000)
    pa.add_argument("--n-envs", type=int, default=8)
    pa.add_argument("--eval-episodes", type=int, default=20)
    pa.add_argument("--audit-episodes", type=int, default=40)

    args = pa.parse_args()

    if args.reward_audit:
        reward_audit(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)