"""
Quick-start training script using Stable Baselines3 (PPO).

Usage:
    python train_example.py                       # train
    python train_example.py --reward-audit        # reward term magnitudes (rule policy)
    python train_example.py --eval                # evaluate saved model
    python train_example.py --eval --render       # evaluate with visualisation
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from light_weight_simulator import LocalPlannerEnv


TRAIN_CFG = {
    "p_ambient_human": 0.12,
    "encounter_t_range": (2.0, 4.5),
    "encounter_jitter": (0.90, 1.08),
    "human_delay": 1.5,
}

EVAL_CFG = {
    "p_ambient_human": 0.0,
    "encounter_t_range": (2.0, 3.5),
    "encounter_jitter": (0.92, 1.05),
    "human_delay": 1.0,
}


def make_env(seed: int, cfg: dict | None = None):
    def _init():
        env = LocalPlannerEnv(config=cfg)
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    n_envs = args.n_envs
    env = VecMonitor(SubprocVecEnv([make_env(i, TRAIN_CFG) for i in range(n_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(1000 + i, EVAL_CFG) for i in range(4)]))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
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
        device="cpu",
        tensorboard_log="./logs/tb",
    )

    print(f"Training for {args.timesteps} timesteps with {n_envs} parallel envs …")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save("./logs/ppo_local_planner")
    print("Model saved to ./logs/ppo_local_planner.zip")
    env.close()
    eval_env.close()


def evaluate(args):
    from stable_baselines3 import PPO

    render = args.render
    save_video = args.save_video
    env = LocalPlannerEnv(
        config=EVAL_CFG,
        render_mode="human" if (render or save_video) else None,
    )
    model = PPO.load("./logs/ppo_local_planner")

    n_ep = args.eval_episodes
    results = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0, "OTHER": 0}

    for ep in range(n_ep):
        obs, info = env.reset(seed=2000 + ep)
        if save_video:
            env.start_recording()
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
            env.render()

        tag = ("COLLISION" if info.get("collision") else
               "SUCCESS" if info.get("success") else
               "TIMEOUT" if info.get("timeout") else "OTHER")
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
            import os
            vid_dir = f"./Evaluation_video/{args.run_name}" if args.run_name else "./video"
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = f"{vid_dir}/eval_ep{ep + 1}_{tag.lower()}.gif"
            env.stop_recording(vid_path)

    env.close()
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}/{n_ep} ({100 * v / n_ep:.0f}%)")


def reward_audit(args):
    """Run a simple rule-based policy and print reward term magnitudes (per-step means)."""
    env = LocalPlannerEnv(
        config={
            "return_reward_breakdown": True,
        },
    )
    term_sums: dict[str, float] = defaultdict(float)
    term_counts: dict[str, int] = defaultdict(int)
    n_ep = args.audit_episodes
    coll = succ = 0
    min_d_list: list[float] = []

    for ep in range(n_ep):
        env.reset(seed=4000 + ep)
        done = False
        info: dict = {}
        while not done:
            # Mild forward local goal: same order of magnitude as a reasonable hand-tuned rule.
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


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--eval", action="store_true", help="evaluate saved model")
    pa.add_argument(
        "--reward-audit",
        action="store_true",
        help="run rule-based rollouts and print reward term magnitudes before training",
    )
    pa.add_argument("--render", action="store_true", help="show visualisation during eval")
    pa.add_argument("--save-video", action="store_true", help="save each eval episode as gif")
    pa.add_argument("--run-name", type=str, default=None,
                    help="name for this run (used as subfolder in Evaluation_video/)")
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
