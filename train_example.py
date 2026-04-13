"""
Quick-start training script using Stable Baselines3 (PPO).

Usage:
    python train_example.py                     # train
    python train_example.py --eval              # evaluate saved model
    python train_example.py --eval --render     # evaluate with visualisation
"""
from __future__ import annotations

import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from light_weight_simulator import LocalPlannerEnv


def make_env(seed: int):
    def _init():
        env = LocalPlannerEnv()
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    n_envs = args.n_envs
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(1000 + i) for i in range(4)]))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=max(5000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/tb",
    )

    print(f"Training for {args.timesteps} timesteps with {n_envs} parallel envs …")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save("./logs/ppo_local_planner")
    print("Model saved to ./logs/ppo_local_planner.zip")
    env.close()
    eval_env.close()


def evaluate(args):
    render = args.render
    env = LocalPlannerEnv(render_mode="human" if render else None)
    model = PPO.load("./logs/ppo_local_planner")

    n_ep = args.eval_episodes
    results = {"SUCCESS": 0, "COLLISION": 0, "TIMEOUT": 0, "OTHER": 0}

    for ep in range(n_ep):
        obs, info = env.reset(seed=2000 + ep)
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
            if render:
                env.render()

        tag = ("COLLISION" if info.get("collision") else
               "SUCCESS" if info.get("success") else
               "TIMEOUT" if info.get("timeout") else "OTHER")
        results[tag] += 1
        print(f"  ep {ep + 1}: {tag}  steps={info['step']}  return={ret:.1f}")

    env.close()
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {v}/{n_ep} ({100 * v / n_ep:.0f}%)")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--eval", action="store_true", help="evaluate saved model")
    pa.add_argument("--render", action="store_true", help="show visualisation during eval")
    pa.add_argument("--timesteps", type=int, default=500_000)
    pa.add_argument("--n-envs", type=int, default=8)
    pa.add_argument("--eval-episodes", type=int, default=20)
    args = pa.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
