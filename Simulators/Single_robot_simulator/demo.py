from __future__ import annotations

import argparse

try:
    from .env import LocalPlannerEnv
    from .policies import (
        dodge_behind_action,
        follow_path_action,
        random_forward_action,
        reactive_avoid_action,
        slow_on_path_action,
        stop_and_wait_action,
    )
    from .rendering import result_tag, show_result
except ImportError:
    from env import LocalPlannerEnv
    from policies import (
        dodge_behind_action,
        follow_path_action,
        random_forward_action,
        reactive_avoid_action,
        slow_on_path_action,
        stop_and_wait_action,
    )
    from rendering import result_tag, show_result


def _run_single_episode(env, action_fn, render: bool = True) -> tuple[dict, float]:
    obs, info = env.reset(seed=0)
    ret = 0.0
    done = False

    while not done:
        action = action_fn(obs, env)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    return info, ret


def demo_random(episodes: int = 3, render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        print(f"\n=== Episode {ep + 1} | {info['behavior']} ===")
        print(f"    obs dim={obs.shape[0]}, action space={env.action_space}")

        ret = 0.0
        done = False
        while not done:
            action = random_forward_action(env)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
            env.render()

        tag = result_tag(info)
        print(f"    steps={info['step']}, return={ret:.1f}, result={tag}")
        show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_follow_path(render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=0)
    print(f"Follow-path baseline | {info['behavior']}")

    ret = 0.0
    done = False
    while not done:
        action = follow_path_action(obs, env, lookahead_idx=2)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_reactive_avoid(render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=0)
    print(f"Reactive avoidance | {info['behavior']}")

    ret = 0.0
    done = False
    while not done:
        action = reactive_avoid_action(obs, env)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_dodge_behind(render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=0)
    print(f"Dodge behind human | {info['behavior']}")

    ret = 0.0
    done = False
    while not done:
        action = dodge_behind_action(obs, env)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_stop_and_wait(render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=0)
    print(f"Stop-and-wait | {info['behavior']}")

    ret = 0.0
    done = False
    while not done:
        action = stop_and_wait_action(obs, env)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_slow_on_path(render: bool = True, save_video: str | None = None):
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=0)
    print(f"Slow on path | {info['behavior']}")

    ret = 0.0
    done = False
    while not done:
        action = slow_on_path_action(obs, env)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Local path planning simulator demo")
    pa.add_argument("--mode", choices=["random", "follow", "avoid", "dodge", "slow", "stop"], default="follow")
    pa.add_argument("--episodes", type=int, default=3)
    pa.add_argument("--no-render", action="store_true")
    pa.add_argument("--save-video", type=str, default=None, metavar="PATH", help="Save episode video")
    args = pa.parse_args()

    _render = not args.no_render
    _vid = args.save_video

    if args.mode == "random":
        demo_random(args.episodes, _render, _vid)
    elif args.mode == "avoid":
        demo_reactive_avoid(_render, _vid)
    elif args.mode == "dodge":
        demo_dodge_behind(_render, _vid)
    elif args.mode == "slow":
        demo_slow_on_path(_render, _vid)
    elif args.mode == "stop":
        demo_stop_and_wait(_render, _vid)
    else:
        demo_follow_path(_render, _vid)