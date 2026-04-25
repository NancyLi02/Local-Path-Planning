from __future__ import annotations

import numpy as np

try:
    from .path import wrap_angle
except ImportError:
    from path import wrap_angle


def compute_reward_terms(env, old_s: float, collision: bool, success: bool) -> dict[str, float]:
    """Per-shaping reward terms plus total."""
    c = env.cfg
    terms: dict[str, float] = {}

    if collision:
        terms["collision"] = float(c["w_collision"])
        terms["total"] = terms["collision"]
        return terms

    r = 0.0

    if env._human_visible:
        dh = float(np.hypot(env.rx - env.hx, env.ry - env.hy))
        if dh < c["safety_dist"]:
            s_term = c["w_safety"] * (c["safety_dist"] - dh) / c["safety_dist"]
            terms["safety"] = float(s_term)
            r += s_term
        else:
            terms["safety"] = 0.0
    else:
        dh = float("inf")
        terms["safety"] = 0.0

    pen_min = float(c.get("path_pen_min", 0.15))
    pen_dist = float(c.get("path_pen_restore_dist", 3.0))
    if dh < pen_dist:
        g = pen_min + (1.0 - pen_min) * (dh / pen_dist)
    else:
        g = 1.0
    terms["path_gate"] = float(g)

    _, lat, _ = env.path.closest_point(
        env.rx, env.ry, s_hint=env.cur_s, search_radius=5.0,
    )

    dev = c["w_deviation"] * g * lat ** 2
    terms["deviation"] = float(dev)
    r += dev

    hdg = c["w_heading"] * g * abs(
        wrap_angle(env.rtheta - env.path.heading(env.cur_s)),
    )
    terms["heading"] = float(hdg)
    r += hdg

    prog = c["w_progress"] * max(0.0, env.cur_s - old_s)
    terms["progress"] = float(prog)
    r += prog

    spd = c["w_speed"] * (env.rv / c["max_v"])
    terms["speed"] = float(spd)
    r += spd

    terms["time"] = float(c["w_time"])
    r += c["w_time"]

    w_return = float(c.get("w_return_path", 3.0))
    if g >= 0.9 and abs(lat) > c["success_lat_thresh"]:
        ret_bonus = w_return * max(0.0, env._prev_abs_lat - abs(lat))
        terms["return_path"] = float(ret_bonus)
        r += ret_bonus
    else:
        terms["return_path"] = 0.0

    env._prev_abs_lat = abs(lat)

    if success:
        terms["success_bonus"] = float(c["w_success"])
        r += c["w_success"]
    else:
        terms["success_bonus"] = 0.0

    terms["total"] = float(r)
    return terms


def compute_reward(env, old_s: float, collision: bool, success: bool) -> float:
    return compute_reward_terms(env, old_s, collision, success)["total"]