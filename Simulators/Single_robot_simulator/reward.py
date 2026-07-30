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

    in_safety_bubble = env._human_visible and dh < c["safety_dist"]
    encounter_outside = env._human_visible and not in_safety_bubble

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

    # Attenuate progress/speed inside the safety bubble so stopping/slowing
    # remains viable near humans; keep full incentives outside for replanning.
    if in_safety_bubble:
        move_scale = float(c.get("encounter_move_scale_inside", 1.0))
    else:
        move_scale = 1.0
    terms["move_scale"] = float(move_scale)

    prog = c["w_progress"] * move_scale * max(0.0, env.cur_s - old_s)
    terms["progress"] = float(prog)
    r += prog

    spd = c["w_speed"] * move_scale * (env.rv / c["max_v"])
    terms["speed"] = float(spd)
    r += spd

    # Penalize standing still outside the safety bubble. During an encounter
    # outside the bubble, use a stronger idle weight scaled by human distance
    # so early detection-range stops (dh ~ 6-8 m) are clearly worse than replanning.
    w_idle = float(c.get("w_idle", 0.0))
    idle_thresh = float(c.get("idle_speed_thresh", 0.25))
    if w_idle != 0.0 and not in_safety_bubble and env.rv < idle_thresh:
        if encounter_outside:
            w_eff = float(c.get("w_idle_encounter", w_idle))
            idle = w_eff * (1.0 - env.rv / max(idle_thresh, 1e-6))
            far_extra = float(c.get("idle_far_scale", 0.0))
            far_cap = float(c.get("idle_far_cap", 2.0))
            sd = float(c["safety_dist"])
            far_mult = 1.0 + far_extra * min((dh - sd) / sd, far_cap)
            idle *= far_mult
        else:
            idle = w_idle * (1.0 - env.rv / max(idle_thresh, 1e-6))
        terms["idle"] = float(idle)
        r += idle
    else:
        terms["idle"] = 0.0

    terms["time"] = float(c["w_time"])
    r += c["w_time"]

    # Reward lateral recovery only while it is SAFE to reconverge -- i.e. the human
    # is no longer a frontal obstacle. Allow when: no human, the human is already far
    # (dh >= safety_dist), the human is BEHIND the robot, or the human has left the
    # corridor. This keeps recovery shaping active through the post-pass phase
    # (including inside the safety bubble while the human clears), but never rewards a
    # reduction in |lat| that would cut in front of / through a human still ahead in
    # the corridor. Distance-only gating (dh >= safety_dist) missed ~half the recovery.
    w_return = float(c.get("w_return_path", 3.0))
    if env._human_visible:
        cr_r, sr_r = np.cos(env.rtheta), np.sin(env.rtheta)
        human_ahead = cr_r * (env.hx - env.rx) + sr_r * (env.hy - env.ry)
        human_cleared = (human_ahead < 0.0) or (not env._in_corridor(env.hx, env.hy))
        allow_return = (dh >= c["safety_dist"]) or human_cleared
    else:
        allow_return = True
    if allow_return and abs(lat) > c["success_lat_thresh"]:
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
