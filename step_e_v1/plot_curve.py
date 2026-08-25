"""Plot the V1 training curve against the V0 teacher.

    python -m step_e_v1.plot_curve --history logs/step_e_v1/v1_history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--history", default="logs/step_e_v1/v1_history.json")
    pa.add_argument("--out", default="outputs/step_e_v1_module_training_curve.png")
    args = pa.parse_args(argv)

    path = Path(args.history) if Path(args.history).is_absolute() else _REPO / args.history
    data = json.loads(path.read_text())
    hist = data["history"]
    steps = [h["step"] for h in hist]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor("white")

    panels = [
        ("ret", "cluster return", data["v0"]["ret"]),
        ("collisions", "worker collisions / episode", data["v0"]["collisions"]),
        ("shield", "shield override rate", data["v0"]["shield"]),
    ]
    for a, (key, label, ref) in zip(ax, panels):
        a.plot(steps, [h[key] for h in hist], color="#1e88e5", lw=1.8, label="V1 (learned)")
        a.axhline(ref, color="#e53935", ls="--", lw=1.4, label="V0 (teacher)")
        a.axhline(data["bc"][key], color="#43a047", ls=":", lw=1.4, label="after BC")
        a.set_xlabel("environment steps"); a.set_title(label)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Step E V1 — PPO behind the safety shield", weight="bold")
    fig.tight_layout()
    out = Path(args.out) if Path(args.out).is_absolute() else _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
