"""Plot the learned policy's training curve against its teacher.

The teacher is whatever ``learning_based.train`` cloned from -- today the
optimization-based planner, an MPC formulation later. The reference lines
are what the learned policy has to beat to be worth its complexity.

    python -m Centralized_Local_Planner.local_path_replanning.plot_curve --history logs/local_path_replanning/learning_based/history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--history", default="logs/local_path_replanning/learning_based/history.json")
    pa.add_argument("--out", default="outputs/5_local_path_replanning/training_curve.png")
    args = pa.parse_args(argv)

    path = Path(args.history) if Path(args.history).is_absolute() else _REPO / args.history
    data = json.loads(path.read_text())
    hist = data["history"]
    # "teacher" is the current key; "v0" is what pre-reorganisation runs wrote.
    ref = data.get("teacher") or data["v0"]
    steps = [h["step"] for h in hist]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor("white")

    panels = [
        ("ret", "cluster return", ref["ret"]),
        ("collisions", "worker collisions / episode", ref["collisions"]),
        ("shield", "shield override rate", ref["shield"]),
    ]
    for a, (key, label, ref) in zip(ax, panels):
        a.plot(steps, [h[key] for h in hist], color="#1e88e5", lw=1.8, label="learning-based")
        a.axhline(ref, color="#e53935", ls="--", lw=1.4, label="teacher")
        a.axhline(data["bc"][key], color="#43a047", ls=":", lw=1.4, label="after BC")
        a.set_xlabel("environment steps"); a.set_title(label)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Learning-based local replanning — PPO behind the safety shield",
                 weight="bold")
    fig.tight_layout()
    out = Path(args.out) if Path(args.out).is_absolute() else _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
