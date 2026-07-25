"""Plot the V1 training curve from a saved train_v1 log file.

    python -m Centralized_Local_Planner.rl.plot_curve <logfile> [out.png]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    log = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parents[2] / "outputs" / "step_e_v1_training_curve.png"
    text = log.read_text()

    v0 = None
    m = re.search(r"V0 baseline.*ret=([\d.]+)", text)
    if m:
        v0 = float(m.group(1))

    steps, ret, comp, prog = [], [], [], []
    for line in text.splitlines():
        m = re.match(r"step\s+(\d+)\s+ret=\s*([\-\d.]+)\s+completion=\s*([\d.]+)%\s+progress=\s*([\d.]+)%", line)
        if m:
            steps.append(int(m.group(1))); ret.append(float(m.group(2)))
            comp.append(float(m.group(3))); prog.append(float(m.group(4)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    ax[0].plot(steps, ret, "-o", ms=3, color="#1565c0", label="V1 (learned)")
    if v0 is not None:
        ax[0].axhline(v0, ls="--", color="#d62728", label=f"V0 baseline ({v0:.1f})")
    ax[0].set_xlabel("env steps"); ax[0].set_ylabel("eval return")
    ax[0].set_title("V1 PPO training return"); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(steps, comp, "-o", ms=3, color="#2ca02c", label="completion %")
    ax[1].plot(steps, prog, "-o", ms=3, color="#ff9800", label="path progress %")
    ax[1].set_xlabel("env steps"); ax[1].set_ylabel("%")
    ax[1].set_title("V1 fleet metrics"); ax[1].grid(alpha=0.3); ax[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"saved -> {out}  ({len(steps)} points)")


if __name__ == "__main__":
    main()
