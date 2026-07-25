"""Backfill TensorBoard event files from a saved train_*.py console log.

Lets you view runs that finished before SummaryWriter logging was added.

    python -m Centralized_Local_Planner.rl.backfill_tb <logfile> <tb_outdir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

_STEP = re.compile(
    r"step\s+(\d+)\s+ret=\s*([\-\d.]+)\s+completion=\s*([\d.]+)%\s+"
    r"progress=\s*([\d.]+)%\s+collided=\s*([\d.]+)")
_BASE = re.compile(r"(?:greedy|V0).*?ret=([\-\d.]+)")


def main():
    log = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = log.read_text()
    base = None
    m = _BASE.search(text)
    if m:
        base = float(m.group(1))
    w = SummaryWriter(str(out))
    n = 0
    for line in text.splitlines():
        m = _STEP.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        w.add_scalar("eval/return", float(m.group(2)), step)
        w.add_scalar("eval/completion", float(m.group(3)) / 100.0, step)
        w.add_scalar("eval/progress", float(m.group(4)) / 100.0, step)
        w.add_scalar("eval/collided", float(m.group(5)), step)
        if base is not None:
            w.add_scalar("ref/baseline_return", base, step)
        n += 1
    w.close()
    print(f"backfilled {n} points from {log.name} -> {out}  (baseline ret={base})")


if __name__ == "__main__":
    main()
