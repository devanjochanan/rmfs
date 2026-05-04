"""
Merge multiple TensorBoard run folders into one continuous timeline.

Use case: a training session was started fresh instead of resumed, producing
two separate run folders that should appear as a single curve. This script
reads each run's scalars in order, shifts later runs' step counters to start
where the previous run ended, and writes a single merged event file.

Usage:
    python merge_tb_runs.py \
        --runs saved_models/runs/pps_ppo_20260424_160925 \
               saved_models/runs/pps_ppo_20260427_101557 \
        --out saved_models/runs/pps_ppo_merged
"""

from __future__ import annotations

import argparse
import os
from typing import List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def merge_runs(run_dirs: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)

    step_offset = 0
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"\n[{run_idx+1}/{len(run_dirs)}] Reading {run_dir}")
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        if not tags:
            print(f"  (no scalars found, skipping)")
            continue

        # Collect every (tag, step, value) and find max step in this run
        events_per_tag = {}
        max_step_this_run = 0
        for tag in tags:
            scalars = ea.Scalars(tag)
            events_per_tag[tag] = scalars
            if scalars:
                max_step_this_run = max(max_step_this_run, max(s.step for s in scalars))

        # Write each scalar with shifted step
        n_total = 0
        for tag, scalars in events_per_tag.items():
            for s in scalars:
                writer.add_scalar(tag, s.value, s.step + step_offset)
                n_total += 1
        writer.flush()

        print(f"  tags : {len(tags)}")
        print(f"  events written : {n_total}")
        print(f"  step range in this run : 0 .. {max_step_this_run}")
        print(f"  shifted into merged   : {step_offset} .. {step_offset + max_step_this_run}")

        step_offset += max_step_this_run + 1

    writer.close()
    print(f"\nMerged log written to: {out_dir}")
    print(f"Total continuous step span: 0 .. {step_offset - 1}")
    print(f"View with: python -m tensorboard.main --logdir \"{out_dir}\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run folders in chronological order")
    ap.add_argument("--out", required=True, help="Output folder for merged run")
    args = ap.parse_args()
    merge_runs(args.runs, args.out)
