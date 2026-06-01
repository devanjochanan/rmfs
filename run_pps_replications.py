"""Run paired PPS policy replications for thesis experiments.

Each replication regenerates a new order set and pod SKU allocation, then runs
the requested PPS policies on that same scenario. Results are written after
each policy run so a long experiment still leaves usable data if interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import random
import shutil
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


POLICY_MODES = ("rika", "random", "ppo")

SCENARIO_INPUT_FILES = (
    "generated_order.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "generated_pod.csv",
    "pods.csv",
    "items.csv",
    "items_dictionary.csv",
    "items_slots_configuration.csv",
    "pods_dictionary.csv",
)

REGENERATED_SCENARIO_FILES = (
    "generated_order.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "pods.csv",
)

RUNTIME_FILES = (
    "assign_order.csv",
    "pod_info.csv",
    "netlogo.state",
    "skus_data.csv",
    "sorted_skus_data.csv",
)

RESULT_FIELDS = (
    "replication",
    "seed",
    "mode",
    "max_ticks",
    "fast_io",
    "backend_steps",
    "simulation_tick",
    "throughput",
    "avg_order_completion_time",
    "pod_visits",
    "pile_on_rate",
    "picked_quantity",
    "total_energy",
    "stop_and_go",
    "total_turning",
    "setup_seconds",
    "run_seconds",
    "total_seconds",
    "model_path",
)

SUMMARY_METRICS = (
    "throughput",
    "avg_order_completion_time",
    "pod_visits",
    "pile_on_rate",
    "picked_quantity",
    "total_energy",
    "stop_and_go",
    "total_turning",
    "run_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Rika, random PPO-style, and trained PPO PPS replications."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=30,
        help="Number of generated scenarios to evaluate.",
    )
    parser.add_argument(
        "--max-ticks",
        type=float,
        default=3000.0,
        help="Stop each run when the backend simulation clock reaches this value.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=POLICY_MODES,
        default=list(POLICY_MODES),
        help="Policies to run for each replication.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260527,
        help="Seed for replication 1. Replication N uses base_seed + N - 1.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional trained PPO .zip model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/replications/<timestamp>.",
    )
    parser.add_argument(
        "--fast-io",
        action="store_true",
        help="Enable RMFS_FAST_TRAIN=1. Default is normal I/O for thesis-style runs.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show backend debug prints.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="Print progress every N real seconds per policy run. Use 0 to disable.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def remove_existing(files: Iterable[str]) -> None:
    for name in files:
        path = Path(name)
        if path.exists():
            path.unlink()


def copy_existing(files: Iterable[str], source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


def restore_scenario_inputs(scenario_dir: Path) -> None:
    remove_existing((*SCENARIO_INPUT_FILES, *RUNTIME_FILES))
    for src in scenario_dir.iterdir():
        if src.is_file() and src.name in SCENARIO_INPUT_FILES:
            shutil.copy2(src, Path(src.name))


def silence_backend(show_log: bool) -> ExitStack:
    stack = ExitStack()
    if not show_log:
        devnull = stack.enter_context(open(os.devnull, "w"))
        stack.enter_context(contextlib.redirect_stdout(devnull))
        stack.enter_context(contextlib.redirect_stderr(devnull))
    return stack


def setup_backend(netlogo, mode: str, seed: int, show_log: bool) -> float:
    set_seed(seed)
    remove_existing(RUNTIME_FILES)
    start = time.perf_counter()
    with silence_backend(show_log):
        netlogo.set_pps_mode(mode)
        result = netlogo.setup()
    if isinstance(result, str) and result.startswith("An error occurred"):
        raise RuntimeError(
            "Backend setup failed. Close Excel, NetLogo, or other Python runs "
            "that may be locking assign_order.csv, pod_info.csv, or warehouse.db. "
            "Rerun with --show-log for the traceback."
        )
    if not Path("netlogo.state").exists():
        raise RuntimeError("Backend setup did not create netlogo.state.")
    return time.perf_counter() - start


def generate_scenario(netlogo, replication: int, seed: int, scenario_dir: Path, show_log: bool) -> None:
    remove_existing((*REGENERATED_SCENARIO_FILES, *RUNTIME_FILES))
    setup_backend(netlogo, mode="rika", seed=seed, show_log=show_log)
    copy_existing(SCENARIO_INPUT_FILES, Path.cwd(), scenario_dir)
    metadata = {
        "replication": replication,
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Orders and pods.csv were regenerated for this replication. "
            "Each policy restores these inputs before setup."
        ),
    }
    (scenario_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_policy(netlogo, args: argparse.Namespace, replication: int, seed: int, mode: str, scenario_dir: Path) -> dict:
    restore_scenario_inputs(scenario_dir)
    setup_seconds = setup_backend(netlogo, mode=mode, seed=seed, show_log=args.show_log)

    with open("netlogo.state", "rb") as file:
        universe = pickle.load(file)

    for obj in universe._objects:
        obj.setUniverse(universe)

    with silence_backend(args.show_log):
        netlogo._configure_pps_rl_strategy(universe)

    if mode == "ppo":
        ppo_active = (
            getattr(universe, "pps_rl", False)
            and not getattr(universe, "pps_rl_random", False)
            and not getattr(universe, "pps_pileon", False)
        )
        if not ppo_active:
            raise RuntimeError(
                "Trained PPO PPS did not activate. The backend likely fell back "
                "to Rika PPS because the PPO model was missing, incompatible, or "
                "failed to load. Check the model path and rerun with --show-log. "
                f"Current PPO model path: {getattr(netlogo, 'PPS_RL_MODEL_PATH', '')}"
            )

    if mode == "random":
        random_active = (
            getattr(universe, "pps_rl", False)
            and getattr(universe, "pps_rl_random", False)
            and not getattr(universe, "pps_pileon", False)
        )
        if not random_active:
            raise RuntimeError("Random PPO-style PPS did not activate.")

    run_start = time.perf_counter()
    last_progress = run_start
    backend_steps = 0
    progress_out = sys.stdout

    with silence_backend(args.show_log):
        while universe._tick < args.max_ticks:
            universe.tick()
            netlogo._apply_pps_rl_policy(universe)
            backend_steps += 1
            now = time.perf_counter()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                last_progress = now
                print(
                    "progress: "
                    f"rep={replication}, mode={mode}, "
                    f"tick={universe._tick:.2f}/{args.max_ticks:g}, "
                    f"steps={backend_steps}, "
                    f"throughput={netlogo._get_throughput(universe)}, "
                    f"elapsed={now - run_start:.1f}s",
                    file=progress_out,
                    flush=True,
                )

    run_seconds = time.perf_counter() - run_start
    total_seconds = setup_seconds + run_seconds

    return {
        "replication": replication,
        "seed": seed,
        "mode": mode,
        "max_ticks": args.max_ticks,
        "fast_io": bool(args.fast_io),
        "backend_steps": backend_steps,
        "simulation_tick": round(universe._tick, 6),
        "throughput": netlogo._get_throughput(universe),
        "avg_order_completion_time": netlogo._get_avg_order_completion_time(universe),
        "pod_visits": netlogo._get_pod_visits(universe),
        "pile_on_rate": netlogo._get_pile_on_rate(universe),
        "picked_quantity": netlogo._get_picked_quantity(universe),
        "total_energy": universe.total_energy,
        "stop_and_go": universe.stop_and_go,
        "total_turning": universe.total_turning,
        "setup_seconds": setup_seconds,
        "run_seconds": run_seconds,
        "total_seconds": total_seconds,
        "model_path": args.model_path or "",
    }


def append_result(csv_path: Path, row: dict) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summaries(results_path: Path, summary_path: Path, paired_path: Path) -> None:
    with results_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary_fields = ["mode", "n"]
    for metric in SUMMARY_METRICS:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for mode, mode_rows in by_mode.items():
            out = {"mode": mode, "n": len(mode_rows)}
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in mode_rows]
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
                out[f"{metric}_min"] = min(values)
                out[f"{metric}_max"] = max(values)
            writer.writerow(out)

    comparisons = (
        ("ppo", "rika"),
        ("ppo", "random"),
        ("random", "rika"),
    )
    paired_rows = []
    by_replication: dict[int, dict[str, dict]] = {}
    for row in rows:
        replication = int(row["replication"])
        by_replication.setdefault(replication, {})[row["mode"]] = row

    for metric in SUMMARY_METRICS:
        for left, right in comparisons:
            diffs = []
            for rep_modes in by_replication.values():
                if left not in rep_modes or right not in rep_modes:
                    continue
                diffs.append(
                    float(rep_modes[left][metric]) - float(rep_modes[right][metric])
                )
            if not diffs:
                continue
            paired_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left}_minus_{right}",
                    "mean_difference": statistics.mean(diffs),
                    "std_difference": (
                        statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                    ),
                    "min_difference": min(diffs),
                    "max_difference": max(diffs),
                    "n": len(diffs),
                }
            )

    paired_fields = [
        "metric",
        "comparison",
        "mean_difference",
        "std_difference",
        "min_difference",
        "max_difference",
        "n",
    ]
    with paired_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / "replications" / f"pps_policy_comparison_{timestamp}"
    )
    scenario_root = output_dir / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_root.mkdir(parents=True, exist_ok=True)

    if args.fast_io:
        os.environ["RMFS_FAST_TRAIN"] = "1"
    else:
        os.environ.pop("RMFS_FAST_TRAIN", None)

    if args.model_path:
        os.environ["PPS_RL_MODEL_PATH"] = args.model_path

    import netlogo

    results_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_by_mode.csv"
    paired_path = output_dir / "paired_differences.csv"

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["started_at"] = datetime.now().isoformat(timespec="seconds")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Fast training I/O: {'on' if args.fast_io else 'off'}")
    print(f"Raw results: {results_path}")

    for replication in range(1, args.replications + 1):
        seed = args.base_seed + replication - 1
        scenario_dir = scenario_root / f"rep_{replication:03d}"
        print(f"\n=== Replication {replication}/{args.replications} seed={seed} ===", flush=True)
        generate_scenario(netlogo, replication, seed, scenario_dir, args.show_log)

        for mode in args.modes:
            print(f"--- Running {mode} ---", flush=True)
            row = run_policy(netlogo, args, replication, seed, mode, scenario_dir)
            append_result(results_path, row)
            print(
                f"done: rep={replication}, mode={mode}, "
                f"throughput={row['throughput']}, "
                f"avg_oct={row['avg_order_completion_time']:.3f}, "
                f"energy={row['total_energy']:.3f}, "
                f"run_seconds={row['run_seconds']:.1f}",
                flush=True,
            )

    write_summaries(results_path, summary_path, paired_path)
    print("\nExperiment complete.")
    print(f"Summary by mode: {summary_path}")
    print(f"Paired differences: {paired_path}")


if __name__ == "__main__":
    main()
