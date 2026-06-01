"""Run one RMFS backend episode without opening the NetLogo GUI.

This uses the same Python backend functions as the NetLogo interface, but keeps
the Inventory object in memory instead of loading/saving netlogo.state every
tick. Fast training I/O is enabled by default for quicker result checks.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import sys
import time
from contextlib import ExitStack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one RMFS Python-backend episode headlessly."
    )
    parser.add_argument(
        "--mode",
        choices=("ppo", "random", "rika", "demand"),
        default="ppo",
        help="PPS mode to use.",
    )
    parser.add_argument(
        "--max-ticks",
        type=float,
        default=3000.0,
        help="Stop when the backend simulation clock reaches this value.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional PPO .zip model path. Defaults to saved_models/pps_rl_best.zip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional simulation seed to apply before backend setup.",
    )
    parser.add_argument(
        "--normal-io",
        action="store_true",
        help="Disable RMFS_FAST_TRAIN and run with normal CSV/database I/O.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show backend debug prints while the episode is running.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=10.0,
        help="Print a progress line every N real seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.normal_io:
        os.environ.pop("RMFS_FAST_TRAIN", None)
    else:
        os.environ["RMFS_FAST_TRAIN"] = "1"

    if args.model_path:
        os.environ["PPS_RL_MODEL_PATH"] = args.model_path
    if args.seed is not None:
        os.environ["RMFS_SIM_SEED"] = str(args.seed)

    import netlogo

    progress_out = sys.stdout

    def maybe_silence_logs() -> ExitStack:
        stack = ExitStack()
        if not args.show_log:
            devnull = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(contextlib.redirect_stdout(devnull))
            stack.enter_context(contextlib.redirect_stderr(devnull))
        return stack

    def progress(message: str) -> None:
        print(message, file=progress_out, flush=True)

    setup_start = time.perf_counter()
    with maybe_silence_logs():
        if args.seed is not None:
            netlogo.set_sim_seed(args.seed)
        netlogo.set_pps_mode(args.mode)
        setup_result = netlogo.setup()

    if (
        isinstance(setup_result, str)
        and setup_result.startswith("An error occurred")
    ):
        raise SystemExit(
            "Backend setup failed. If assign_order.csv or pod_info.csv is open "
            "in Excel, NetLogo, or another Python process, close it and run again. "
            "Rerun with --show-log if you need the full traceback."
        )

    if not os.path.exists("netlogo.state"):
        raise SystemExit("Backend setup did not create netlogo.state.")

    with open("netlogo.state", "rb") as file:
        universe = pickle.load(file)

    for obj in universe._objects:
        obj.setUniverse(universe)

    with maybe_silence_logs():
        netlogo._configure_pps_rl_strategy(universe)

    run_start = time.perf_counter()
    last_progress = run_start
    backend_steps = 0

    with maybe_silence_logs():
        while universe._tick < args.max_ticks:
            universe.tick()
            netlogo._apply_pps_rl_policy(universe)
            backend_steps += 1
            now = time.perf_counter()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                last_progress = now
                progress(
                    "progress: "
                    f"tick={universe._tick:.2f}/{args.max_ticks:g}, "
                    f"steps={backend_steps}, "
                    f"throughput={netlogo._get_throughput(universe)}, "
                    f"elapsed={now - run_start:.1f}s"
                )

    run_elapsed = time.perf_counter() - run_start
    total_elapsed = time.perf_counter() - setup_start

    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed if args.seed is not None else ''}")
    print(f"Fast training I/O: {'off' if args.normal_io else 'on'}")
    print(f"Backend steps: {backend_steps}")
    print(f"Simulation tick: {universe._tick:.2f}")
    print(f"Setup + run seconds: {total_elapsed:.2f}")
    print(f"Run seconds: {run_elapsed:.2f}")
    print(f"Throughput: {netlogo._get_throughput(universe)}")
    print(f"Avg order completion time: {netlogo._get_avg_order_completion_time(universe)}")
    print(f"Pod visits: {netlogo._get_pod_visits(universe)}")
    print(f"Pile-on rate: {netlogo._get_pile_on_rate(universe)}")
    print(f"Picked quantity: {netlogo._get_picked_quantity(universe)}")
    print(f"Total energy: {universe.total_energy}")
    print(f"Stop-and-go: {universe.stop_and_go}")
    print(f"Total turning: {universe.total_turning}")


if __name__ == "__main__":
    main()
