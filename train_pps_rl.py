"""
PPO training script for PPS (Pick Pod Selection) in RMFS.

SB3 controls the full training loop through a single DummyVecEnv.
n_steps is set large enough to contain one full episode, so PPO
updates happen approximately once per episode.

Uses Stable-Baselines3 PPO with:
  - Custom Gymnasium env (PPSEnv)
  - TensorBoard logging (pps_reward/ and pps_metrics/ sections)
  - CSV data recording
  - Model save/load + checkpoints

Usage:
    python train_pps_rl.py --episodes 500 --max-ticks 5000
    python train_pps_rl.py --resume
    python train_pps_rl.py --eval
    tensorboard --logdir saved_models/runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from pps_env import PPSEnv

# ---------------------------------------------------------------------------
# Paths (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
MODEL_DIR = "saved_models"
LOG_DIR = os.path.join("saved_models", "runs")
METRICS_DIR = os.path.join("saved_models", "metrics")
BEST_MODEL = os.path.join(MODEL_DIR, "pps_ppo_best")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
TRAIN_STATE_FILE = os.path.join(MODEL_DIR, "train_state.json")


# ---------------------------------------------------------------------------
# CSV Metrics Recorder
# ---------------------------------------------------------------------------
class MetricsRecorder:
    """Records per-episode metrics to a CSV file for later analysis."""

    HEADER = [
        "episode", "timestamp",
        "throughput", "cumulative_path_cost",
        "pile_on_rate", "pile_on_items", "pile_on_visits",
        "avg_order_completion_time",
        "total_reward", "episode_steps", "sim_ticks",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def record(self, episode: int, info: Dict, total_reward: float, steps: int):
        row = [
            episode,
            datetime.now().isoformat(),
            info.get("throughput", 0),
            info.get("cumulative_path_cost", 0.0),
            info.get("pile_on_rate", 0.0),
            info.get("pile_on_items", 0),
            info.get("pile_on_visits", 0),
            info.get("avg_order_completion_time", 0.0),
            total_reward,
            steps,
            info.get("tick", 0),
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Custom Callback — handles TensorBoard, CSV, console, checkpoints
# ---------------------------------------------------------------------------
class PPSMetricsCallback(BaseCallback):
    """
    Tracks per-episode metrics inside SB3's training loop.

    TensorBoard sections:
      pps_reward/   : total_reward, pile_on_rate, avg_order_completion_time
      pps_metrics/  : throughput, cumulative_path_cost, episode_steps
    """

    def __init__(
        self,
        tb_writer: SummaryWriter | None = None,
        recorder: MetricsRecorder | None = None,
        verbose: int = 0,
        episode_offset: int = 0,
        best_throughput: int = 0,
        n_envs: int = 1,
        pbar: tqdm | None = None,
    ):
        super().__init__(verbose)
        self._tb_writer = tb_writer
        self._recorder = recorder
        self._best_throughput = best_throughput
        self._episode_count = episode_offset
        self._n_envs = n_envs
        # Per-env accumulators (support multi-env rollouts)
        self._env_reward = np.zeros(n_envs, dtype=np.float64)
        self._env_steps = np.zeros(n_envs, dtype=np.int64)
        self._env_start = [time.time()] * n_envs
        self._pbar = pbar

    def _on_step(self) -> bool:
        # Accumulate per-env reward and step counts
        rewards = self.locals.get("rewards", [])
        for i, r in enumerate(rewards):
            self._env_reward[i] += float(r)
            self._env_steps[i] += 1

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            if i < len(dones) and dones[i]:
                self._episode_count += 1
                ep = self._episode_count
                ep_reward = float(self._env_reward[i])
                ep_steps = int(self._env_steps[i])
                ep_elapsed = time.time() - self._env_start[i]

                throughput = info.get("throughput", 0)
                pile_on = info.get("pile_on_rate", 0.0)
                avg_oct = info.get("avg_order_completion_time", 0.0)
                cpc = info.get("cumulative_path_cost", 0.0)

                # TensorBoard
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("pps_reward/total_reward", ep_reward, ep)
                    self._tb_writer.add_scalar("pps_reward/pile_on_rate", pile_on, ep)
                    self._tb_writer.add_scalar("pps_reward/avg_order_completion_time", avg_oct, ep)
                    self._tb_writer.add_scalar("pps_metrics/throughput", throughput, ep)
                    self._tb_writer.add_scalar("pps_metrics/cumulative_path_cost", cpc, ep)
                    self._tb_writer.add_scalar("pps_metrics/episode_steps", ep_steps, ep)
                    self._tb_writer.flush()

                # CSV
                if self._recorder is not None:
                    self._recorder.record(ep, info, ep_reward, ep_steps)

                # tqdm progress bar — one line, auto-updating
                if self._pbar is not None:
                    self._pbar.update(1)
                    self._pbar.set_postfix({
                        "tp": throughput,
                        "po": f"{pile_on:.2f}",
                        "oct": f"{avg_oct:.0f}",
                        "rew": f"{ep_reward:.1f}",
                        "t": f"{ep_elapsed:.1f}s",
                    })

                # Save best model
                if throughput > self._best_throughput:
                    self._best_throughput = throughput
                    self.model.save(BEST_MODEL)
                    if self._pbar is not None:
                        self._pbar.write(
                            f"  >> ep {ep}: new best throughput {throughput} - model saved"
                        )

                # Periodic checkpoint
                if ep % 10 == 0:
                    cp_path = os.path.join(CHECKPOINT_DIR, f"pps_ppo_ep{ep}")
                    self.model.save(cp_path)

                # Reset this env's accumulators
                self._env_reward[i] = 0.0
                self._env_steps[i] = 0
                self._env_start[i] = time.time()

        return True


# ---------------------------------------------------------------------------
# Stop training after a fixed number of session episodes
# ---------------------------------------------------------------------------
class StopAfterEpisodesCallback(BaseCallback):
    """Stops `model.learn()` after a target number of episodes in this session.

    Lets us pass a large total_timesteps to learn() (so `_total_timesteps`
    matches the LR decay plan), while still limiting how much we train
    per invocation.
    """

    def __init__(self, session_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.session_episodes = session_episodes
        self._session_ep_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._session_ep_count += 1
                if self._session_ep_count >= self.session_episodes:
                    print(
                        f"\n[StopCallback] Session target reached: "
                        f"{self._session_ep_count} episodes. Stopping."
                    )
                    return False
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_pps_env(**kwargs):
    """Factory for creating PPSEnv instances (single-process / DummyVecEnv)."""
    def _init():
        return PPSEnv(**kwargs)
    return _init


# Static input files each worker needs its own copy of.
_WORKER_STATIC_FILES = [
    "items.csv",
    "pods.csv",
    "generated_pod.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "allowed_direction_changes.csv",
    "pod_info.csv",
    "generated_order.csv",
    "generated_database_order.csv",
    "generated_backlog.csv",
]


def make_worker_env(worker_id: int, base_dir: str, max_ticks: int):
    """Factory for SubprocVecEnv workers.

    Each worker:
      1. Creates a private tempdir.
      2. Copies static input CSVs from base_dir into it.
      3. chdirs into the tempdir so all CWD-relative file I/O is isolated.
      4. Monkey-patches model.order_generator.parent_directory to point there
         (order_generator uses an absolute path computed at import time).
      5. Creates a fresh PPSEnv inside the tempdir.

    Note: each worker regenerates its own orders independently, so random
    seeds differ naturally — good for RL sample diversity.
    """
    def _init():
        workdir = tempfile.mkdtemp(prefix=f"pps_worker_{worker_id}_")

        # Copy static files (skip missing ones silently)
        for fname in _WORKER_STATIC_FILES:
            src = os.path.join(base_dir, fname)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(workdir, fname))
                except Exception:
                    pass

        os.makedirs(os.path.join(workdir, "output"), exist_ok=True)
        os.chdir(workdir)

        # Redirect order_generator's absolute path to the worker dir
        import model.order_generator as og_mod
        og_mod.parent_directory = workdir

        return PPSEnv(max_episode_ticks=max_ticks)
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    total_episodes: int = 50,
    resume: bool = False,
    learning_rate: float = 3e-4,
    lr_end: float | None = None,
    lr_plan_episodes: int | None = None,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    max_episode_ticks: int = 32400,
    n_steps: int = 4096,
    n_envs: int = 1,
):
    """
    Train PPO for PPS using SB3's standard training loop.

    n_steps controls how many PPS decision steps SB3 collects before
    running a PPO update. Setting n_steps large enough to contain one
    full episode means the policy updates approximately once per episode.

    total_timesteps = n_steps * total_episodes, so the model runs
    for roughly total_episodes episodes before stopping.

    Args:
        total_episodes: Approximate number of episodes to train.
        resume: Load existing model checkpoint.
        learning_rate: PPO learning rate.
        batch_size: Minibatch size for SGD within each update.
        n_epochs: PPO epochs (passes over rollout data) per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping parameter.
        ent_coef: Entropy bonus coefficient (encourages exploration).
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clipping norm.
        max_episode_ticks: Max simulation ticks per episode.
        n_steps: Steps per PPO rollout (set >= typical episode length).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Vec env: Dummy (serial) for n_envs=1, Subproc (parallel) otherwise
    if n_envs > 1:
        base_dir = os.getcwd()
        env_fns = [
            make_worker_env(i, base_dir, max_episode_ticks)
            for i in range(n_envs)
        ]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"Using SubprocVecEnv with {n_envs} parallel workers.")
    else:
        vec_env = DummyVecEnv([make_pps_env(max_episode_ticks=max_episode_ticks)])

    # Load previous training state if resuming
    prev_episode_count = 0
    prev_best_throughput = 0
    run_name = None
    prev_plan_total_timesteps = None

    if resume and os.path.exists(TRAIN_STATE_FILE):
        with open(TRAIN_STATE_FILE, "r") as f:
            state = json.load(f)
        prev_episode_count = state.get("episode_count", 0)
        prev_best_throughput = state.get("best_throughput", 0)
        run_name = state.get("run_name", None)
        prev_plan_total_timesteps = state.get("plan_total_timesteps", None)
        print(f"Restoring training state: {prev_episode_count} episodes, best throughput {prev_best_throughput}")
        if prev_plan_total_timesteps is not None:
            print(f"  Restored LR decay plan: {prev_plan_total_timesteps} timesteps")

    if run_name is None:
        run_name = f"pps_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # LR schedule plan (persisted across resumes so decay curve is continuous)
    if prev_plan_total_timesteps is not None:
        plan_total_timesteps = prev_plan_total_timesteps
    else:
        plan_episodes = lr_plan_episodes if lr_plan_episodes is not None else total_episodes
        plan_total_timesteps = n_steps * plan_episodes

    # Final LR for linear decay (default: 10% of start LR)
    if lr_end is None:
        lr_end = learning_rate * 0.1

    # TensorBoard + CSV (reuse same folder on resume for continuous graph)
    tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    csv_path = os.path.join(METRICS_DIR, f"metrics_{run_name}.csv")
    recorder = MetricsRecorder(csv_path)

    if resume and os.path.exists(BEST_MODEL + ".zip"):
        print(f"Resuming from {BEST_MODEL}")
        model = PPO.load(BEST_MODEL, env=vec_env, device="cpu")
        # PPO.load restores lr_schedule (the linear decay fn) and optimizer state.
        current_progress = 1.0 - float(model.num_timesteps) / float(plan_total_timesteps)
        current_progress = max(0.0, min(1.0, current_progress))
        current_lr = model.lr_schedule(current_progress)
        print(f"  Restored num_timesteps: {model.num_timesteps}")
        print(f"  Current scheduled LR:   {current_lr:.2e}")
        model.n_steps = n_steps
        # Recreate rollout buffer to match new n_steps
        from stable_baselines3.common.buffers import DictRolloutBuffer
        model.rollout_buffer = DictRolloutBuffer(
            n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=1,
        )
    else:
        print("Training from scratch")
        # Linear LR schedule from learning_rate -> lr_end over plan_total_timesteps.
        # get_linear_fn is picklable, so it persists through model.save/load.
        lr_schedule = get_linear_fn(learning_rate, lr_end, end_fraction=1.0)
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=lr_schedule,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log=None,
            device="cpu",
        )

    # Pass total_timesteps so SB3's _total_timesteps equals plan_total_timesteps
    # (needed for the linear LR schedule to decay over the full plan horizon).
    # The StopAfterEpisodesCallback limits actual run length to `total_episodes`.
    if resume:
        learn_total_timesteps = max(1, plan_total_timesteps - model.num_timesteps)
    else:
        learn_total_timesteps = plan_total_timesteps

    print(f"\n{'='*60}")
    print(f"PPS RL Training")
    print(f"{'='*60}")
    print(f"  Session episodes         : {total_episodes}")
    print(f"  Parallel envs            : {n_envs}")
    print(f"  LR decay plan (ts total) : {plan_total_timesteps}")
    print(f"  LR decay plan (episodes) : {plan_total_timesteps // n_steps}")
    print(f"  n_steps per rollout      : {n_steps}")
    print(f"  Initial learning rate    : {learning_rate}")
    print(f"  Final learning rate      : {lr_end}")
    print(f"  Batch size               : {batch_size}")
    print(f"  PPO epochs/update        : {n_epochs}")
    print(f"  Gamma                    : {gamma}")
    print(f"  GAE lambda               : {gae_lambda}")
    print(f"  Clip range               : {clip_range}")
    print(f"  Entropy coef             : {ent_coef}")
    print(f"  Max episode ticks        : {max_episode_ticks}")
    print(f"  TensorBoard              : python -m tensorboard.main --logdir \"{os.path.abspath(LOG_DIR)}\"")
    print(f"  CSV metrics              : {csv_path}")
    print(f"{'='*60}\n")

    # tqdm progress bar: one line per episode, auto-updating
    pbar = tqdm(
        total=total_episodes,
        desc="Training",
        unit="ep",
        dynamic_ncols=True,
    )

    # Custom callback handles all logging
    metrics_cb = PPSMetricsCallback(
        tb_writer=tb_writer,
        recorder=recorder,
        verbose=1,
        episode_offset=prev_episode_count,
        best_throughput=prev_best_throughput,
        n_envs=n_envs,
        pbar=pbar,
    )

    # Stop after `total_episodes` episodes in THIS session (across all envs)
    stop_cb = StopAfterEpisodesCallback(session_episodes=total_episodes, verbose=1)

    try:
        model.learn(
            total_timesteps=learn_total_timesteps,
            callback=CallbackList([metrics_cb, stop_cb]),
            reset_num_timesteps=not resume,
        )
    finally:
        pbar.close()

    # Save final model
    final_path = os.path.join(MODEL_DIR, "pps_ppo_final")
    model.save(final_path)

    # Save training state for resume continuity
    with open(TRAIN_STATE_FILE, "w") as f:
        json.dump({
            "episode_count": metrics_cb._episode_count,
            "best_throughput": metrics_cb._best_throughput,
            "run_name": run_name,
            "plan_total_timesteps": plan_total_timesteps,
        }, f)

    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best throughput: {metrics_cb._best_throughput}")
    print(f"Total episodes so far: {metrics_cb._episode_count}")

    tb_writer.close()
    vec_env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(n_episodes: int = 5):
    """Evaluate trained PPS model and print per-episode results."""
    if not os.path.exists(BEST_MODEL + ".zip"):
        print(f"No model found at {BEST_MODEL}. Train first.")
        return

    env = PPSEnv()
    model = PPO.load(BEST_MODEL, device="cpu")

    results = []
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        results.append(info)
        print(
            f"Episode {ep}/{n_episodes}: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"throughput={info.get('throughput', 0)}, "
            f"pile_on={info.get('pile_on_rate', 0):.2f}, "
            f"avg_oct={info.get('avg_order_completion_time', 0):.1f}, "
            f"path_cost={info.get('cumulative_path_cost', 0):.1f}"
        )

    # Summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({n_episodes} episodes)")
    avg_tp = np.mean([r.get("throughput", 0) for r in results])
    avg_po = np.mean([r.get("pile_on_rate", 0) for r in results])
    avg_oct = np.mean([r.get("avg_order_completion_time", 0) for r in results])
    avg_cpc = np.mean([r.get("cumulative_path_cost", 0) for r in results])
    print(f"  Avg throughput          : {avg_tp:.1f}")
    print(f"  Avg pile-on rate        : {avg_po:.2f}")
    print(f"  Avg order completion    : {avg_oct:.1f} ticks")
    print(f"  Avg cumulative path cost: {avg_cpc:.1f}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPS RL agent")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate trained model")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Approximate number of episodes to train")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate (start of linear decay)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="Final learning rate (default: lr * 0.1)")
    parser.add_argument("--lr-plan-episodes", type=int, default=None,
                        help="LR decay horizon in episodes. Persisted across "
                             "resumes. Default on first run = --episodes.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Override model save directory")
    parser.add_argument("--max-ticks", type=int, default=32400,
                        help="Max simulation ticks per episode (default 32400 = 9h)")
    parser.add_argument("--n-steps", type=int, default=4096,
                        help="Steps per PPO rollout (set >= typical episode length)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel envs (SubprocVecEnv). 1 = serial.")

    args = parser.parse_args()

    # Override paths if --save-path provided
    if args.save_path:
        save_dir = os.path.dirname(args.save_path) or MODEL_DIR
        MODEL_DIR = save_dir
        LOG_DIR = os.path.join(save_dir, "runs")
        METRICS_DIR = os.path.join(save_dir, "metrics")
        BEST_MODEL = os.path.splitext(args.save_path)[0] + "_best"
        CHECKPOINT_DIR = os.path.join(save_dir, "checkpoints")

    if args.eval:
        evaluate()
    else:
        train(
            total_episodes=args.episodes,
            resume=args.resume,
            learning_rate=args.lr,
            lr_end=args.lr_end,
            lr_plan_episodes=args.lr_plan_episodes,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            max_episode_ticks=args.max_ticks,
            n_steps=args.n_steps,
            n_envs=args.n_envs,
        )
