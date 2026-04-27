"""
PPO training script for Joint POA + PPS in RMFS.

Single centralized agent controls both order-to-station assignment (POA)
and pod selection (PPS) in one action: [order_index, station_index, pod_index].

Uses Stable-Baselines3 PPO with:
  - Custom Gymnasium env (JointEnv)
  - TensorBoard logging (joint_reward/ and joint_metrics/ sections)
  - CSV data recording
  - Model save/load + checkpoints

Usage:
    python train_joint_rl.py --episodes 500 --max-ticks 3000
    python train_joint_rl.py --resume --episodes 500
    python train_joint_rl.py --eval
    tensorboard --logdir saved_models/joint_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from joint_env import JointEnv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join("saved_models", "joint")
LOG_DIR = os.path.join("saved_models", "joint_runs")
METRICS_DIR = os.path.join("saved_models", "joint_metrics")
BEST_MODEL = os.path.join(MODEL_DIR, "joint_ppo_best")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
TRAIN_STATE_FILE = os.path.join(MODEL_DIR, "train_state.json")


# ---------------------------------------------------------------------------
# CSV Metrics Recorder
# ---------------------------------------------------------------------------
class MetricsRecorder:
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
# Custom Callback
# ---------------------------------------------------------------------------
class JointMetricsCallback(BaseCallback):
    """
    TensorBoard sections:
      joint_reward/  : total_reward, pile_on_rate, avg_order_completion_time
      joint_metrics/ : throughput, cumulative_path_cost, episode_steps
    """

    def __init__(
        self,
        tb_writer: SummaryWriter | None = None,
        recorder: MetricsRecorder | None = None,
        verbose: int = 0,
        episode_offset: int = 0,
        best_throughput: int = 0,
    ):
        super().__init__(verbose)
        self._tb_writer = tb_writer
        self._recorder = recorder
        self._best_throughput = best_throughput
        self._episode_count = episode_offset
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._ep_start = time.time()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        if len(rewards) > 0:
            self._episode_reward += float(rewards[0])
        self._episode_steps += 1

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            if i < len(dones) and dones[i]:
                self._episode_count += 1
                ep = self._episode_count
                ep_elapsed = time.time() - self._ep_start

                throughput = info.get("throughput", 0)
                pile_on = info.get("pile_on_rate", 0.0)
                avg_oct = info.get("avg_order_completion_time", 0.0)
                cpc = info.get("cumulative_path_cost", 0.0)

                # TensorBoard
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "joint_reward/total_reward", self._episode_reward, ep
                    )
                    self._tb_writer.add_scalar(
                        "joint_reward/pile_on_rate", pile_on, ep
                    )
                    self._tb_writer.add_scalar(
                        "joint_reward/avg_order_completion_time", avg_oct, ep
                    )
                    self._tb_writer.add_scalar(
                        "joint_metrics/throughput", throughput, ep
                    )
                    self._tb_writer.add_scalar(
                        "joint_metrics/cumulative_path_cost", cpc, ep
                    )
                    self._tb_writer.add_scalar(
                        "joint_metrics/episode_steps", self._episode_steps, ep
                    )
                    self._tb_writer.flush()

                # CSV
                if self._recorder is not None:
                    self._recorder.record(
                        ep, info, self._episode_reward, self._episode_steps
                    )

                # Console
                print(
                    f"[Episode {ep:>3}] "
                    f"steps={self._episode_steps:>5}  "
                    f"reward={self._episode_reward:>8.2f}  "
                    f"throughput={throughput:>4}  "
                    f"pile_on={pile_on:.2f}  "
                    f"avg_oct={avg_oct:.1f}  "
                    f"path_cost={cpc:.1f}  "
                    f"time={ep_elapsed:.1f}s"
                )

                # Save best model
                if throughput > self._best_throughput:
                    self._best_throughput = throughput
                    self.model.save(BEST_MODEL)
                    print(f"  >> New best throughput: {throughput} - model saved!")

                # Periodic checkpoint
                if ep % 10 == 0:
                    cp_path = os.path.join(CHECKPOINT_DIR, f"joint_ppo_ep{ep}")
                    self.model.save(cp_path)

                # Reset per-episode accumulators
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._ep_start = time.time()

        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_joint_env(**kwargs):
    def _init():
        return JointEnv(**kwargs)
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    total_episodes: int = 50,
    resume: bool = False,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    max_episode_ticks: int = 3000,
    n_steps: int = 4096,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_joint_env(max_episode_ticks=max_episode_ticks)])

    # Load training state if resuming
    prev_episode_count = 0
    prev_best_throughput = 0
    run_name = None

    if resume and os.path.exists(TRAIN_STATE_FILE):
        with open(TRAIN_STATE_FILE, "r") as f:
            state = json.load(f)
        prev_episode_count = state.get("episode_count", 0)
        prev_best_throughput = state.get("best_throughput", 0)
        run_name = state.get("run_name", None)
        print(f"Restoring: {prev_episode_count} episodes, best throughput {prev_best_throughput}")

    if run_name is None:
        run_name = f"joint_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    csv_path = os.path.join(METRICS_DIR, f"metrics_{run_name}.csv")
    recorder = MetricsRecorder(csv_path)

    if resume and os.path.exists(BEST_MODEL + ".zip"):
        print(f"Resuming from {BEST_MODEL}")
        # Load model — SB3 restores weights, optimizer state (Adam momentum),
        # and all hyperparameters from the saved checkpoint automatically.
        # We do NOT override learning_rate here so the optimizer continues
        # from where it left off without instability.
        model = PPO.load(BEST_MODEL, env=vec_env, device="cpu")
        saved_lr = model.learning_rate
        print(f"  Restored learning rate: {saved_lr}")
        model.n_steps = n_steps
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
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=learning_rate,
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

    total_timesteps = n_steps * total_episodes

    print(f"\n{'='*60}")
    print(f"Joint POA+PPS RL Training")
    print(f"{'='*60}")
    print(f"  Total episodes (approx) : {total_episodes}")
    print(f"  Total timesteps         : {total_timesteps}")
    print(f"  n_steps per rollout     : {n_steps}")
    print(f"  Learning rate           : {learning_rate}")
    print(f"  Batch size              : {batch_size}")
    print(f"  PPO epochs/update       : {n_epochs}")
    print(f"  Gamma                   : {gamma}")
    print(f"  GAE lambda              : {gae_lambda}")
    print(f"  Clip range              : {clip_range}")
    print(f"  Entropy coef            : {ent_coef}")
    print(f"  Max episode ticks       : {max_episode_ticks}")
    print(f"  TensorBoard             : tensorboard --logdir {LOG_DIR}")
    print(f"  CSV metrics             : {csv_path}")
    print(f"{'='*60}\n")

    metrics_cb = JointMetricsCallback(
        tb_writer=tb_writer,
        recorder=recorder,
        verbose=1,
        episode_offset=prev_episode_count,
        best_throughput=prev_best_throughput,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_cb,
        reset_num_timesteps=not resume,
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, "joint_ppo_final")
    model.save(final_path)

    with open(TRAIN_STATE_FILE, "w") as f:
        json.dump({
            "episode_count": metrics_cb._episode_count,
            "best_throughput": metrics_cb._best_throughput,
            "run_name": run_name,
        }, f)

    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best throughput: {metrics_cb._best_throughput}")
    print(f"Total episodes: {metrics_cb._episode_count}")

    tb_writer.close()
    vec_env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(n_episodes: int = 5, max_episode_ticks: int = 3000):
    if not os.path.exists(BEST_MODEL + ".zip"):
        print(f"No model found at {BEST_MODEL}. Train first.")
        return

    env = JointEnv(max_episode_ticks=max_episode_ticks)
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
    parser = argparse.ArgumentParser(description="Train Joint POA+PPS RL agent")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate trained model")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Approximate number of episodes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--max-ticks", type=int, default=3000,
                        help="Max simulation ticks per episode")
    parser.add_argument("--n-steps", type=int, default=4096,
                        help="Steps per PPO rollout")

    args = parser.parse_args()

    if args.eval:
        evaluate(max_episode_ticks=args.max_ticks)
    else:
        train(
            total_episodes=args.episodes,
            resume=args.resume,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            max_episode_ticks=args.max_ticks,
            n_steps=args.n_steps,
        )
