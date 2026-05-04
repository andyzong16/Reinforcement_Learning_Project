#!/usr/bin/env python3
"""
Standard PPO self-play trainer with dense competitive reward shaping (see utils.CompetitiveRewardWrapper).

Features:
  - Multi-policy self-play (default + opponent archive)
  - Tuned reward coefficients and PPO hyperparameters
  - Ray Tune checkpoints + optional copy into TEAM_AGENT for submission

Usage:
  python train_strong_selfplay.py
  python train_strong_selfplay.py --timesteps 5000000 --workers 2 --export

  # Slurm / many runs on one node (avoid Unity port collisions):
  python train_strong_selfplay.py --base-port 16000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env

# RLlib may import cv2 on some versions even without vision — stub if missing.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    _cv2_stub = types.ModuleType("cv2")

    def _cv2_missing(*args, **kwargs):
        raise ModuleNotFoundError(
            "Install opencv-python-headless for image-based models."
        )

    _cv2_stub.resize = _cv2_missing
    _cv2_stub.cvtColor = _cv2_missing
    _cv2_stub.COLOR_RGB2GRAY = 0
    sys.modules["cv2"] = _cv2_stub

NUM_ENVS_PER_WORKER = 1
EXPERIMENT_NAME = "PPO_strong_selfplay"
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "TEAM_AGENT"

# Tuned dense reward: emphasizes ball progress + pressure, limits noise via caps in utils.
STRONG_REWARD_CONFIG = {
    "ball_progress_coef": 0.08,
    "approach_ball_coef": 0.02,
    "touch_ball_coef": 0.012,
    "behind_ball_coef": 0.01,
    "defense_shape_coef": 0.014,
    "spacing_coef": 0.005,
    "own_goal_danger_coef": 0.025,
    "step_penalty": 0.0008,
    "touch_distance": 1.6,
    "ideal_min_spacing": 2.0,
    "ideal_max_spacing": 8.0,
    "max_abs_dense_reward_per_agent": 0.22,
}


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3", "opponent_4"],
        p=[0.42, 0.22, 0.16, 0.12, 0.08],
    )


class SelfPlayUpdateCallback(DefaultCallbacks):
    """Rotate frozen opponents when the learned policy is clearly improving."""

    def on_train_result(self, **info):
        result = info.get("result") or {}
        trainer = info.get("trainer")
        if trainer is None:
            return
        rew = result.get("episode_reward_mean")
        if rew is None:
            return
        if rew > 0.65:
            print("---- Rotating opponent archive ----")
            w = trainer.get_weights(
                ["default", "opponent_1", "opponent_2", "opponent_3"]
            )
            trainer.set_weights(
                {
                    "opponent_4": w["opponent_3"],
                    "opponent_3": w["opponent_2"],
                    "opponent_2": w["opponent_1"],
                    "opponent_1": w["default"],
                }
            )


def export_best_checkpoint_to_team_agent(checkpoint_path: str, dest: Path) -> None:
    """Copy best RLlib checkpoint + params.pkl into TEAM_AGENT layout for autograder."""
    ck = Path(checkpoint_path).resolve()
    if not ck.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ck}")
    run_dir = ck.parent.parent
    params_src = run_dir / "params.pkl"
    if not params_src.is_file():
        params_src = ck.parent / "params.pkl"
    if not params_src.is_file():
        print("Warning: params.pkl not found next to run; export may be incomplete.")

    dest.mkdir(parents=True, exist_ok=True)
    inner = dest / "checkpoint_000001"
    inner.mkdir(parents=True, exist_ok=True)
    for item in ck.parent.iterdir():
        if item.is_file():
            shutil.copy2(item, inner / item.name)
    if params_src.is_file():
        shutil.copy2(params_src, dest / "params.pkl")
        shutil.copy2(params_src, inner / "params.pkl")
    (dest / "best_checkpoint.txt").write_text(
        "checkpoint_000001/" + ck.name + "\n", encoding="utf-8"
    )
    print(f"Exported checkpoint to {dest} (see checkpoint_000001/ and params.pkl)")


def parse_args():
    p = argparse.ArgumentParser(description="PPO self-play with strong shaped reward")
    p.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TRAIN_TIMESTEPS", 5_000_000)),
        help="Total environment steps (tune stop timesteps_total)",
    )
    p.add_argument("--workers", type=int, default=1, help="RLlib num_workers")
    p.add_argument(
        "--name", type=str, default=EXPERIMENT_NAME, help="Ray experiment / folder name"
    )
    p.add_argument(
        "--export",
        action="store_true",
        help="After training, copy best checkpoint into TEAM_AGENT",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Target folder for --export (default: ./TEAM_AGENT)",
    )
    p.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Unity base_port (default: 15000 + random offset)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (numpy/ray)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    base_port = args.base_port
    if base_port is None:
        base_port = 15000 + (os.getpid() % 8000) + random.randint(0, 500)

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
    )

    tune.registry.register_env("SoccerStrong", create_rllib_env)

    temp = create_rllib_env(
        {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "use_custom_reward": True,
            "reward_config": STRONG_REWARD_CONFIG,
            "base_port": base_port,
        }
    )
    obs_space = temp.observation_space
    act_space = temp.action_space
    temp.close()

    # Scale batch to workers+driver sampling (1 local worker in typical setup).
    n_parallel = max(1, args.workers) * NUM_ENVS_PER_WORKER
    train_batch = min(24_000, 4000 * n_parallel)
    frag_len = min(3000, max(1000, train_batch // max(1, n_parallel * 2)))

    analysis = tune.run(
        "PPO",
        name=args.name,
        local_dir="./ray_results",
        checkpoint_freq=50,
        checkpoint_at_end=True,
        stop={"timesteps_total": args.timesteps},
        config={
            "env": "SoccerStrong",
            "framework": "torch",
            "log_level": "INFO",
            "num_gpus":0,
            "num_workers": args.workers,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "callbacks": SelfPlayUpdateCallback,
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "use_custom_reward": True,
                "reward_config": STRONG_REWARD_CONFIG,
                "base_port": base_port,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": frag_len,
            "train_batch_size": train_batch,
            "sgd_minibatch_size": min(2048, max(256, train_batch // 4)),
            "num_sgd_iter": 10,
            "batch_mode": "complete_episodes",
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.005,
            "vf_loss_coeff": 1.0,
            "grad_clip": 0.5,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                    "opponent_4": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
        },
    )

    best = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(
        trial=best, metric="episode_reward_mean", mode="max"
    )
    print("\nBest trial:", best)
    print("Best checkpoint:", best_ckpt)

    if args.export and best_ckpt:
        export_best_checkpoint_to_team_agent(str(best_ckpt), args.export_dir)
    print("Done.")


if __name__ == "__main__":
    main()
