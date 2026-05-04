#!/usr/bin/env python3
"""
Train PPO against the CEIA baseline (frozen opponent) in team_vs_policy mode.

Uses utils.CompetitiveRewardWrapper for dense shaping (same coefficients as train_strong_selfplay)
plus optional CEIA **imitation** reward (match CEIA's action in flattened action space).
CEIA policy is loaded from the bundled ceia_baseline_agent checkpoint (Policy.from_checkpoint when possible).

Usage:
  python train_vs_ceia.py --timesteps 2000000 --export
  CEIA_CHECKPOINT_PATH=/path/to/checkpoint-2449 python train_vs_ceia.py

  # Avoid Unity port clashes on shared nodes:
  python train_vs_ceia.py --base-port 18000 --workers 1 --export

  # Copy best-so-far checkpoint during training (whenever Tune saves a new ckpt):
  python train_vs_ceia.py --live-best-dir ./TEAM2_AGENT

Why ``episode_reward_mean`` can sit near -2:
  RLlib reports the **sum of per-step rewards** over an episode. CEIA wins most games, so the
  **sparse** soccer signal stays strongly negative; dense + imitation nudge the mean only a bit.
  That does **not** mean gradients are zero (watch ``policy_loss``, ``vf_explained_var``).

How to improve results:
  * Run **multi-million** timesteps (e.g. 5–20M); tens of thousands is too little for vs-CEIA.
  * **Pretrain** with ``train_strong_selfplay.py`` (self-play), then fine-tune here, or alternate.
  * Keep ``imitation_mismatch_penalty`` at **0** unless you know you want pessimistic BC pressure.
  * Pick checkpoints by **eval win rate** vs CEIA in Unity, not only ``episode_reward_mean``.
  * If learning stalls: raise ``entropy_coeff`` slightly, or lower ``lr``; add workers if RAM allows.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune.callback import Callback
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
import gym
import soccer_twos
from soccer_twos import EnvType

from utils import create_rllib_env

# RLlib may import cv2 at import time on some installs.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    _cv2 = types.ModuleType("cv2")

    def _missing(*a, **k):
        raise ModuleNotFoundError("pip install opencv-python-headless")

    _cv2.resize = _missing
    _cv2.cvtColor = _missing
    _cv2.COLOR_RGB2GRAY = 0
    sys.modules["cv2"] = _cv2

NUM_ENVS_PER_WORKER = 1
EXPERIMENT_NAME = "PPO_vs_ceia"
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "TEAM_AGENT"
POLICY_NAME = "default"

_DEFAULT_CEIA = (
    Path(__file__).resolve().parent
    / "ceia_baseline_agent"
    / "ray_results"
    / "PPO_selfplay_twos"
    / "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    / "checkpoint_002449"
    / "checkpoint-2449"
)

# Match train_strong_selfplay shaping (competitive dense reward).
# Imitation: reward matching CEIA's flat action; keep mismatch_penalty at 0 so you do not
# subtract ~0.003 every step when the student disagrees with CEIA (that drags means toward -2).
REWARD_CONFIG = {
    "ball_progress_coef": 0.10,
    "approach_ball_coef": 0.025,
    "touch_ball_coef": 0.014,
    "behind_ball_coef": 0.012,
    "defense_shape_coef": 0.016,
    "spacing_coef": 0.005,
    "own_goal_danger_coef": 0.025,
    "step_penalty": 0.0005,
    "touch_distance": 1.6,
    "ideal_min_spacing": 2.0,
    "ideal_max_spacing": 8.0,
    "max_abs_dense_reward_per_agent": 0.26,
    "imitation_coef": 0.035,
    "imitation_mismatch_penalty": 0.0,
}

_CEIA_POLICY = None
# CEIA checkpoint uses MultiDiscrete([3,3,3]); flatten_branched env expects flat int indices.
_CEIA_ACTION_NVEC: tuple = ()


def _cache_ceia_action_space(policy) -> None:
    global _CEIA_ACTION_NVEC
    sp = policy.action_space
    if hasattr(sp, "nvec"):
        _CEIA_ACTION_NVEC = tuple(int(x) for x in sp.nvec)
    elif isinstance(sp, gym.spaces.Discrete):
        _CEIA_ACTION_NVEC = ()
    else:
        _CEIA_ACTION_NVEC = (3, 3, 3)


def _ceia_action_to_flat_int(action) -> int:
    """Map RLlib MultiDiscrete output to ActionFlattener flat index (same order as itertools.product)."""
    a = np.asarray(action, dtype=np.int64).ravel()
    if _CEIA_ACTION_NVEC:
        nvec = _CEIA_ACTION_NVEC
        if a.size != len(nvec):
            a = a[: len(nvec)]
        flat = 0
        for i in range(len(nvec)):
            flat = flat * int(nvec[i]) + int(a[i])
        return int(flat)
    return int(a.ravel()[0])


def ceia_teacher_flat_action(obs: np.ndarray) -> int:
    """
    Expert action in flattened SoccerTwos space (same encoding as flatten_branched training).
    Used as a behavioral-cloning shaping signal on the blue team.
    """
    pol = _load_ceia_policy()
    if not _CEIA_ACTION_NVEC:
        _cache_ceia_action_space(pol)
    a, *_ = pol.compute_single_action(np.asarray(obs, dtype=np.float32))
    return _ceia_action_to_flat_int(a)


def _ceia_checkpoint_path() -> str:
    p = os.environ.get("CEIA_CHECKPOINT_PATH", "").strip()
    if p:
        return os.path.abspath(p)
    return str(_DEFAULT_CEIA.resolve())


def _load_ceia_policy():
    """Load CEIA PPO policy once per process (Ray worker)."""
    global _CEIA_POLICY
    if _CEIA_POLICY is not None:
        return _CEIA_POLICY

    ckpt = _ceia_checkpoint_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"CEIA checkpoint not found: {ckpt}\n"
            "Set CEIA_CHECKPOINT_PATH or install ceia_baseline_agent under repo root."
        )

    try:
        from ray.rllib.policy.policy import Policy

        restored = Policy.from_checkpoint(ckpt)
        if isinstance(restored, dict):
            _CEIA_POLICY = restored.get(POLICY_NAME) or next(iter(restored.values()))
        else:
            _CEIA_POLICY = restored
        _cache_ceia_action_space(_CEIA_POLICY)
        return _CEIA_POLICY
    except Exception:
        pass

    config_dir = os.path.dirname(ckpt)
    params_path = os.path.join(config_dir, "params.pkl")
    if not os.path.isfile(params_path):
        params_path = os.path.join(os.path.dirname(config_dir), "params.pkl")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"CEIA params.pkl not found near {ckpt}")

    with open(params_path, "rb") as f:
        config = pickle.load(f)

    multi = config.get("multiagent") or {}
    policies = multi.get("policies") or {}
    spec = policies.get(POLICY_NAME) or (
        next(iter(policies.values())) if policies else None
    )
    obs_space = spec[1] if spec and len(spec) > 1 else None
    act_space = spec[2] if spec and len(spec) > 2 else None

    class DummyMA(MultiAgentEnv):
        observation_space = obs_space
        action_space = act_space

        def reset(self):
            if obs_space is not None and hasattr(obs_space, "sample"):
                o = obs_space.sample()
            else:
                o = np.zeros((336,), dtype=np.float32)
            return {0: o, 1: o, 2: o, 3: o}

        def step(self, action_dict):
            o = self.reset()
            r = {i: 0.0 for i in (0, 1, 2, 3)}
            d = {i: True for i in (0, 1, 2, 3)}
            d["__all__"] = True
            i = {k: {} for k in (0, 1, 2, 3)}
            return o, r, d, i

    config["num_workers"] = 0
    config["num_gpus"] = 0
    config.pop("disable_env_checking", None)
    tune.registry.register_env("CeiaDummyMA", lambda *_: DummyMA())
    config["env"] = "CeiaDummyMA"

    trainer = PPOTrainer(env=config["env"], config=config)
    trainer.restore(ckpt)
    _CEIA_POLICY = trainer.get_policy(POLICY_NAME)
    _cache_ceia_action_space(_CEIA_POLICY)
    return _CEIA_POLICY


def ceia_opponent_policy(observation: np.ndarray, *args, **kwargs):
    pol = _load_ceia_policy()
    if not _CEIA_ACTION_NVEC:
        _cache_ceia_action_space(pol)
    action, *_ = pol.compute_single_action(observation)
    return _ceia_action_to_flat_int(action)


def export_best_checkpoint_to_team_agent(checkpoint_path: str, dest: Path) -> None:
    ck = Path(checkpoint_path).resolve()
    if not ck.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ck}")
    run_dir = ck.parent.parent
    params_src = run_dir / "params.pkl"
    if not params_src.is_file():
        params_src = ck.parent / "params.pkl"

    import shutil

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
    print(f"Exported to {dest}")


class LiveBestCheckpointCallback(Callback):
    """
    Whenever Tune writes a persistent checkpoint, if ``episode_reward_mean`` (or chosen
    metric) is the best so far, copy that checkpoint + params into ``dest_dir`` in the
    same layout as ``export_best_checkpoint_to_team_agent`` (usable while training).
    """

    def __init__(
        self,
        dest_dir: Path,
        metric: str = "episode_reward_mean",
        mode: str = "max",
    ):
        super().__init__()
        self.dest_dir = Path(dest_dir).resolve()
        self.metric = metric
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")

    def _is_better(self, score: float) -> bool:
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return False
        if self.mode == "max":
            return score > self.best
        return score < self.best

    def _maybe_export(self, trial) -> None:
        result = trial.last_result or {}
        raw = result.get(self.metric)
        if raw is None:
            return
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return
        if not self._is_better(score):
            return
        ck = trial.checkpoint
        path = getattr(ck, "value", None) if ck is not None else None
        if not path or not os.path.isfile(path):
            return
        self.best = score
        try:
            export_best_checkpoint_to_team_agent(str(path), self.dest_dir)
            print(
                f"[live-best] {self.metric}={score:.5f} -> {self.dest_dir}",
                flush=True,
            )
        except Exception as e:
            print(f"[live-best] export failed: {e}", flush=True)

    def on_trial_save(self, iteration, trials, trial, **info):
        self._maybe_export(trial)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._maybe_export(trial)


def parse_args():
    p = argparse.ArgumentParser(description="PPO vs CEIA baseline with shaped reward")
    p.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TRAIN_TIMESTEPS", 5_000_000)),
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", type=str, default=EXPERIMENT_NAME)
    p.add_argument("--export", action="store_true")
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p.add_argument("--base-port", type=int, default=None)
    p.add_argument(
        "--ceia-checkpoint",
        type=str,
        default=None,
        help="Path to CEIA checkpoint file (sets CEIA_CHECKPOINT_PATH)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--live-best-dir",
        type=Path,
        default=None,
        help=(
            "If set, after each Tune checkpoint save, export the best-so-far trial "
            "(by episode_reward_mean) to this directory in TEAM_AGENT layout."
        ),
    )
    p.add_argument(
        "--live-best-metric",
        type=str,
        default="episode_reward_mean",
        help="Metric for live-best export (default: episode_reward_mean).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.ceia_checkpoint:
        os.environ["CEIA_CHECKPOINT_PATH"] = os.path.abspath(args.ceia_checkpoint)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Validate CEIA exists before long Ray run
    if not os.path.isfile(_ceia_checkpoint_path()):
        raise SystemExit(
            f"Missing CEIA checkpoint at {_ceia_checkpoint_path()!r}. "
            "Pass --ceia-checkpoint or set CEIA_CHECKPOINT_PATH."
        )

    base_port = args.base_port
    if base_port is None:
        base_port = 15000 + (os.getpid() % 8000) + random.randint(0, 500)

    # Dashboard is off; raylet metrics agent may still warn on some HPC nodes (harmless).
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
    )

    tune.registry.register_env("SoccerVsCeia", create_rllib_env)

    env_cfg = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": True,
        "flatten_branched": True,
        "opponent_policy": ceia_opponent_policy,
        "use_custom_reward": True,
        "reward_config": REWARD_CONFIG,
        "imitation_teacher_fn": ceia_teacher_flat_action,
        "base_port": base_port,
    }

    temp = create_rllib_env(dict(env_cfg))
    obs_space = temp.observation_space
    act_space = temp.action_space
    temp.close()

    n_parallel = max(1, args.workers) * NUM_ENVS_PER_WORKER
    train_batch = min(24_000, 4000 * n_parallel)
    frag_len = min(3000, max(1000, train_batch // max(1, n_parallel * 2)))

    callbacks = []
    if args.live_best_dir is not None:
        callbacks.append(
            LiveBestCheckpointCallback(
                args.live_best_dir,
                metric=args.live_best_metric,
                mode="max",
            )
        )

    analysis = tune.run(
        "PPO",
        name=args.name,
        local_dir="./ray_results",
        checkpoint_freq=50,
        checkpoint_at_end=True,
        stop={"timesteps_total": args.timesteps},
        callbacks=callbacks or None,
        config={
            "env": "SoccerVsCeia",
            "framework": "torch",
            "log_level": "INFO",
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": args.workers,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "env_config": env_cfg,
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
            # Slightly more exploration early vs a strong fixed opponent.
            "entropy_coeff": 0.008,
            # vf_share_layers=True under-fits the value head unless vf_loss is higher than 1.
            "vf_loss_coeff": 3.0,
            "grad_clip": 0.5,
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
