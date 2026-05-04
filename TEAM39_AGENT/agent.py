import os
import pickle
from pathlib import Path
from typing import Dict
import warnings

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default"  # this may be useful when training with selfplay
MODULE_DIR = Path(__file__).resolve().parent


def _checkpoint_candidates():
    # Common export layouts:
    # - TEAM_AGENT/checkpoint/checkpoint-<n>
    # - TEAM_AGENT/checkpoint_<xxxxx>/checkpoint-<n>
    patterns = [
        str(MODULE_DIR / "checkpoint" / "checkpoint-*"),
        str(MODULE_DIR / "checkpoint_*" / "checkpoint-*"),
        str(MODULE_DIR / "ray_results" / "**" / "checkpoint_*" / "checkpoint-*"),
    ]
    candidates = []
    for pattern in patterns:
        for p in sorted(MODULE_DIR.glob(pattern.replace(str(MODULE_DIR) + "/", ""))):
            # Skip metadata sidecars.
            if p.name.endswith(".tune_metadata"):
                continue
            candidates.append(p)
    # Prefer larger files first (tiny/truncated files often fail unpickle).
    return sorted(
        [p for p in candidates if p.is_file()],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )


def _pick_checkpoint_path() -> Path:
    candidates = _checkpoint_candidates()
    return candidates[0] if candidates else None


def _pick_params_path(checkpoint_path: Path):
    candidates = [
        checkpoint_path.parent / "params.pkl" if checkpoint_path else None,
        checkpoint_path.parent.parent / "params.pkl" if checkpoint_path else None,
        MODULE_DIR / "params.pkl",
    ]
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def _train_env_config(config: dict) -> dict:
    ec = config.get("env_config")
    if isinstance(ec, dict):
        return ec
    inner = config.get("config")
    if isinstance(inner, dict):
        ec = inner.get("env_config")
        if isinstance(ec, dict):
            return ec
    return {}


def _dummy_spaces_from_train_config(config: dict, env: gym.Env):
    """
    Autograder `env` may use MultiDiscrete (RLlib logits dim 9) while training with
    flatten_branched=True uses Discrete(27). Dummy env for restore must match training.
    """
    obs_space = env.observation_space
    act_space = env.action_space
    ec = _train_env_config(config)
    if not ec:
        return obs_space, act_space
    if ec.get("flatten_branched"):
        n = 27
        if hasattr(env.action_space, "nvec"):
            n = int(np.prod(env.action_space.nvec))
        act_space = gym.spaces.Discrete(n)
    return obs_space, act_space


def _resolve_policy(agent: PPOTrainer):
    for pid in ("default_policy", POLICY_NAME):
        try:
            return agent.get_policy(pid)
        except Exception:
            continue
    return agent.get_policy("default_policy")


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.policy = None
        self._fallback_action = env.action_space.sample()
        self._action_flattener = None
        if hasattr(env.action_space, "nvec"):
            try:
                from gym_unity.envs import ActionFlattener

                self._action_flattener = ActionFlattener(env.action_space.nvec)
            except Exception:
                pass

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                local_mode=True,
                num_cpus=1,
            )

        checkpoint_path = _pick_checkpoint_path()
        params_path = _pick_params_path(checkpoint_path)
        if checkpoint_path is None or params_path is None:
            warnings.warn(
                "TEAM_AGENT checkpoint/params files are missing. "
                "Falling back to random actions. Include checkpoint-* and params.pkl "
                "in the submission package for trained behavior."
            )
            return

        with open(params_path, "rb") as f:
            config = pickle.load(f)

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        # Newer RLlib supports this flag, but older autograder Ray versions reject
        # unknown config keys. Remove it to keep compatibility.
        config.pop("disable_env_checking", None)

        # Fast path for newer Ray checkpoints.
        try:
            from ray.rllib.policy.policy import Policy

            restored = Policy.from_checkpoint(str(checkpoint_path))
            if isinstance(restored, dict):
                self.policy = (
                    restored.get("default_policy")
                    or restored.get(POLICY_NAME)
                    or next(iter(restored.values()))
                )
            else:
                self.policy = restored
            if self.policy is not None:
                return
        except Exception:
            pass

        # RLlib requires MultiAgentEnv when the checkpoint has multiple policies
        # (e.g. self-play with default + opponent_*). A plain gym.Env raises
        # ValueError in trainer setup on the autograder.
        multi = config.get("multiagent") or {}
        policy_names = list((multi.get("policies") or {}).keys())
        use_multi_dummy = len(policy_names) > 1

        obs_space, act_space = _dummy_spaces_from_train_config(config, env)

        if use_multi_dummy:

            class DummyMultiEnv(MultiAgentEnv):
                observation_space = obs_space
                action_space = act_space

                def reset(self):
                    if hasattr(obs_space, "sample"):
                        o = obs_space.sample()
                    else:
                        o = np.zeros((336,), dtype=np.float32)
                    return {0: o, 1: o, 2: o, 3: o}

                def step(self, action_dict):
                    o = self.reset()
                    rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    dones = {
                        0: True,
                        1: True,
                        2: True,
                        3: True,
                        "__all__": True,
                    }
                    infos = {0: {}, 1: {}, 2: {}, 3: {}}
                    return o, rewards, dones, infos

            tune.registry.register_env("DummyEnv", lambda *_: DummyMultiEnv())
        else:
            class DummyGymEnv(gym.Env):
                observation_space = obs_space
                action_space = act_space

                def reset(self):
                    if hasattr(self.observation_space, "sample"):
                        return self.observation_space.sample()
                    return np.zeros((336,), dtype=np.float32)

                def step(self, action):
                    return self.reset(), 0.0, True, {}

            tune.registry.register_env("DummyEnv", lambda *_: DummyGymEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        agent = PPOTrainer(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(str(checkpoint_path))
        # get policy for evaluation
        self.policy = _resolve_policy(agent)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        if self.policy is None:
            for player_id in observation:
                actions[player_id] = self._fallback_action
            return actions

        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            raw, *_ = self.policy.compute_single_action(observation[player_id])
            raw = np.asarray(raw)
            if self._action_flattener is not None and (
                raw.shape == () or raw.size == 1
            ):
                actions[player_id] = self._action_flattener.lookup_action(
                    int(raw.ravel()[0])
                )
            else:
                actions[player_id] = raw
        return actions
