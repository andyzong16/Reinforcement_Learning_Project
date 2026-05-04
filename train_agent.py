import os
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1
EXPERIMENT_NAME = "PPO_selfplay_custom_reward"


def policy_mapping_fn(agent_id, *args, **kwargs):
    # Train blue-side policy "default"; sample current or archived opponents for the other side.
    if agent_id == 0:
        return "default"
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3", "opponent_4"],
            p=[0.40, 0.25, 0.15, 0.10, 0.10],
        )


class SelfPlayUpdateCallback(DefaultCallbacks):
    """
    Rotates archived opponents when training reward gets strong enough.
    """

    def on_train_result(self, **info):
        result = info["result"]
        trainer = info["trainer"]

        rew = result.get("episode_reward_mean", None)
        if rew is None:
            return

        # You can tighten this once training stabilizes.
        if rew > 0.75:
            print("---- Rotating opponent archive ----")
            weights = trainer.get_weights(
                ["default", "opponent_1", "opponent_2", "opponent_3"]
            )
            trainer.set_weights(
                {
                    "opponent_4": weights["opponent_3"],
                    "opponent_3": weights["opponent_2"],
                    "opponent_2": weights["opponent_1"],
                    "opponent_1": weights["default"],
                }
            )


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    tune.registry.register_env("SoccerCustom", create_rllib_env)

    base_port = 15000 + (os.getpid() % 10000)

    temp_env = create_rllib_env(
        {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "use_custom_reward": True,
            "base_port": base_port,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    reward_config = {
        "ball_progress_coef": 0.060,
        "approach_ball_coef": 0.015,
        "touch_ball_coef": 0.010,
        "behind_ball_coef": 0.008,
        "defense_shape_coef": 0.012,
        "spacing_coef": 0.004,
        "own_goal_danger_coef": 0.020,
        "step_penalty": 0.001,
        "touch_distance": 1.75,
        "ideal_min_spacing": 2.0,
        "ideal_max_spacing": 8.0,
        "max_abs_dense_reward_per_agent": 0.20,
    }

    analysis = tune.run(
        "PPO",
        name=EXPERIMENT_NAME,
        local_dir="./ray_results",
        checkpoint_freq=50,
        checkpoint_at_end=True,
        stop={
            "timesteps_total": 2000,
            # or use a time cap if you want:
            # "time_total_s": 4 * 3600,
        },
        config={
            # --------------------
            # system
            # --------------------
            "env": "SoccerCustom",
            "framework": "torch",
            "log_level": "INFO",
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "callbacks": SelfPlayUpdateCallback,

            # --------------------
            # env config
            # --------------------
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "use_custom_reward": True,
                "reward_config": reward_config,
                "base_port": base_port,
            },

            # --------------------
            # model
            # --------------------
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
            },

            # --------------------
            # PPO hyperparameters
            # --------------------
            "rollout_fragment_length": 3000,
            "train_batch_size": 24000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 10,
            "batch_mode": "complete_episodes",

            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.003,
            "vf_loss_coeff": 1.0,
            "grad_clip": 0.5,

            # --------------------
            # self-play
            # --------------------
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

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print("\nBest trial:\n", best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric="episode_reward_mean",
        mode="max",
    )
    print("\nBest checkpoint:\n", best_checkpoint)
    print("\nDone training.")