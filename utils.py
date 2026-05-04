from random import uniform as randfloat
from typing import Any, Callable, Dict, Optional
import math

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


# =========================
# RLlib wrappers
# =========================

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    RLlib-compatible wrapper.
    """
    pass


class CustomRewardRLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    RLlib-compatible wrapper used when reward shaping is enabled.
    """
    pass


# =========================
# Helper functions
# =========================

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _vec2(v, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
        return np.asarray([float(v[0]), float(v[1])], dtype=np.float32)
    return default


def _team_of(agent_id: int) -> int:
    # 0,1 = blue team ; 2,3 = orange team
    return 0 if agent_id in (0, 1) else 1


def _teammates(agent_id: int):
    return [0, 1] if agent_id in (0, 1) else [2, 3]


def _own_goal_x(team: int) -> float:
    # Blue starts on the left, Orange on the right.
    return -16.0 if team == 0 else 16.0


def _opp_goal_x(team: int) -> float:
    return 16.0 if team == 0 else -16.0


def _attack_direction(team: int) -> float:
    # Blue attacks +x, Orange attacks -x
    return 1.0 if team == 0 else -1.0


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# =========================
# Reward shaping wrapper
# =========================

class CompetitiveRewardWrapper(gym.Wrapper):
    """
    Adds dense team-level reward shaping while preserving the environment's sparse goal reward.

    It uses the extra `info` returned by soccer_twos step():
      - ball_info.position, ball_info.velocity
      - player_info.position, player_info.velocity, player_info.rotation_y

    Optional behavioral cloning: pass ``teacher_policy_fn(obs_np) -> int`` (flat action in the
    same space as ``flatten_branched`` Discrete) and set ``imitation_coef`` /
    ``imitation_mismatch_penalty`` in ``reward_config`` to reward matching the teacher.

    This wrapper is intentionally defensive:
      - if any field is missing, that shaping term becomes zero
      - sparse env reward always remains present
    """

    def __init__(
        self,
        env,
        reward_config: Optional[Dict[str, float]] = None,
        teacher_policy_fn: Optional[Callable[[np.ndarray], int]] = None,
    ):
        super().__init__(env)

        default_cfg = {
            # dense reward coefficients
            "ball_progress_coef": 0.060,
            "approach_ball_coef": 0.015,
            "touch_ball_coef": 0.010,
            "behind_ball_coef": 0.008,
            "defense_shape_coef": 0.012,
            "spacing_coef": 0.004,
            "own_goal_danger_coef": 0.020,
            "step_penalty": 0.001,

            # geometry
            "field_goal_x": 16.0,
            "danger_zone_x": 10.0,
            "touch_distance": 1.75,
            "ideal_min_spacing": 2.0,
            "ideal_max_spacing": 8.0,

            # safety
            "max_abs_dense_reward_per_agent": 0.20,

            # optional: behavioral cloning signal vs a frozen teacher (see teacher_policy_fn)
            "imitation_coef": 0.0,
            "imitation_mismatch_penalty": 0.0,
        }

        self.cfg = default_cfg
        if reward_config:
            self.cfg.update(reward_config)

        # Callable(obs_np) -> int flat action in the same encoding as the learner (Discrete 0..n-1).
        self.teacher_policy_fn: Optional[Callable[[np.ndarray], int]] = teacher_policy_fn

        self.prev_obs = None
        self.prev_info = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.prev_info = None
        return obs

    def step(self, action):
        imitation = self._imitation_blue_additions(action)

        obs, reward, done, info = self.env.step(action)

        # team_vs_policy + single_player returns scalar (sum of blue rewards). Still apply
        # dense shaping for agents 0+1 so training signal is not only sparse {-2, ...}.
        if not isinstance(reward, dict):
            base = float(np.asarray(reward).item())
            extra = 0.0
            if self.prev_info is not None and isinstance(info, dict):
                dense_additions = self._compute_dense_reward(self.prev_info, info)
                extra = float(dense_additions[0] + dense_additions[1])
            im_extra = float(imitation[0] + imitation[1])
            shaped_scalar = base + extra + im_extra
            self.prev_obs = obs
            self.prev_info = info
            return obs, shaped_scalar, done, info

        shaped_reward = dict(reward)

        if self.prev_info is not None and isinstance(info, dict):
            dense_additions = self._compute_dense_reward(self.prev_info, info)
            for agent_id, add_r in dense_additions.items():
                shaped_reward[agent_id] = float(shaped_reward.get(agent_id, 0.0) + add_r)

        for aid in (0, 1):
            shaped_reward[aid] = float(shaped_reward.get(aid, 0.0) + imitation[aid])

        self.prev_obs = obs
        self.prev_info = info
        return obs, shaped_reward, done, info

    def _student_action_flat(self, action) -> int:
        """Map env action (Discrete index or MultiDiscrete branch vector) to flat int."""
        sp = self.env.action_space
        a = np.asarray(action, dtype=np.int64)
        if isinstance(sp, gym.spaces.Discrete):
            return int(a.ravel()[0])
        if hasattr(sp, "nvec"):
            a = a.ravel()[: len(sp.nvec)]
            nvec = tuple(int(x) for x in sp.nvec)
            flat = 0
            for i in range(len(nvec)):
                flat = flat * nvec[i] + int(a[i])
            return int(flat)
        return int(a.ravel()[0])

    def _imitation_bonus_pair(self, obs_np: np.ndarray, action) -> float:
        if self.teacher_policy_fn is None:
            return 0.0
        coef = float(self.cfg.get("imitation_coef", 0.0))
        mis = float(self.cfg.get("imitation_mismatch_penalty", 0.0))
        if coef == 0.0 and mis == 0.0:
            return 0.0
        try:
            obs_f = np.asarray(obs_np, dtype=np.float32)
            teacher = int(self.teacher_policy_fn(obs_f))
            student = self._student_action_flat(action)
            if teacher == student:
                return coef
            return -mis
        except Exception:
            return 0.0

    def _imitation_blue_additions(self, action) -> Dict[int, float]:
        """Uses self.prev_obs (state the policy saw for this action) vs teacher."""
        out = {0: 0.0, 1: 0.0}
        prev = self.prev_obs
        if prev is None:
            return out

        if isinstance(prev, dict) and isinstance(action, dict):
            for aid in (0, 1):
                if aid in prev and aid in action:
                    out[aid] = self._imitation_bonus_pair(prev[aid], action[aid])
            return out

        # Single-agent / team scalar obs: split bonus evenly across blue agents (one policy).
        b = self._imitation_bonus_pair(prev, action)
        out[0] = 0.5 * b
        out[1] = 0.5 * b
        return out

    def _extract_ball_pos(self, info_dict: Dict[int, Dict[str, Any]]):
        # ball_info is typically repeated under each agent info; take first available.
        for aid in [0, 1, 2, 3]:
            data = info_dict.get(aid, {})
            pos = _vec2(_safe_get(data, "ball_info", "position"))
            if pos is not None:
                return pos
        return None

    def _extract_player_pos(self, info_dict: Dict[int, Dict[str, Any]], agent_id: int):
        data = info_dict.get(agent_id, {})
        return _vec2(_safe_get(data, "player_info", "position"))

    def _compute_dense_reward(
        self,
        prev_info: Dict[int, Dict[str, Any]],
        info: Dict[int, Dict[str, Any]],
    ) -> Dict[int, float]:
        dense = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        prev_ball = self._extract_ball_pos(prev_info)
        ball = self._extract_ball_pos(info)

        if prev_ball is None or ball is None:
            return dense

        # ---------- Team-level shaping ----------
        team_dense = {0: 0.0, 1: 0.0}

        for team in [0, 1]:
            direction = _attack_direction(team)
            own_goal_x = _own_goal_x(team)
            opp_goal_x = _opp_goal_x(team)

            # 1) Ball progress toward opponent goal
            # Positive if the ball moves in your attacking direction.
            ball_progress = float((ball[0] - prev_ball[0]) * direction)
            team_dense[team] += self.cfg["ball_progress_coef"] * _clip(ball_progress, -1.0, 1.0)

            # 2) Own-goal danger penalty
            # Penalize when the ball is deep on your defensive side.
            defensive_side = (ball[0] * direction) < 0.0
            if defensive_side:
                dist_to_own_goal = abs(ball[0] - own_goal_x)
                danger = _clip((self.cfg["danger_zone_x"] - dist_to_own_goal) / self.cfg["danger_zone_x"], 0.0, 1.0)
                team_dense[team] -= self.cfg["own_goal_danger_coef"] * danger

            # 3) Tiny step penalty
            team_dense[team] -= self.cfg["step_penalty"]

            # ---------- Player-level shaping for each teammate ----------
            mates = _teammates(0 if team == 0 else 2)
            p0, p1 = mates

            prev_p0 = self._extract_player_pos(prev_info, p0)
            prev_p1 = self._extract_player_pos(prev_info, p1)
            cur_p0 = self._extract_player_pos(info, p0)
            cur_p1 = self._extract_player_pos(info, p1)

            if prev_p0 is not None and cur_p0 is not None:
                team_dense[team] += self._player_shape(team, prev_p0, cur_p0, prev_ball, ball)
            if prev_p1 is not None and cur_p1 is not None:
                team_dense[team] += self._player_shape(team, prev_p1, cur_p1, prev_ball, ball)

            # 4) Team spacing bonus
            if cur_p0 is not None and cur_p1 is not None:
                spacing = float(np.linalg.norm(cur_p0 - cur_p1))
                if self.cfg["ideal_min_spacing"] <= spacing <= self.cfg["ideal_max_spacing"]:
                    team_dense[team] += self.cfg["spacing_coef"]
                elif spacing < self.cfg["ideal_min_spacing"]:
                    team_dense[team] -= self.cfg["spacing_coef"]

        # Split team dense reward evenly to both teammates
        for aid in [0, 1]:
            dense[aid] += 0.5 * team_dense[0]
        for aid in [2, 3]:
            dense[aid] += 0.5 * team_dense[1]

        # Clip final dense reward for stability
        max_abs = self.cfg["max_abs_dense_reward_per_agent"]
        for aid in dense:
            dense[aid] = float(_clip(dense[aid], -max_abs, max_abs))

        return dense

    def _player_shape(self, team, prev_player, cur_player, prev_ball, ball):
        shaped = 0.0
        direction = _attack_direction(team)
        own_goal_x = _own_goal_x(team)

        # A) Approach the ball
        prev_dist = float(np.linalg.norm(prev_player - prev_ball))
        cur_dist = float(np.linalg.norm(cur_player - ball))
        improvement = _clip(prev_dist - cur_dist, -1.0, 1.0)
        shaped += self.cfg["approach_ball_coef"] * improvement

        # B) Touch / challenge bonus
        if cur_dist <= self.cfg["touch_distance"]:
            shaped += self.cfg["touch_ball_coef"]

        # C) Stay "behind" the ball relative to attack direction
        # Blue attacks right: player_x <= ball_x is usually better in attack support
        # Orange attacks left: player_x >= ball_x is usually better
        if (direction > 0 and cur_player[0] <= ball[0]) or (direction < 0 and cur_player[0] >= ball[0]):
            shaped += self.cfg["behind_ball_coef"]

        # D) Defensive recovery / between-ball-and-goal shaping
        # Reward being between the ball and own goal when ball is on the defensive side.
        ball_on_defensive_side = (ball[0] * direction) < 0.0
        if ball_on_defensive_side:
            if direction > 0:
                between = own_goal_x <= cur_player[0] <= ball[0]
            else:
                between = ball[0] <= cur_player[0] <= own_goal_x
            if between:
                shaped += self.cfg["defense_shape_coef"]

        return shaped


# =========================
# Factory used by RLlib
# =========================

def create_rllib_env(env_config: dict = {}):
    """
    Creates an RLlib environment and prepares it to be instantiated by Ray workers.

    Recognized env_config keys:
      - variation
      - opponent_policy
      - multiagent
      - use_custom_reward: bool
      - reward_config: dict
      - imitation_teacher_fn: optional callable(obs_np) -> int flat action (BC signal)
    """
    if hasattr(env_config, "worker_index"):
        # Keep worker ids away from low numbers to avoid collisions with
        # lingering Unity workers from previous crashed runs.
        env_config["worker_id"] = 1000 + (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    use_custom_reward = bool(env_config.get("use_custom_reward", False))
    reward_config = env_config.get("reward_config", {})

    # soccer_twos.make only needs the env-related keys
    soccer_env_config = dict(env_config)
    soccer_env_config.pop("use_custom_reward", None)
    soccer_env_config.pop("reward_config", None)
    soccer_env_config.pop("imitation_teacher_fn", None)

    env = soccer_twos.make(**soccer_env_config)

    if use_custom_reward:
        env = CompetitiveRewardWrapper(
            env,
            reward_config=reward_config,
            teacher_policy_fn=env_config.get("imitation_teacher_fn"),
        )

    if "multiagent" in env_config and not env_config["multiagent"]:
        return env

    if use_custom_reward:
        return CustomRewardRLLibWrapper(env)
    return RLLibWrapper(env)


# =========================
# Curriculum helpers
# =========================

def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s