import math
import numpy as np

from gymnasium.spaces import Box
from robot_environments.base_world_env import BaseWorldEnv

"""
Env for training on moving backward and forward
"""

DEFAULT_REWARD_WEIGHTS = dict(alive=0.6, pitch=0.05, speed_error=0.15, lean=5.0, direction_error=0.15)

class ControlAgentTrainingEnv(BaseWorldEnv):
    def __init__(self, reward_weights=None, **kwargs):

        # Modularize reward weights and track reward terms separately
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.reward_terms = {}

        BaseWorldEnv.__init__(self, **kwargs)

        # base observation space is 6 values with no heading signal at all - extend it
        # with sin/cos of current heading and sin/cos of heading error, so the policy
        # can actually condition its turning on where the target is (see _get_obs)
        base_low, base_high = self.observation_space.low, self.observation_space.high
        self.observation_space = Box(
            np.concatenate([base_low, [-1.0, -1.0, -1.0, -1.0]]),
            np.concatenate([base_high, [1.0, 1.0, 1.0, 1.0]]),
            dtype=np.float32,
        )

        self.delay_target_speed = 0.0
        self.delay_target_yaw = 0.0

        self.pitch_offset = 0.0

    def get_pitch(self) -> float:
        p = super().get_pitch()
        return p + self.pitch_offset

    def _direction_error(self) -> float:
        # signed shortest-path heading error, wrapped to (-pi, pi]
        return (self.target_yaw - self.get_yaw() + math.pi) % (2 * math.pi) - math.pi

    def _get_obs(self):
        obs = BaseWorldEnv._get_obs(self)
        direction_error = self._direction_error()
        heading_obs = np.array([
            math.sin(self.get_yaw()),
            math.cos(self.get_yaw()),
            math.sin(direction_error),
            math.cos(direction_error),
        ], dtype=np.float32)
        return np.concatenate([obs, heading_obs])

    def step(self, a):
        if self.data.time > 5.5:
            self.target_wheel_speed = 3.0 * self.delay_target_speed
        elif self.data.time > 4.5:
            self.target_wheel_speed = 2.0 * self.delay_target_speed
        elif self.data.time > 3.0:
            self.target_wheel_speed = -1.0 * self.delay_target_speed
        elif self.data.time > 1.0:
            self.target_wheel_speed = self.delay_target_speed


        ob, reward, terminated, truncated, info = BaseWorldEnv.step(self, a)
        info["reward_terms"] = self.reward_terms

        return ob, reward, terminated, truncated, info

    def reset_model(self):
        self.target_wheel_speed = 0
        self.target_yaw = 0

        # between -10 and 10
        self.delay_target_speed = self.np_random.uniform(low=-10.0, high=10)
        # now between 10 to 20 or -20 to -10
        if self.delay_target_speed > 0:
            self.delay_target_speed += 10
        else:
            self.delay_target_speed -= 10

        self.delay_target_yaw = self.np_random.uniform(low=0.0, high=2.0* np.pi)
        self.target_yaw = self.delay_target_yaw

        # 2 degrees +/-
        self.pitch_offset = self.np_random.uniform(low=-0.0349066, high=0.0349066)
        return BaseWorldEnv.reset_model(self)

    def _get_reward(self):
        pitch = self.get_pitch()
        wheel_speed = self.get_wheel_speed()

        # Differences in wheel speed
        dv = self.target_wheel_speed - wheel_speed
        MAX_DV = 40.0
        max_dv = np.clip(dv, -MAX_DV, MAX_DV)
        # will be -1 to 1
        dv_n = max_dv / MAX_DV
        dv_s = abs(dv_n)

        # Differences in direction (Direction/Heading Error)
        direction_error = self._direction_error()

        # Reward Shaping: What direction should the agent lean in to remain upright?
        lean_dir = 0.0
        if self.target_wheel_speed > 0 and self.target_wheel_speed > wheel_speed:
            # then reward for leaning forward, needs to speed up forward
            lean_dir = -1.0
        elif self.target_wheel_speed < 0 and self.target_wheel_speed < wheel_speed:
            # then reward for leaning backwards, needs to speed up backwards
            lean_dir = 1.0
        elif self.target_wheel_speed > 0 and self.target_wheel_speed < wheel_speed:
            # then reward for leaning backward, needs to slow down going forward
            lean_dir = 1.0
        elif self.target_wheel_speed < 0 and self.target_wheel_speed > wheel_speed:
            # then reward for leaning backwards, needs to speed up backwards
            lean_dir = -1.0


        self.reward_terms = {
            "alive": self.reward_weights["alive"],
            "pitch": -1 * self.reward_weights["pitch"] * abs(pitch),
            "speed_error": -1 * self.reward_weights["speed_error"] * dv_s,
            "lean": lean_dir * pitch * self.reward_weights["lean"] * dv_s,
            "direction_error": -1 * self.reward_weights["direction_error"] * abs(direction_error)
        }

        return sum(self.reward_terms.values())