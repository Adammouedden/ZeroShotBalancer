import numpy as np

from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv

"""
Env for training on moving backward and forward

ON ATTEMPT 5:
# Times (s) of each random turn. Offset from the speed changes (1, 3, 4.5, 5.5) so turns and speed changes
# are practised separately, and spread over the 30 s episode instead of ending at 5.5 s.
YAW_TURN_TIMES = (2.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0)

    def step(self, a):
        if self.data.time > 5.5:
            self.target_wheel_speed = 3.0 * self.delay_target_speed
        elif self.data.time > 4.5:
            self.target_wheel_speed = 2.0 * self.delay_target_speed
        elif self.data.time > 3.0:
            self.target_wheel_speed = -1.0 * self.delay_target_speed
        elif self.data.time > 1.0:
            self.target_wheel_speed = self.delay_target_speed

        # latest heading whose turn time has passed (start heading before the first turn)
        self.target_yaw = self.start_yaw
        for turn_time, heading in zip(YAW_TURN_TIMES, self.target_headings):
            if self.data.time > turn_time:
                self.target_yaw = heading

        ob, reward, terminated, truncated, info = ControlAgentBaseEnv.step(self, a)
        info["reward_terms"] = self.reward_terms

        return ob, reward, terminated, truncated, info

    def reset_model(self):
        self.target_wheel_speed = 0

        # between -10 and 10
        self.delay_target_speed = self.np_random.uniform(low=-10.0, high=10)
        # now between 10 to 20 or -20 to -10
        if self.delay_target_speed > 0:
            self.delay_target_speed += 10
        else:
            self.delay_target_speed -= 10

        # 2 degrees +/-
        self.pitch_offset = self.np_random.uniform(low=-0.0349066, high=0.0349066)
        ControlAgentBaseEnv.reset_model(self)

        # Headings are relative to the random direction the robot faces after reset, so "no turn" means no turn.
        # Each turn is a fresh random angle added to the previous heading, wrapped to [-pi, pi).
        self.start_yaw = self.get_yaw()
        turns = self.np_random.uniform(low=-np.pi, high=np.pi, size=len(YAW_TURN_TIMES))
        self.target_headings = (self.start_yaw + np.cumsum(turns) + np.pi) % (2 * np.pi) - np.pi
        self.target_yaw = self.start_yaw

        # the observation includes heading error, so rebuild it now that target_yaw is set
        return self._get_obs()

"""

DEFAULT_REWARD_WEIGHTS = dict(alive=0.6, pitch=0.05, speed_error=0.15, lean=5.0, direction_error=0.15)

class ControlAgentTrainingEnv(ControlAgentBaseEnv):
    def __init__(self, reward_weights=None, **kwargs):

        # Modularize reward weights and track reward terms separately
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.reward_terms = {}

        super().__init__('base_world_env.xml', **kwargs)

        self.delay_target_speed = 0.0
        self.delay_target_yaw = 0.0

        self.pitch_offset = 0.0

    def get_pitch(self) -> float:
        p = super().get_pitch()
        return p + self.pitch_offset

    def step(self, a):
        if self.data.time > 5.5:
            self.target_wheel_speed = 3.0 * self.delay_target_speed
        elif self.data.time > 4.5:
            self.target_wheel_speed = 2.0 * self.delay_target_speed
        elif self.data.time > 3.0:
            self.target_wheel_speed = -1.0 * self.delay_target_speed
        elif self.data.time > 1.0:
            self.target_wheel_speed = self.delay_target_speed


        ob, reward, terminated, truncated, info = ControlAgentBaseEnv.step(self, a)
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

        self.delay_target_yaw = self.np_random.uniform(low=-1*np.pi, high=np.pi)
        self.target_yaw = self.delay_target_yaw

        # 2 degrees +/-
        self.pitch_offset = self.np_random.uniform(low=-0.0349066, high=0.0349066)
        return ControlAgentBaseEnv.reset_model(self)


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

        # Differences in direction (Direction/Heading Error), modded to fit between [-pi, pi] radians
        direction_error = (self.target_yaw - self.get_yaw() + np.pi) % (2*np.pi) - np.pi

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
        } # future plans are to change direction_error to: -1 * weight * (1 - cos(direction_error)) for smooth training with a maximum at 0

        return sum(self.reward_terms.values())