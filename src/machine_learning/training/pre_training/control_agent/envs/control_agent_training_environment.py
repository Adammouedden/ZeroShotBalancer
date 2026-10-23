import numpy as np

from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv

# STR-75: reward weights pulled out into a dict so each term can be tuned and logged per
# run. RewardTermsCallback writes these to reward_weights.json beside the run's tensorboard
# events, which is what identifies the experiment a historical training log belongs to.
DEFAULT_REWARD_WEIGHTS = dict(
    alive=0.6,
    pitch=0.05, 
    speed_error=0.15, 
    lean=5.0,               # STR-97: halved from 10 so speed/heading tracking matter more
    direction_error=0.15, 
    uneven_wheel_speed=0.007  # STR-98 removed this; restored afterwards
)

# Seeded so target speeds, yaws and pitch offsets repeat across runs, making two
# experiments comparable. Not tied to a ticket; added in commit 557e536.
SEED = 42

class ControlAgentTrainingEnv(ControlAgentBaseEnv):
    def __init__(self, reward_weights=None, **kwargs):

        # STR-75: modularize reward weights and track reward terms separately, so Tensorboard
        # shows each component instead of only the summed scalar
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.reward_terms = {}

        super().__init__('base_world_env.xml', **kwargs)

        self.delay_target_speed = 0.0
        self.delay_target_yaw = 0.0

        self.pitch_offset = 0.0

        self.rng = np.random.default_rng(SEED)

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
        # STR-75: RewardTermsCallback reads this to log each reward term separately
        info["reward_terms"] = self.reward_terms

        return ob, reward, terminated, truncated, info

    def reset_model(self):
        self.target_wheel_speed = 0
        self.target_yaw = 0

        # between -10 and 10
        self.delay_target_speed = self.rng.uniform(low=-10.0, high=10)
        # now between 10 to 20 or -20 to -10
        if self.delay_target_speed > 0:
            self.delay_target_speed += 10
        else:
            self.delay_target_speed -= 10

        # One random target heading per episode.
        self.delay_target_yaw = self.rng.uniform(low=-1*np.pi, high=np.pi)
        self.target_yaw = self.delay_target_yaw

        # 2 degrees +/-
        self.pitch_offset = self.rng.uniform(low=-0.0349066, high=0.0349066)
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
        # STR-96: uses get_yaw() (actual heading), not the earlier get_wheel_yaw()/get_yaw_dot()
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

        # Punish non-smooth turning
        # Restored after STR-98 removed it; without it the agent spins through the target
        # heading to farm the alive reward while briefly zeroing direction error
        uneven_wheel_speed = abs(self.target_yaw - self.get_wheel_yaw())
        
        # TODO: Need to normalize per- reward term. Requires running std dev and mean
        
        self.reward_terms = {
            "alive": self.reward_weights["alive"],
            "pitch": -1 * self.reward_weights["pitch"] * abs(pitch),
            "speed_error": -1 * self.reward_weights["speed_error"] * dv_s,
            "lean": lean_dir * pitch * self.reward_weights["lean"] * dv_s,
            "direction_error": -1 * self.reward_weights["direction_error"] * abs(direction_error),
            "uneven_wheel_speed": -1 * self.reward_weights["uneven_wheel_speed"] * uneven_wheel_speed
        }
        # STR-103 (not implemented): change direction_error to
        # -1 * weight * (1 - cos(direction_error)) for a smooth penalty with a maximum at 0.
        # Note this also rescales the term - the largest penalty goes from weight*pi to
        # 2*weight - so the weight needs revisiting alongside it.

        

        return sum(self.reward_terms.values())