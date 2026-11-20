import numpy as np
from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv


DEFAULT_REWARD_WEIGHTS = dict(
    alive=0.6,
    pitch=0.05,
    speed_error=0.15,
    direction_error=0.15,
)
#Motor limits?
MAX_TARGET_SPEED = 20.6
SEED = 42


class MinimalTrainingEnv(ControlAgentBaseEnv):

    def __init__(self, reward_weights=None, xml_file='base_world_calibrated.xml', **kwargs):
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.reward_terms = {}
        
        super().__init__(xml_file, **kwargs)
        
        self.delay_target_speed = 0.0
        self.delay_target_yaw = 0.0
        
        self.rng = np.random.default_rng(SEED)

    def reset_model(self):
        self.target_wheel_speed=0.0
        #Without our max_target speed of the motors and with checked both
        #self.delay_target_speed = self.rng.uniform(low=-10.0, high=10)
        self.delay_target_speed = self.rng.uniform(low=-MAX_TARGET_SPEED/3, high=MAX_TARGET_SPEED/3)

        self.delay_target_yaw = self.rng.uniform(low=-1*np.pi, high=np.pi)
        self.target_yaw = self.delay_target_yaw

        return super().reset_model()

    def step(self, a):
        
        if self.data.time > 5.5:
            self.target_wheel_speed = 3.0 * self.delay_target_speed
        elif self.data.time > 4.5:
            self.target_wheel_speed = 2.0 * self.delay_target_speed
        elif self.data.time > 3.0:
            self.target_wheel_speed = -1.0 * self.delay_target_speed
        elif self.data.time > 1.0:
            self.target_wheel_speed = self.delay_target_speed
        
        
        ob, reward, terminated, truncated, info = super().step(a)
        # STR-75: RewardTermsCallback reads this to log each reward term separately
        info["reward_terms"] = self.reward_terms
        
        return ob, reward, terminated, truncated, info

    def _get_reward(self):
        pitch = self.get_pitch()
        wheel_speed=self.get_wheel_speed()
        dv=self.target_wheel_speed - wheel_speed
        MAX_DV=40.0
        max_dv = np.clip(dv, -MAX_DV, MAX_DV)
        dv_n = max_dv / MAX_DV
        dv_s = abs(dv_n)

        direction_error = (self.target_yaw - self.get_yaw() + np.pi) % (2*np.pi) - np.pi
        self.reward_terms = {
            "alive": self.reward_weights["alive"],
            "pitch": -1 * self.reward_weights["pitch"] * abs(pitch),
            "speed_error": -1 * self.reward_weights["speed_error"]*dv_s,
            "direction_error": -1 * self.reward_weights["direction_error"] * abs(direction_error),
        }
        return sum(self.reward_terms.values())
