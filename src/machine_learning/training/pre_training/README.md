# Pre-Training

This directory contains all code pertaining to the initial training phase of the Zero-Shot Balancer agent

## Experiment Log:

The purpose of this log is so that all team members who wish to work on training can build off of past work in a documented manner. 

The objective of this work is twofold:
1. Surpass the reference-baseline (already achieved)
2. Perform well on the [hard evaluation](../../evaluation/mixed_schedule_hard.json)

    - As of right now, "perform well" means 0 falls and minimal direction_error, as the agent is good at staying upright and good at matching target speed

[Get Started](control_agent/README.md)

---
### Experiment 1:
---

#### If the agent is rewarded for minimizing differences in target yaw and actual yaw, then its weights will be updated such that it will be more likely to take actions that will cause it to face the target yaw

New reward function that leverages get_yaw() to compute direction_error, instead of previously get_wheel_yaw() or get_yaw_dot()

direction_error = target_yaw - get_yaw()

(Results on this experiment are not obvious because the evaluation was too easy, it was upgraded for subsequent experiments)

[Results](../../evaluation/comparison_testing/test-01)


[Major improvements](control_agent/logs/ControlAgentTrainingEnv_PPO_2)

Logs are off by 1 because the first one was a smoke test

---
### Experiment 2:
---
### 

#### If the lean reward term is lowered such that it does not dominate the other terms, then maximizing total reward will not be so synonymous with maximizing the lean reward term

Scalar reward weight for lean halved from 10 to 5

[Results](../../evaluation/comparison_testing/test-02)

[Major improvements](control_agent/logs/ControlAgentTrainingEnv_PPO_3)

---
### Experiment 3:
---

#### If the agent is no longer rewarded for minimizing differences in left and right wheel speeds, then it will have an easier time turning and matching target yaw

Removing dyd, differences in left_wheel_velocity and right_wheel_velocity (l_wheel_v - r_wheel_v) which was used as a penalty for excessive spinning, thought to be redundant

[Results](../../evaluation/comparison_testing/test-03)


[Major regressions](control_agent/logs/ControlAgentTrainingEnv_PPO_4)

---
### Experiment 4:
---

#### If the agent can actually observe a measurement of its yaw (rather than just l_wheel_speed - r_wheel_speed), then it will be better capable of minimizing direction error (target yaw - actual yaw)

Updated observations to now receive direction_error as input, (self.target_yaw - self.get_yaw() + np.pi) % (2*np.pi) - np.pi

[Results so far](../../evaluation/comparison_testing/test-04)

[Logs](control_agent/logs/ControlAgentTrainingEnv_PPO_5)



---
### Experiment 5:
---

#### If the agent receives a penalty for excessive turning, then it can no longer perform Reward Hacking by constantly spinning to eventually minimize direction_error briefly while receiving great reward for still remaining upright

Restoring dyd, differences in left_wheel_velocity and right_wheel_velocity (l_wheel_v - r_wheel_v) which was used as a penalty for excessive spinning, empirically proven to NOT be redundant

---
### Experiment 6:
---

#### If the agent is trained on an environment that target yaw at a set time interval (rather than once per episode), then it will generalize better to the evaluation environment that contains swapping target yaw

Altering training code to contain random target_yaw swapping at set time intervals, making the training env more identically distributed to the evaluation env

---
### Experiment 7:
---

#### If the agent receives a smoother reward term for direction error that handles values at -pi and pi cleaner, then stable weight updates will lead to improved reward maximization

Alter the reward term for direction_error from -w * abs(direction_error) to -w * (1 - cos(direction_error))

---