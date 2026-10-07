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
### STR-96 — Use actual yaw to calculate direction error

**Jira:** [STR-96](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-96?atlOrigin=eyJpIjoiMWM4NzQ4ZTM2YWU3NGU4YTk1ODhlYzEzYTMwZjE2NmEiLCJwIjoiaiJ9)  
**Previously:** Experiment 1

#### Hypothesis
Rewarding the agent for reducing the difference between its target heading and actual yaw should improve its ability to face the requested direction.

#### Change
Update the direction-error reward to use `get_yaw()` instead of the previous `get_wheel_yaw()` or `get_yaw_dot()` measures:

`direction_error = target_yaw - get_yaw()`

This makes the reward depend directly on heading error.

#### Evaluation and results
The training log reports major improvements. However, the evaluation was too easy to clearly establish whether heading tracking improved. A harder evaluation was introduced for subsequent experiments.

- [Evaluation results](../../evaluation/comparison_testing/test-01)
- [Training logs](control_agent/logs/ControlAgentTrainingEnv_PPO_2)

**Log numbering:** This experiment uses training log 2 because the first training run was a smoke test.

---
### STR-97 — Reduce the lean reward weight

**Jira:** [STR-97](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-97?atlOrigin=eyJpIjoiODM5OTkzNDM3OTA3NDYwMjkzZTMzNjU4MGU5OTI1NjMiLCJwIjoiaiJ9)  
**Previously:** Experiment 2

#### Hypothesis
Reducing the lean reward weight should make upright posture less dominant in the total reward, allowing target-speed and heading tracking to have greater influence on learning.

#### Change
Halve the scalar weight of the lean reward term from 10 to 5.

#### Evaluation and results
The experiment log reports major improvements following this change. The entry does not include quantitative metrics, so the linked evaluation results and training logs provide the supporting record.

- [Evaluation results](../../evaluation/comparison_testing/test-02)
- [Training logs](control_agent/logs/ControlAgentTrainingEnv_PPO_3)

**Log numbering:** This experiment uses training log 3 because the first training run was a smoke test.

---
### STR-98 — Remove the wheel-speed difference penalty

**Jira:** [STR-98](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-98?atlOrigin=eyJpIjoiYWU0YjIyMmExNTQzNGYwN2I2MzgyZWUwZmI3YWNhMTYiLCJwIjoiaiJ9)  
**Previously:** Experiment 3

#### Hypothesis
Removing the penalty for differences between left and right wheel speeds should allow the agent to turn more freely and improve target-heading tracking.

#### Change
Remove the `dyd` penalty, which uses the difference between left and right wheel velocities:

`dyd = l_wheel_v - r_wheel_v`

This term discouraged excessive spinning but was considered potentially redundant with the direction-error reward.

#### Evaluation and results
The experiment log reports major regressions after removing the penalty. This outcome does not support the proposed change and motivates testing restoration of the penalty in Experiment 5.

The existing entry does not quantify which performance metrics worsened.

- [Evaluation results](../../evaluation/comparison_testing/test-03)
- [Training logs](control_agent/logs/ControlAgentTrainingEnv_PPO_4)

**Log numbering:** This experiment uses training log 4 because the first training run was a smoke test.

---

### STR-99 — Include heading error in the agent’s observations

**Jira:** [STR-99](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-99?atlOrigin=eyJpIjoiODFiZTVkNjg0Yjk1NDE3MGEyMTQxMjlhZjE0OWVmNGYiLCJwIjoiaiJ9)  
**Previously:** Experiment 4

#### Hypothesis
Giving the agent a direct observation of its heading error should improve its ability to turn toward the target heading, compared with relying on the difference between left and right wheel speeds.

#### Change
Update the observations to include the difference between target yaw and actual yaw, wrapped to [-pi, pi):

`direction_error = (self.target_yaw - self.get_yaw() + np.pi) % (2 * np.pi) - np.pi`

Wrapping the angle represents the shortest signed heading error across the -pi/pi boundary.

#### Evaluation and results
The experiment log reports substantial improvements, with evaluation results showing improvement over the regressions recorded in Experiment 3.

The existing entry does not provide quantitative metrics or establish whether performance exceeded the best earlier experiment.

- [Evaluation results](../../evaluation/comparison_testing/test-04)
- [Training logs](control_agent/logs/ControlAgentTrainingEnv_PPO_5)

**Log numbering:** This experiment uses training log 5 because the first training run was a smoke test.

---
### STR-100 — Restore the wheel-speed difference penalty

**Jira:** [STR-100](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-100?atlOrigin=eyJpIjoiMjk1NTVkZGJiMzAyNDExZDg5YjQ3MjRhYTIyYjE1YTkiLCJwIjoiaiJ9)  
**Previously:** Experiment 5

#### Hypothesis
Restoring the penalty for unequal wheel speeds should discourage excessive spinning and encourage controlled target-heading tracking while maintaining balance.

Without this penalty, the agent may repeatedly spin through the target heading, briefly reducing direction error while continuing to collect reward for remaining upright.

#### Change
Restore the `dyd` penalty based on the difference between left and right wheel velocities:

`dyd = l_wheel_v - r_wheel_v`

This penalty was removed in STR-98 (formerly Experiment 3), which reported major regressions.

#### Evaluation and results
The original entry identifies excessive spinning as reward hacking and motivates restoring the penalty. It does not include evaluation results or training-log links for this experiment.

The effect of restoring the penalty should be assessed through spinning behavior, heading error, and balance performance.

---
### STR-101 — Change target yaw periodically during training

**Jira:** [STR-101](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-101?atlOrigin=eyJpIjoiZTlmOWRkZTg4NDYyNGFiNDgxN2JkNjFlZWEyNDNhMzgiLCJwIjoiaiJ9)  
**Previously:** Experiment 6  
**Status:** Planned

#### Hypothesis
Changing target yaw during an episode should improve the agent’s ability to respond to heading changes in evaluation, compared with training on one fixed target yaw per episode.

#### Proposed change
Introduce a scheduler that samples a new random target yaw at fixed time intervals within each training episode.

This exposes the agent to repeated heading changes, bringing training conditions closer to the evaluation environment.

#### Evaluation plan
Compare the updated agent against the preceding configuration using the same evaluation conditions.

Assess heading-tracking error and recovery after target changes, while checking that balance and target-speed tracking are maintained.

#### Results
Not yet performed according to the existing experiment log.

---
### STR-102 — Introduce two-phase curriculum learning

**Jira:** [STR-102](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-102?atlOrigin=eyJpIjoiMDIyZDE3OGViNTc4NGU0Y2I2NmViNmVmNjQ0OTJmYzQiLCJwIjoiaiJ9)  
**Previously:** Experiment 7  
**Status:** Planned

#### Hypothesis
Training basic balance and target-speed tracking before introducing changing target headings should help the agent learn heading tracking while retaining previously learned behavior.

#### Proposed change
Split training into two phases:

1. **Balance and target-speed tracking:** Hold target yaw at the robot’s initial heading throughout each episode. This removes target-heading changes, although heading error can still develop if the robot turns.
2. **Dynamic heading tracking:** Save the phase-one checkpoint and continue training with the periodic target-yaw scheduler introduced in STR-101.

A fixed target yaw alone does not disable the heading reward. Document any changes to reward weights between phases.

#### Evaluation plan
Compare the curriculum-trained agent with an agent trained using dynamic target yaw from the start, keeping the total training budget and evaluation conditions comparable.

Assess falls, target-speed error, heading error, and recovery after target-heading changes. Check whether phase two preserves balance and speed-tracking performance.

#### Results
Not yet performed according to the existing experiment log.

**Dependency:** STR-101 — periodic target-yaw changes during training.

---
### STR-103 — Use a cosine-based heading-error penalty

**Jira:** [STR-103](https://ucf-team-xx2ob1z2.atlassian.net/browse/STR-103)  
**Previously:** Experiment 8

#### Hypothesis
A smooth, periodic heading-error penalty may improve learning and target-heading tracking compared with an absolute-error penalty, particularly around the -pi/pi boundary.

#### Proposed change
Replace the heading-error reward term:

`-w * abs(direction_error)`

with:

`-w * (1 - cos(direction_error))`

Here, `w` is the reward weight and `direction_error` is measured in radians.

The cosine-based penalty is smooth across angle wrapping and applies a gentler penalty near zero error. This changes both the shape and scale of the reward: with the same weight, the maximum penalty magnitude changes from `w * pi` to `2 * w` for wrapped heading error.

#### Evaluation plan
Compare against the absolute-error penalty under the same training budget and evaluation conditions. Record the reward weight used in each configuration.

Assess heading error, response to target-heading changes, excessive spinning, falls, and target-speed tracking. Use these performance metrics rather than total reward alone, since the reward definition changes.

#### Results
The existing experiment entry does not include results or confirm completion.