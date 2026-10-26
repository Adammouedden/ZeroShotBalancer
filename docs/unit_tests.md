# Unit Tests
Unit tests are needed to test the input data of our RL training pipeline. The state data (see below) must be validated that they are coming from the correct source in MuJoCo and that they are normalized properly.

Additionally, both actions (left/right wheel speed) must be unit-tested.

The goal of these tests is to ensure no undefined behavior from our I/O sources prior to a full training run/eval.

## Quickstart
Run the tests using `uv run pytest -v tests/robot_environments/test_control_agent_base_env.py` from `src/machine_learning`.

## Observations
There are 6 raw observations assembled in [ControlAgentBaseEnv](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L260):

1. Normalized pitch
1. Normalized pitch rate
1. Left wheel speed _(observation)_
1. Right wheel speed _(observation)_
1. Difference between target and actual wheel speed
1. Difference between target and actual yaw

## Actions
Each action has 2 continuous components in `[-1, 1]`, defined by the [action space](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L82) and converted into commands in [ControlAgentBaseEnv.step()](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L110).

1. Left wheel velocity adjustment, `action[0]`
1. Right wheel velocity adjustment, `action[1]`

We will follow the standard Arrange, Act, Assert (AAA) lifecycle.

## Arrange

### Pitch
The System Under Test (SUT) here is the pitch of the self-balancing agent. It is algorithmically defined as the `get_pitch()` function found [here](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L201).

`get_pitch` takes the quaternions from `robot_body`, where the rotation order `[w, x, y, z]` is mapped to `[quat[0], quat[1], quat[2], quat[3]]`. Using these, we can create a `Rotation` object and then calculate the Euler angles. The function that does this wraps the first of these angles (`x`) between $[-180°, 180°]$, and outputs it in radians.

In our case, we are only interested in extracting the `x` component of the pitch as our returned value (`return angles[0]`).

The conditions under which we will test will range from simple calculations to more complex. Inputs will be known quaternions.

#### Context
For a rotation by angle $\theta$ around a unit axis $(a_x, a_y, a_z)$, the quaternion is:

$$[w, x, y, z] = \left[\cos\left(\frac{\theta}{2}\right), a_x \sin\left(\frac{\theta}{2}\right), a_y \sin\left(\frac{\theta}{2}\right), a_z \sin\left(\frac{\theta}{2}\right)\right]$$

A unit axis is simply a direction vector with length 1.

To further clarify the above with an example: the X-axis direction is $(1, 0, 0)$. The half-angle is $90 / 2 = 45$, so:

- `w` is $\cos(45\degree) \approx 0.7071$
- `x` is $1\sin(45°) \approx 0.7071$
- `y` and `z` are zero because the axis has no Y or Z component

This produces the second test vector below, $+90°$  X-axis rotation.

#### Standard Test Cases

| Test case | Input quaternion | Expected pitch (radians) | Degree equivalent |
| --- | --- | --- | --- |
| Identity (no rotation) | `[1, 0, 0, 0]` | `0` | 0° |
| +90° around X | `[0.7071, 0.7071, 0, 0]` | `π / 2` | +90° |
| −90° around X | `[0.7071, -0.7071, 0, 0]` | `-π / 2` | −90° |
| +30° around Y | `[0.9659, 0, 0.2588, 0]` | `0` | 0° |
| −30° around Y | `[0.9659, 0, -0.2588, 0]` | `0` | 0° |
| +90° around Z | `[0.7071, 0, 0, 0.7071]` | `0` | 0° |
| −90° around Z | `[0.7071, 0, 0, -0.7071]` | `0` | 0° |

The Y and Z cases verify that rotations around other axes do not
incorrectly register as pitch in these configurations.

The standard Y cases deliberately avoid ±90°, where the `xyz` Euler representation encounters gimbal lock. This is when the same orientation can be represented with nonzero X and Z Euler angles.

#### Additional cases

| Test case | Input quaternion | Expected pitch (radians) | Degree equivalent |
| --- | --- | --- | --- |
| $+30°$ around X | `[0.9659, 0.2588, 0, 0]` | `π / 6` | $+30°$ |
| $+180°$ around X | `[0, 1, 0, 0]` | Either `π` or `-π` | $±180°$ |
| +180° around X (calculated quaternion) | `[np.cos(np.pi / 2), np.sin(np.pi / 2), 0, 0]` | Either `π` or `-π` | ±180° |
| Negated $−90°$ Z quaternion | `[-0.7071, 0, 0, 0.7071]` | `0` | $0°$ |
| $+185°$ around X | `[-0.0436, 0.9990, 0, 0]` | `-35π / 36` | $−175°$ |
| $+175°$ around X | `[0.0436, 0.9990, 0, 0]` | `35π / 36` | $+175°$ |

- **180° case:** A zero scalar component is valid.
- **Quaternion negation:** `q` and `-q` represent exactly the same orientation. Compare this result with the original −90° Z case.
- **Angle wrapping:** +185° and −175° describe the same X rotation. The returned angle wraps into the supported range.
- **Boundary equivalence:** At exactly 180°, both endpoints are valid because +π and −π represent the same X rotation.

### Pitch Rate
Pitch rate is defined as `pitch_dot` within the `ControlAgentBaseEnv`, and algorithmically defined by `get_pitch_dot_alt()`, found [here](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L216). An alternative `get_pitch_dot()` function exists which is not currently being used.

First, the pitch is retrieved using `get_pitch()`, alongside the current timestamp. Using the last time stamp if available, `pitch_dot` is then calculated by subtracting the current `pitch` and the `last_pitch` then dividing by their time delta `dt`.

#### Test Cases
| Test case | Previous state (`last_pitch`, `last_time`) | Current (`pitch`, `time`) | Expected pitch rate (rad/s) |
| --- | --- | --- | --- |
| First reading: no history | `(None, None)` | `(0.2, 1.0)` | `0` |
| Missing previous time | `(0.1, None)` | `(0.2, 1.0)` | `0` |
| Missing previous pitch | `(None, 0.5)` | `(0.2, 1.0)` | `0` |
| Unchanged pitch | `(0.2, 1.0)` | `(0.2, 1.1)` | `0` |
| Increasing pitch | `(0.1, 1.0)` | `(0.2, 1.1)` | Approximately `1.0` |
| Decreasing pitch | `(0.2, 1.0)` | `(0.1, 1.1)` | Approximately `-1.0` |
| Crossing zero | `(-0.1, 1.0)` | `(0.1, 1.2)` | Approximately `1.0` |
| Longer time interval | `(0.1, 1.0)` | `(0.2, 1.2)` | Approximately `0.5` |
| Same timestamp | `(0.1, 1.0)` | `(0.2, 1.0)` | `0` |
| Time moves backward | `(0.1, 1.0)` | `(0.2, 0.0)` | `0` |
| Consecutive readings | Initially `(0.0, 1.0)` | First `(0.1, 1.1)`, then `(0.3, 1.2)` | Approximately `1.0`, then `2.0` |
| Angle wrapping diagnostic | `(np.deg2rad(179), 1.0)` | `(np.deg2rad(-179), 1.1)` | Current implementation: approximately `-62.482787`; continuous positive crossing: approximately `+0.349066` |
| Observation scaling | Controlled raw pitch rate of `0.5` | Call `_get_obs()` with valid history and positive `dt` | `_get_obs()[1]` approximately `0.5 / PITCH_DOT_MAX`, currently `0.5` |

### Left/Right Wheel Speed Observation
Both wheel speeds are unpacked from the `get_wheel_velocities()` function, found [here](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L233). It reads each wheel's joint angular velocity from MuJoCo: `torso_{w}_wheel.qvel[0]` (where $w \in {l,r}$). It returns these raw values in `(left, right)` order, in radians per second, without changing their signs. 

Both wheels spin "forward", but one is spinning in a negative direction as it's rotated 180 degrees from the other. Our [robot XML](/src/machine_learning/training/robot_environments/robot-02.xml) defines left wheel axis as $(-1, 0, 0)$ and right wheel axis as $(+1, 0, 0)$. These axes point in opposite direction, so reversing the sign describes the same physical rotation. Each velocity is scaled as:
$$v_scaled = \frac{v_raw}{170.0} \times 4$$
Thus, a raw velocity of $170$ becomes $4$, not $1$.

#### Test Cases
Each case outlined below will be run twice, once for the left wheel and once for the right. The other wheel's velocity will be set to zero. Verifying that left → tuple `[0]`, observation `[2]`; right → tuple `[1]`, observation `[3]`.
| Test case | Selected wheel’s joint velocity (rad/s) | Expected raw wheel velocity | Expected scaled observation |
| --- | --- | --- | --- |
| Stationary wheel | `0` | `0` | `0` |
| Positive velocity | `42.5` | `42.5` | `1` |
| Negative velocity | `-42.5` | `-42.5` | `-1` |
| Positive scaling reference | `170` | `170` | `4` |
| Negative scaling reference | `-170` | `-170` | `-4` |
| Above scaling reference | `212.5` | `212.5` | `5`; no clipping |
| Below negative scaling reference | `-212.5` | `-212.5` | `-5`; no clipping |

We will also use distinct simultaneous velocities to verify that the sources are independent and returned in the correct order.

| Test case | Joint velocities `(left, right)` (rad/s) | Expected raw tuple | Expected observations `[2:4]` |
| --- | --- | --- | --- |
| Distinct wheel velocities | `(42.5, -85)` | `(42.5, -85)` | `[1, -2]` |
| Swapped wheel velocities | `(-85, 42.5)` | `(-85, 42.5)` | `[-2, 1]` |

### Context for Target Variables below
`target_wheel_speed` and `target_yaw` describe how fast we want the robot to move and which direction we want it to face. They are set during training or evaluation, then compared with the robot’s measured movement. For these tests, we set the targets ourselves so we can check the differences against known values.

### Target and Actual Wheel Speed Difference

The delta of wheel speed is computed by subtracting the `target_wheel_speed` variable and the results of `get_wheel_speed()`, defined [here](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L241). This function takes both wheel velocities, negates the right one, adds them both, and then divides by 2.
$$ \text{wheel\_speed} = \frac{vel_l + (-1 vel_r)}{2}$$
$$ \frac{\text{target} - \text{wheel\_speed}}{42.5}$$

#### Test Cases
| Test case | Raw wheel velocities `(left, right)` (rad/s) | Target speed (rad/s) | Expected `get_wheel_speed()` (rad/s) | Expected scaled difference `(target − actual) / 42.5` |
| --- | --- | --- | --- | --- |
| Stationary, zero target | `(0, 0)` | `0` | `0` | `0` |
| Stationary, positive target | `(0, 0)` | `42.5` | `0` | `1` |
| Stationary, negative target | `(0, 0)` | `-42.5` | `0` | `-1` |
| Matching positive target | `(42.5, -42.5)` | `42.5` | `42.5` | `0` |
| Matching negative target | `(-42.5, 42.5)` | `-42.5` | `-42.5` | `0` |
| Actual speed below target | `(42.5, -42.5)` | `85` | `42.5` | `1` |
| Actual speed above target | `(85, -85)` | `42.5` | `85` | `-1` |
| Moving opposite to target | `(-42.5, 42.5)` | `42.5` | `-42.5` | `2` |
| Left wheel only | `(85, 0)` | `0` | `42.5` | `-1` |
| Right wheel only | `(0, -85)` | `0` | `42.5` | `-1` |
| Unequal wheel speeds | `(85, -42.5)` | `0` | `63.75` | `-1.5` |
| Equal raw velocities cancel | `(42.5, 42.5)` | `0` | `0` | `0` |
| Large positive difference; no clipping | `(0, 0)` | `170` | `0` | `4` |
| Large negative difference; no clipping | `(170, -170)` | `0` | `170` | `-4` |

### Target and Actual Yaw Difference
The delta of the heading/yaw difference is defined as follows:
$$(\text{target\_yaw} - \text{yaw} + \pi)\bmod (2\pi) - \pi$$
This formula wraps the heading into $[-\pi, \pi)$, in radians, where `target_yaw` is a set variable and `get_yaw()` is defined [here](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L246). Just like in `get_pitch()`, this function extracts the quaternion from `robot_body`, calculates the rotation, converts the angles to Euler angles and uses only the third angle.

#### Test Cases
These focus on the yaw difference.

| Test case | Target yaw (rad) | Actual yaw (rad) | Expected wrapped difference (rad) |
| --- | --- | --- | --- |
| Both zero | `0` | `0` | `0` |
| Matching nonzero headings | `π / 3` | `π / 3` | Approximately `0` |
| Positive difference | `π / 2` | `0` | Approximately `π / 2` |
| Negative difference | `0` | `π / 2` | Approximately `-π / 2` |
| Headings on opposite sides of zero | `π / 6` | `-π / 6` | Approximately `π / 3` |
| Positive crossing of ±180° boundary | `np.deg2rad(-179)` | `np.deg2rad(179)` | Approximately `np.deg2rad(2)` |
| Negative crossing of ±180° boundary | `np.deg2rad(179)` | `np.deg2rad(-179)` | Approximately `np.deg2rad(-2)` |
| Positive half-turn | `π` | `0` | Approximately `-π` |
| Negative half-turn | `-π` | `0` | Approximately `-π` |
| Equivalent ±180° headings | `π` | `-π` | Approximately `0` |
| Target includes a full revolution | `2π + π / 4` | `0` | Approximately `π / 4` |
| Actual includes a full revolution | `0` | `2π + π / 4` | Approximately `-π / 4` |

#### Additional Test Cases
These focuse on the quaternion extraction.

| Test case | Input quaternion `[w, x, y, z]` | Expected `get_yaw()` (rad) | Target yaw (rad) | Expected wrapped difference (rad) |
| --- | --- | --- | --- | --- |
| Identity | `[1, 0, 0, 0]` | `0` | `0` | `0` |
| +90° around Z | `[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]` | Approximately `π / 2` | `0` | Approximately `-π / 2` |
| −90° around Z | `[np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]` | Approximately `-π / 2` | `0` | Approximately `π / 2` |
| +30° around X | `[np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]` | Approximately `0` | `π / 4` | Approximately `π / 4` |

### Left/Right Wheel Speed Action
[`ControlAgentBaseEnv.step(self, a)`](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L110) receives the action as a two-element array in `(left, right)` order. During testing, [`model.predict(obs)`](/src/machine_learning/training/pre_training/control_agent/sb_rl.py#L168) produces the action and the test loop passes it to [`env.step(action)`](/src/machine_learning/training/pre_training/control_agent/sb_rl.py#L175). During training, [`model.learn()`](/src/machine_learning/training/pre_training/control_agent/sb_rl.py#L575) runs this loop internally, where Stable-Baselines3 samples policy actions, clips them to the action-space bounds for the current PPO configuration, and calls the environment's `step()`. [`ControlAgentTrainingEnv.step()`](/src/machine_learning/training/pre_training/control_agent/envs/control_agent_training_environment.py#L92) updates the training targets, then [forwards the action to the base class](/src/machine_learning/training/pre_training/control_agent/envs/control_agent_training_environment.py#L103).

The base class reads the current joint angular velocities and [calculates the commands](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L113):

```python
command_left = current_left_velocity + action[0] * WHEEL_SPEED_DELTA_MAX
command_right = current_right_velocity + action[1] * WHEEL_SPEED_DELTA_MAX
```

[`WHEEL_SPEED_DELTA_MAX`](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L23) is currently `4.0` rad/s per unit action. An action of zero commands the wheel's current measured velocity. An action of `+1` adds `4.0` rad/s, and `-1` subtracts `4.0` rad/s. Each wheel uses its own joint sign convention; the right-wheel action is not negated when assigning its command.

The commands are [written to the velocity actuators](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L116):

```python
env.data.actuator("motor_l_wheel").ctrl[0]
env.data.actuator("motor_r_wheel").ctrl[0]
```

The unit tests arrange known initial joint velocities, call `step(action)`, and assert the two `ctrl` values. 

For clarity, commanded velocity (`ctrl`) and measured velocity (`qvel`) are different quantities. These tests verify the command written before [physics advances](/src/machine_learning/training/robot_environments/ControlAgentBaseEnv.py#L118). They do not require the measured velocity to reach the command immediately.

#### Test Cases

Each row verifies both actuator commands. Velocities and commands are in rad/s.

| Test case | Initial joint velocities `(left, right)` | Action `(left, right)` | Expected actuator commands `(left, right)` |
| --- | --- | --- | --- |
| Stationary, zero action | `(0, 0)` | `(0, 0)` | `(0, 0)` |
| Moving, zero action | `(10, -10)` | `(0, 0)` | `(10, -10)` |
| Positive left adjustment only | `(10, -10)` | `(1, 0)` | `(14, -10)` |
| Negative left adjustment only | `(10, -10)` | `(-1, 0)` | `(6, -10)` |
| Positive right adjustment only | `(10, -10)` | `(0, 1)` | `(10, -6)` |
| Negative right adjustment only | `(10, -10)` | `(0, -1)` | `(10, -14)` |
| Both positive adjustments | `(10, -10)` | `(1, 1)` | `(14, -6)` |
| Both negative adjustments | `(10, -10)` | `(-1, -1)` | `(6, -14)` |
| Opposite adjustments, left positive | `(10, -10)` | `(1, -1)` | `(14, -14)` |
| Opposite adjustments, right positive | `(10, -10)` | `(-1, 1)` | `(6, -6)` |
| Fractional adjustments | `(10, -10)` | `(0.5, -0.5)` | `(12, -12)` |
| Distinct fractional adjustments | `(10, -10)` | `(-0.25, 0.75)` | `(9, -7)` |
| Unequal initial velocities | `(7, -13)` | `(0.25, -0.5)` | `(8, -15)` |
| Reversed initial velocities | `(-10, 10)` | `(0.5, -0.5)` | `(-8, 8)` |
| Start from rest with opposite commands | `(0, 0)` | `(1, -1)` | `(4, -4)` |
| Cross zero on both wheels | `(-1, 1)` | `(0.5, -0.5)` | `(1, -1)` |
| Reach zero on both wheels | `(4, -4)` | `(-1, 1)` | `(0, 0)` |

#### Consecutive Calls

Each command must use the joint's current measured velocity. With physics disabled, writing `ctrl` does not change `qvel`, so repeating an action must not accumulate adjustments from the previous command.

| Test case | Joint velocities before call | Previous actuator commands | Action | Expected actuator commands |
| --- | --- | --- | --- | --- |
| First call | `(10, -10)` | `(0, 0)` | `(1, -1)` | `(14, -14)` |
| Repeat with unchanged measurements | `(10, -10)` | `(14, -14)` | `(1, -1)` | `(14, -14)` |
| Repeat after updating joint measurements | `(12, -9)` | `(14, -14)` | `(1, -1)` | `(16, -13)` |

These rows will be run in sequence within one test, updating the joint velocities explicitly before the third call.

#### Actuator Command Boundaries

The [left](/src/machine_learning/training/robot_environments/robot-02.xml#L22) and [right](/src/machine_learning/training/robot_environments/robot-02.xml#L23) velocity actuators declare `ctrllimited="true"` and `ctrlrange="-78.54 78.54"`. The Python `step()` writes the computed commands directly, without clipping them to those limits.

For these command-assignment tests, physics is disabled so the expected values refer to the requests written to ctrl. Actuator saturation and wheel response require separate simulation tests.

| Test case | Initial joint velocities `(left, right)` | Action | Expected stored `ctrl` values |
| --- | --- | --- | --- |
| Reach positive and negative limits | `(74.54, -74.54)` | `(1, -1)` | `(78.54, -78.54)` |
| Exceed positive and negative limits | `(78, -78)` | `(1, -1)` | `(82, -82)` |
| Exceed limits with reversed wheel signs | `(-78, 78)` | `(-1, 1)` | `(-82, 82)` |

## Act

The file [`test_control_agent_base_env.py`](/src/machine_learning/tests/robot_environments/test_control_agent_base_env.py) contains the observation and action tests outlined above. Run the tests using `uv run pytest -v tests/robot_environments/test_control_agent_base_env.py` from `src/machine_learning`.
