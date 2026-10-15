# Unit Tests
Unit tests are needed to test the input data of our RL training pipeline. The state data (see below) must be validated that they are coming from the correct source in MuJoCo and that they are normalized properly.

Additionally, both actions (left/right wheel speed) must be unit-tested.

The goal of these tests is to ensure no undefined behavior from our I/O sources prior to a full training run/eval.

## Observations
There are 6 raw observations assembled in [ControlAgentBaseEnv](../../robot_environments/ControlAgentBaseEnv.py#L261):

1. Normalized pitch
1. Normalized pitch rate
1. Left wheel speed
1. Right wheel speed
1. Difference between target and actual wheel speed
1. Difference between target and actual yaw

We will follow the standard Arrange, Act, Assert (AAA) lifecycle.

## Arrange

### Pitch
The System Under Test (SUT) here is the pitch of the self-balancing agent. It is algorithimically defined as the `get_pitch()` function found [here](../../robot_environments/ControlAgentBaseEnv.py#L201).

`get_pitch` takes the quaternions from `robot_body`, where the rotation order `[w, x, y, z]` is mapped to `[quat[0], quat[1], quat[2], quat[3]]`. Using these, we can create a `Rotation` object and then calculate the Euler angles. The function that does this wraps the first of these angles (`x`) between $[-180°, 180°]$, and outputs it in radians.

In our case, we are only interested in extracting the `x` component of the pitch as our returned value (`return angles[0]`).

The conditions under which we will test will range from simple calculations to more complex. Inputs will be known quaternions.

#### Context
For a rotation by angle $\theta$ around a unit axis $(a_x, a_y, a_z)$, the quaternion is:

$$[w, x, y, z] = \left[\cos\left(\frac{\theta}{2}\right), a_x \sin\left(\frac{\theta}{2}\right), a_y \sin\left(\frac{\theta}{2}\right), a_z \sin\left(\frac{\theta}{2}\right)\right]$$

A unit axis is simply a direction vector with length 1.

To further clarify the above with an example: the X-axis direction is $(1, 0, 0)$. The half-angle is $90 / 2 = 45$, so:

- `w` is $\cos(45\degree) = 0.7071$
- `x` is $1\sin(45°) = 0.7071$
- `y` and `z` are zero because the axis has no Y or Z component

This produces the second test vector below, $+90°$  X-axis rotation.

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

### Left Wheel Speed

### Right Wheel Speed

### Target and Actual Wheel Speed Difference

### Target and Actual Yaw Difference

## Act

The file [`test_control_agent_base_env.py`](../../../tests/robot_environments/test_control_agent_base_env.py) contains all the unit tests illustrated above. Run it using `uv run pytest -v tests/robot_environments/test_control_agent_base_env.py` from `src/machine_learning`.