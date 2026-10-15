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

#### Standard test vectors
- Identity (no rotation): `[1, 0, 0, 0]`.\
*In this case, the expected first angle is 0° .*
- $+90°$  X-axis rotation: `[0.7071, 0.7071, 0, 0]`.\
*Expected angle: 90° pitch.*
- $-90°$ X-axis: `[0.7071, -0.7071, 0, 0]`.\
*Expected: -90° pitch.*
- $+90°$ Y-axis rotation: `[0.7071, 0, 0.7071, 0]`.\
*Expected: 0° pitch (only pertains to `x`).*
- $-90°$ Y-axis rotation: `[0.7071, 0, -0.7071, 0]`.\
*Expected: 0° pitch.*
- $+90°$ Z-axis rotation: `[0.7071, 0, 0, 0.7071]`.
- $-90°$ Z-axis rotation: `[0.7071, 0, 0, -0.7071]`.

#### Other manageable cases
- $+30°$ X-axis rotation: `[0.9659, 0.2588, 0, 0]`
- $+180°$ Y-axis rotation:  `[0, 1, 0, 0]`
- Negation of $-90°$ Z-axis quaternion components: `[-0.7071, 0, 0, 0.7071]`.\
*Expected to be the same orientation, just the other way.*
- $+185°$ X-axis rotation: 
- $+175°$ X-axis rotation: 