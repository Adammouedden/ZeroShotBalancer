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
The System Under Test (SUT) here is the pitch of the self-balancing agent. It is algorithimically defined as `GET_PITCH / PITCH_MAX`, where `PITCH_MAX` is `0.25`. 

`GET_PITCH` takes the quaternions from `robot_body`, where the rotation order `[w, x, y, z, ]` is mapped to `[quat[0], quat[1], quat[2], quat[3]]`. Using these, we can create a `Rotation` object and then calculate the angle.

The conditions under which we will test will range from simple calculations to more complex. Inputs will be known quaternions.

**Standard test vectors:**
- Identity (no rotation): [1, 0, 0, 0].
In this case, the expected first angle is 0 deg.
- +90 deg X-axis rotation: [0.7071, 0.7071, 0, 0].
Expected angle: 90 deg.
- -90 deg X-axis: [0.7071, -0.7071, 0, 0].
Expected angle: -90 deg.

