"""Tests for wheel action commands with physics disabled."""

import numpy as np
import pytest


@pytest.fixture
def action_env(env, monkeypatch):
    # Keep real joint/actuator data while isolating command assignment from physics.
    monkeypatch.setattr(
        "robot_environments.ControlAgentBaseEnv.mujoco.mj_step",
        lambda model, data, nstep: None,
    )
    monkeypatch.setattr(
        "robot_environments.ControlAgentBaseEnv.mujoco.mj_rnePostConstraint",
        lambda model, data: None,
    )
    # The base environment relies on subclasses to provide a reward function.
    monkeypatch.setattr(env, "_get_reward", lambda: 0.0, raising=False)
    monkeypatch.setattr(env, "_update_camera_follow", lambda: None)
    env.set_state(env.init_qpos.copy(), np.zeros_like(env.data.qvel))
    env.data.ctrl[:] = 0.0
    return env


@pytest.mark.parametrize(
    "joint_velocities, action, expected_commands",
    [
        pytest.param((0, 0), (0, 0), (0, 0), id="stationary-zero-action"),
        pytest.param((10, -10), (0, 0), (10, -10), id="moving-zero-action"),
        pytest.param((10, -10), (1, 0), (14, -10), id="positive-left-only"),
        pytest.param((10, -10), (-1, 0), (6, -10), id="negative-left-only"),
        pytest.param((10, -10), (0, 1), (10, -6), id="positive-right-only"),
        pytest.param((10, -10), (0, -1), (10, -14), id="negative-right-only"),
        pytest.param((10, -10), (1, 1), (14, -6), id="both-positive"),
        pytest.param((10, -10), (-1, -1), (6, -14), id="both-negative"),
        pytest.param((10, -10), (1, -1), (14, -14), id="opposite-left-positive"),
        pytest.param((10, -10), (-1, 1), (6, -6), id="opposite-right-positive"),
        pytest.param((10, -10), (0.5, -0.5), (12, -12), id="fractional-adjustments"),
        pytest.param((10, -10), (-0.25, 0.75), (9, -7), id="distinct-fractional-adjustments"),
        pytest.param((7, -13), (0.25, -0.5), (8, -15), id="unequal-initial-velocities"),
        pytest.param((-10, 10), (0.5, -0.5), (-8, 8), id="reversed-initial-velocities"),
        pytest.param((0, 0), (1, -1), (4, -4), id="start-from-rest-opposite-commands"),
        pytest.param((-1, 1), (0.5, -0.5), (1, -1), id="cross-zero"),
        pytest.param((4, -4), (-1, 1), (0, 0), id="reach-zero"),
    ],
)
def test_step_wheel_action_commands(action_env, joint_velocities, action, expected_commands):
    # Arrange: each action component adjusts its corresponding measured velocity.
    action_env.data.joint("torso_l_wheel").qvel[0] = joint_velocities[0]
    action_env.data.joint("torso_r_wheel").qvel[0] = joint_velocities[1]
    action = np.array(action, dtype=np.float32)

    # Act
    action_env.step(action)

    # Assert: commands retain the joint signs and left/right ordering.
    assert action_env.data.actuator("motor_l_wheel").ctrl[0] == pytest.approx(expected_commands[0])
    assert action_env.data.actuator("motor_r_wheel").ctrl[0] == pytest.approx(expected_commands[1])


def test_step_wheel_action_consecutive_calls(action_env):
    # Arrange: previous commands start at zero, with physics disabled by the fixture.
    action_env.data.joint("torso_l_wheel").qvel[0] = 10.0
    action_env.data.joint("torso_r_wheel").qvel[0] = -10.0
    action = np.array([1.0, -1.0], dtype=np.float32)

    # Act / Assert: the first call requests an adjustment from measured velocity.
    action_env.step(action)
    assert action_env.data.actuator("motor_l_wheel").ctrl[0] == pytest.approx(14.0)
    assert action_env.data.actuator("motor_r_wheel").ctrl[0] == pytest.approx(-14.0)
    assert action_env.get_wheel_velocities() == pytest.approx((10.0, -10.0))

    # Repeating the action must not accumulate adjustments from the previous ctrl.
    action_env.step(action)
    assert action_env.data.actuator("motor_l_wheel").ctrl[0] == pytest.approx(14.0)
    assert action_env.data.actuator("motor_r_wheel").ctrl[0] == pytest.approx(-14.0)
    assert action_env.get_wheel_velocities() == pytest.approx((10.0, -10.0))

    # Update the measurements explicitly before the third call.
    action_env.data.joint("torso_l_wheel").qvel[0] = 12.0
    action_env.data.joint("torso_r_wheel").qvel[0] = -9.0
    action_env.step(action)
    assert action_env.data.actuator("motor_l_wheel").ctrl[0] == pytest.approx(16.0)
    assert action_env.data.actuator("motor_r_wheel").ctrl[0] == pytest.approx(-13.0)


@pytest.mark.parametrize(
    "joint_velocities, action, expected_commands",
    [
        pytest.param((74.54, -74.54), (1, -1), (78.54, -78.54), id="reach-actuator-limits"),
        pytest.param((78, -78), (1, -1), (82, -82), id="exceed-actuator-limits"),
        pytest.param((-78, 78), (-1, 1), (-82, 82), id="exceed-limits-reversed-signs"),
    ],
)
def test_step_wheel_action_command_boundaries(
    action_env, joint_velocities, action, expected_commands
):
    # Arrange: commands approach or exceed the XML control limits of +/-78.54.
    action_env.data.joint("torso_l_wheel").qvel[0] = joint_velocities[0]
    action_env.data.joint("torso_r_wheel").qvel[0] = joint_velocities[1]
    action = np.array(action, dtype=np.float32)

    # Act: physics stays disabled so we can inspect the stored requests.
    action_env.step(action)

    # Assert: Python writes the requested values without clipping to actuator limits.
    assert action_env.data.actuator("motor_l_wheel").ctrl[0] == pytest.approx(expected_commands[0])
    assert action_env.data.actuator("motor_r_wheel").ctrl[0] == pytest.approx(expected_commands[1])
