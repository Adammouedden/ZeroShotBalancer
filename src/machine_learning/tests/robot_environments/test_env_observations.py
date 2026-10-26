"""Tests for extracting and scaling robot observations from MuJoCo state."""

import numpy as np
import pytest

from robot_environments.ControlAgentBaseEnv import (
    ControlAgentBaseEnv,
    PITCH_DOT_MAX,
    PITCH_MAX,
    WHEEL_SPEED_MAX,
)
from sb_rl import ACTION_LABELS, OBS_LABELS


@pytest.fixture
def env():
    environment = ControlAgentBaseEnv(env_filename="base_world_env.xml")
    try:
        environment.reset()
        yield environment
    finally:
        environment.close()


@pytest.mark.parametrize(
    "quat, expected_pitches",
    [
        pytest.param([1, 0, 0, 0], (0.0,), id="identity"),
        pytest.param([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], (np.pi/2,), id="positive-90-x"),
        pytest.param([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0], (-np.pi/2,), id="negative-90-x"),
        pytest.param([np.cos(np.pi / 12), 0, np.sin(np.pi / 12), 0], (0,), id="positive-30-y"),
        pytest.param([np.cos(-np.pi / 12), 0, np.sin(-np.pi / 12), 0], (0,), id="negative-30-y"),
        pytest.param([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], (0,), id="positive-90-z"),
        pytest.param([np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)], (0,), id="negative-90-z"),
        pytest.param([np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0], (np.pi/6,), id="positive-30-x"),
        pytest.param([0, 1, 0, 0], (np.pi, -np.pi,), id="180-x"),
        pytest.param([np.cos(np.pi / 2), np.sin(np.pi / 2), 0, 0], (np.pi, -np.pi,), id="180-x-calculated"),
        pytest.param([-np.cos(-np.pi / 4), 0, 0, -np.sin(-np.pi / 4)], (0,), id="negated-negative-90-z"),
        pytest.param([np.cos(np.deg2rad(185) / 2), np.sin(np.deg2rad(185) / 2), 0, 0], (-35*np.pi/36,), id="185-x-wraps-negative"),
        pytest.param([np.cos(np.deg2rad(175) / 2), np.sin(np.deg2rad(175) / 2), 0, 0], (35*np.pi/36,), id="175-x"),
    ],
)
def test_get_pitch(env, quat, expected_pitches):
    # Arrange: MuJoCo free-joint quaternions use [w, x, y, z] order.
    qpos = env.init_qpos.copy()
    qvel = np.zeros_like(env.data.qvel)
    qpos[3:7] = quat
    env.set_state(qpos, qvel)

    # Act
    pitch = env.get_pitch()

    # Assert
    assert any(
        np.isclose(pitch, expected, atol=1e-8, rtol=0)
        for expected in expected_pitches
    ), f"Quaternion {quat}: expected one of {expected_pitches}, got {pitch}"


@pytest.mark.parametrize(
    "last_pitch, last_time, pitch, time, expected_rate",
    [
        pytest.param(None, None, 0.2, 1.0, 0.0, id="first-reading"),
        pytest.param(0.1, None, 0.2, 1.0, 0.0, id="missing-previous-time"),
        pytest.param(None, 0.5, 0.2, 1.0, 0.0, id="missing-previous-pitch"),
        pytest.param(0.2, 1.0, 0.2, 1.1, 0.0, id="unchanged-pitch"),
        pytest.param(0.1, 1.0, 0.2, 1.1, 1.0, id="increasing-pitch"),
        pytest.param(0.2, 1.0, 0.1, 1.1, -1.0, id="decreasing-pitch"),
        pytest.param(-0.1, 1.0, 0.1, 1.2, 1.0, id="crossing-zero"),
        pytest.param(0.1, 1.0, 0.2, 1.2, 0.5, id="longer-time-interval"),
        pytest.param(0.1, 1.0, 0.2, 1.0, 0.0, id="same-timestamp"),
        pytest.param(0.1, 1.0, 0.2, 0.0, 0.0, id="time-moves-backward"),
    ],
)
def test_get_pitch_dot_alt(
    env, monkeypatch, last_pitch, last_time, pitch, time, expected_rate
):
    # Arrange: control pitch independently of quaternion extraction.
    env.last_pitch = last_pitch
    env.last_time = last_time
    env.data.time = time
    monkeypatch.setattr(env, "get_pitch", lambda: pitch)

    # Act
    pitch_rate = env.get_pitch_dot_alt()

    # Assert: even missing history or nonpositive dt must refresh history.
    assert pitch_rate == pytest.approx(expected_rate, abs=1e-8, rel=0)
    assert env.last_pitch == pitch
    assert env.last_time == time


def test_get_pitch_dot_alt_consecutive_readings(env, monkeypatch):
    # Arrange
    env.last_pitch = 0.0
    env.last_time = 1.0
    pitches = iter([0.1, 0.3])
    monkeypatch.setattr(env, "get_pitch", lambda: next(pitches))

    # Act / Assert: the second call must use history saved by the first.
    env.data.time = 1.1
    assert env.get_pitch_dot_alt() == pytest.approx(1.0)
    assert env.last_pitch == 0.1
    assert env.last_time == 1.1

    env.data.time = 1.2
    assert env.get_pitch_dot_alt() == pytest.approx(2.0)
    assert env.last_pitch == 0.3
    assert env.last_time == 1.2


def test_get_pitch_dot_alt_angle_wrapping_diagnostic(env, monkeypatch):
    # Arrange: a positive crossing of 180 degrees wraps to -179 degrees.
    env.last_pitch = np.deg2rad(179)
    env.last_time = 1.0
    pitch = np.deg2rad(-179)
    env.data.time = 1.1
    monkeypatch.setattr(env, "get_pitch", lambda: pitch)

    # Act
    pitch_rate = env.get_pitch_dot_alt()

    # Assert: document the current raw difference (-358 degrees / 0.1 s).
    # Unwrapping would instead give +2 degrees / 0.1 s, or +0.349066 rad/s.
    assert pitch_rate == pytest.approx(-62.482787, abs=1e-6, rel=0)
    assert env.last_pitch == pitch
    assert env.last_time == 1.1


@pytest.mark.parametrize("pitch_dot_max", [PITCH_DOT_MAX, 2.0], ids=["current-scale", "non-unit-scale"])
def test_get_obs_pitch_rate_scaling(env, monkeypatch, pitch_dot_max):
    # Arrange: use the real derivative calculation with valid history.
    env.last_pitch = 0.1
    env.last_time = 1.0
    env.data.time = 1.2
    monkeypatch.setattr(env, "get_pitch", lambda: 0.2)
    monkeypatch.setattr(
        "robot_environments.ControlAgentBaseEnv.PITCH_DOT_MAX", pitch_dot_max
    )

    # Act
    observation = env._get_obs()

    # Assert: observation index 1 contains the normalized 0.5 rad/s rate.
    assert observation[1] == pytest.approx(0.5 / pitch_dot_max)
    assert env.last_pitch == 0.2
    assert env.last_time == 1.2


@pytest.mark.parametrize(
    "joint_name, wheel_index",
    [
        pytest.param("torso_l_wheel", 0, id="left-wheel"),
        pytest.param("torso_r_wheel", 1, id="right-wheel"),
    ],
)
@pytest.mark.parametrize(
    "raw_velocity, expected_scaled",
    [
        pytest.param(0.0, 0.0, id="stationary"),
        pytest.param(42.5, 1.0, id="positive-velocity"),
        pytest.param(-42.5, -1.0, id="negative-velocity"),
        pytest.param(170.0, 4.0, id="positive-scaling-reference"),
        pytest.param(-170.0, -4.0, id="negative-scaling-reference"),
        pytest.param(212.5, 5.0, id="above-scaling-reference-no-clipping"),
        pytest.param(-212.5, -5.0, id="below-negative-scaling-reference-no-clipping"),
    ],
)
def test_wheel_velocity_and_observation_scaling(
    env, joint_name, wheel_index, raw_velocity, expected_scaled
):
    # Arrange: set the actual joint state, keeping the other wheel stationary.
    env.set_state(env.init_qpos.copy(), np.zeros_like(env.data.qvel))
    env.data.joint(joint_name).qvel[0] = raw_velocity
    expected_velocities = [0.0, 0.0]
    expected_velocities[wheel_index] = raw_velocity
    expected_observations = [0.0, 0.0]
    expected_observations[wheel_index] = expected_scaled

    # Act: exercise both MuJoCo joint lookup and observation assembly.
    velocities = env.get_wheel_velocities()
    observation = env._get_obs()

    # Assert: left maps to tuple index 0 / obs index 2, right to 1 / 3.
    assert velocities == pytest.approx(tuple(expected_velocities))
    assert observation[2:4] == pytest.approx(expected_observations)


@pytest.mark.parametrize(
    "joint_velocities, expected_observations",
    [
        pytest.param((42.5, -85.0), [1.0, -2.0], id="distinct-wheel-velocities"),
        pytest.param((-85.0, 42.5), [-2.0, 1.0], id="swapped-wheel-velocities"),
    ],
)
def test_wheel_velocity_sources_are_independent(
    env, joint_velocities, expected_observations
):
    # Arrange: give both joints distinct velocities to detect swaps or aliasing.
    env.set_state(env.init_qpos.copy(), np.zeros_like(env.data.qvel))
    env.data.joint("torso_l_wheel").qvel[0] = joint_velocities[0]
    env.data.joint("torso_r_wheel").qvel[0] = joint_velocities[1]

    # Act
    velocities = env.get_wheel_velocities()
    observation = env._get_obs()

    # Assert: raw signs and left/right ordering survive both read paths.
    assert velocities == pytest.approx(joint_velocities)
    assert observation[2:4] == pytest.approx(expected_observations)


@pytest.mark.parametrize(
    "joint_velocities, target_speed, expected_speed, expected_difference",
    [
        pytest.param((0.0, 0.0), 0.0, 0.0, 0.0, id="stationary-zero-target"),
        pytest.param((0.0, 0.0), 42.5, 0.0, 1.0, id="stationary-positive-target"),
        pytest.param((0.0, 0.0), -42.5, 0.0, -1.0, id="stationary-negative-target"),
        pytest.param((42.5, -42.5), 42.5, 42.5, 0.0, id="matching-positive-target"),
        pytest.param((-42.5, 42.5), -42.5, -42.5, 0.0, id="matching-negative-target"),
        pytest.param((42.5, -42.5), 85.0, 42.5, 1.0, id="actual-below-target"),
        pytest.param((85.0, -85.0), 42.5, 85.0, -1.0, id="actual-above-target"),
        pytest.param((-42.5, 42.5), 42.5, -42.5, 2.0, id="moving-opposite-to-target"),
        pytest.param((85.0, 0.0), 0.0, 42.5, -1.0, id="left-wheel-only"),
        pytest.param((0.0, -85.0), 0.0, 42.5, -1.0, id="right-wheel-only"),
        pytest.param((85.0, -42.5), 0.0, 63.75, -1.5, id="unequal-wheel-speeds"),
        pytest.param((42.5, 42.5), 0.0, 0.0, 0.0, id="equal-raw-velocities-cancel"),
        pytest.param((0.0, 0.0), 170.0, 0.0, 4.0, id="large-positive-difference-no-clipping"),
        pytest.param((170.0, -170.0), 0.0, 170.0, -4.0, id="large-negative-difference-no-clipping"),
    ],
)
def test_get_obs_target_wheel_speed_difference(
    env, joint_velocities, target_speed, expected_speed, expected_difference
):
    # Arrange: use actual joint velocities and a directly assigned speed target.
    env.set_state(env.init_qpos.copy(), np.zeros_like(env.data.qvel))
    env.data.joint("torso_l_wheel").qvel[0] = joint_velocities[0]
    env.data.joint("torso_r_wheel").qvel[0] = joint_velocities[1]
    env.target_wheel_speed = target_speed

    # Act
    wheel_speed = env.get_wheel_speed()
    observation = env._get_obs()

    # Assert: account for the right wheel's sign, then scale target minus actual.
    assert wheel_speed == pytest.approx(expected_speed)
    assert observation[4] == pytest.approx(expected_difference)


@pytest.mark.parametrize(
    "target_yaw, actual_yaw, expected_difference",
    [
        pytest.param(0.0, 0.0, 0.0, id="both-zero"),
        pytest.param(np.pi / 3, np.pi / 3, 0.0, id="matching-nonzero-headings"),
        pytest.param(np.pi / 2, 0.0, np.pi / 2, id="positive-difference"),
        pytest.param(0.0, np.pi / 2, -np.pi / 2, id="negative-difference"),
        pytest.param(np.pi / 6, -np.pi / 6, np.pi / 3, id="opposite-sides-of-zero"),
        pytest.param(np.deg2rad(-179), np.deg2rad(179), np.deg2rad(2), id="positive-boundary-crossing"),
        pytest.param(np.deg2rad(179), np.deg2rad(-179), np.deg2rad(-2), id="negative-boundary-crossing"),
        pytest.param(np.pi, 0.0, -np.pi, id="positive-half-turn"),
        pytest.param(-np.pi, 0.0, -np.pi, id="negative-half-turn"),
        pytest.param(np.pi, -np.pi, 0.0, id="equivalent-180-degree-headings"),
        pytest.param(2 * np.pi + np.pi / 4, 0.0, np.pi / 4, id="target-includes-full-revolution"),
        pytest.param(0.0, 2 * np.pi + np.pi / 4, -np.pi / 4, id="actual-includes-full-revolution"),
    ],
)
def test_get_obs_target_yaw_difference(
    env, monkeypatch, target_yaw, actual_yaw, expected_difference
):
    # Arrange: isolate wrapping, including headings beyond one full revolution.
    env.target_yaw = target_yaw
    monkeypatch.setattr(env, "get_yaw", lambda: actual_yaw)

    # Act
    observation = env._get_obs()

    # Assert: index 5 is the signed heading error in radians; +pi maps to -pi.
    assert observation[5] == pytest.approx(expected_difference)


@pytest.mark.parametrize(
    "quat, expected_yaw, target_yaw, expected_difference",
    [
        pytest.param([1, 0, 0, 0], 0.0, 0.0, 0.0, id="identity"),
        pytest.param(
            [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
            np.pi / 2, 0.0, -np.pi / 2, id="positive-90-z",
        ),
        pytest.param(
            [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)],
            -np.pi / 2, 0.0, np.pi / 2, id="negative-90-z",
        ),
        pytest.param(
            [np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0],
            0.0, np.pi / 4, np.pi / 4, id="positive-30-x",
        ),
    ],
)
def test_get_yaw_and_observation_difference(
    env, quat, expected_yaw, target_yaw, expected_difference
):
    # Arrange: exercise real quaternion extraction with MuJoCo's [w, x, y, z].
    qpos = env.init_qpos.copy()
    qpos[3:7] = quat
    env.set_state(qpos, np.zeros_like(env.data.qvel))
    env.target_yaw = target_yaw

    # Act
    yaw = env.get_yaw()
    observation = env._get_obs()

    # Assert: verify the extracted heading and its error in the observation.
    assert yaw == pytest.approx(expected_yaw, abs=1e-8, rel=0)
    assert observation[5] == pytest.approx(expected_difference)


def test_obs_labels_cover_the_observation_space(env):
    # Arrange / Act / Assert: a label per observation, so none is printed unnamed and none of
    # the values is dropped by the zip in sb_rl.format_vector.
    assert len(OBS_LABELS) == env.observation_space.shape[0], (
        f"OBS_LABELS has {len(OBS_LABELS)} names for "
        f"{env.observation_space.shape[0]} observations: {OBS_LABELS}"
    )


def test_action_labels_cover_the_action_space(env):
    # Arrange / Act / Assert: likewise for the actions the policy returns.
    assert len(ACTION_LABELS) == env.action_space.shape[0], (
        f"ACTION_LABELS has {len(ACTION_LABELS)} names for "
        f"{env.action_space.shape[0]} actions: {ACTION_LABELS}"
    )


@pytest.mark.parametrize(
    "label, expected_value",
    [
        pytest.param("pitch", 0.1 / PITCH_MAX, id="pitch"),
        pytest.param("pitch_dot", 0.2 / PITCH_DOT_MAX, id="pitch_dot"),
        pytest.param("wheel_l", 10.0 / WHEEL_SPEED_MAX * 4, id="wheel_l"),
        pytest.param("wheel_r", 20.0 / WHEEL_SPEED_MAX * 4, id="wheel_r"),
        pytest.param("d_speed", 25.0 / WHEEL_SPEED_MAX * 4, id="d_speed"),
        pytest.param("d_yaw", 0.7, id="d_yaw"),
    ],
)
def test_obs_labels_match_observation_order(env, monkeypatch, label, expected_value):
    # Arrange: every observation gets a distinct value, so pairing a label with the wrong
    # position cannot coincidentally produce the expected number.
    monkeypatch.setattr(env, "get_pitch", lambda: 0.1)
    monkeypatch.setattr(env, "get_pitch_dot_alt", lambda: 0.2)
    monkeypatch.setattr(env, "get_wheel_velocities", lambda: (10.0, 20.0))
    monkeypatch.setattr(env, "get_wheel_speed", lambda: 5.0)
    monkeypatch.setattr(env, "get_yaw", lambda: 0.0)
    env.target_wheel_speed = 30.0
    env.target_yaw = 0.7

    # Act: label the observation the way sb_rl prints it.
    labelled = dict(zip(OBS_LABELS, env._get_obs()))

    # Assert: a reordered _get_obs() fails here naming the label, rather than only tripping the
    # index-based assertions above with a bare numeric mismatch.
    assert labelled[label] == pytest.approx(expected_value), (
        f"OBS_LABELS is out of sync with _get_obs(): {label} is labelling the wrong "
        f"position. Labels are {OBS_LABELS}, observation is {env._get_obs()}"
    )
