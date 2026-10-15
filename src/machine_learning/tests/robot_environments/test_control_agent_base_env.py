"""
Tests for extracting robot pitch from MuJoCo orientation.
Run using `uv run pytest -v tests/robot_environments/test_control_agent_base_env.py`
"""

import numpy as np
import pytest

from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv


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
