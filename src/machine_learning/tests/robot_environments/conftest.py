"""Shared MuJoCo environment fixture for control-agent tests."""

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
