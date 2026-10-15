import numpy as np

from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv
'''
test_control_agent_base_env.py

sb_rl.py imports envs → registers "ControlAgentTrainingEnv"
    ↓
gym.make("ControlAgentTrainingEnv")
    ↓
Creates ControlAgentTrainingEnv, which inherits ControlAgentBaseEnv
    ↓
sb_rl.py adds monitoring, video, and observation normalization wrappers
    ↓
model.learn() drives the interaction:
    reset() → initial observation → policy chooses action
                                      ↓
    next observation ← step(action) ←─┘
'''
def unit_test_pitch(qpos, qvel, unit_tests):
    # Robot body's free-joint quaternion, in MuJoCo's [w, x, y, z] order.
    for test_pair in unit_tests:
        quat, *expected_pitch = test_pair
        qpos[3:7] = quat

        # Update the simulation's derived body orientation
        env.set_state(qpos, qvel) 
        
        pitch = env.get_pitch()

        assert any(np.isclose(pitch, expected) for expected in expected_pitch), (
            f"Quaternion {quat}: expected one of {expected_pitch}, got {pitch}"
        )

env = ControlAgentBaseEnv(env_filename="base_world_env.xml")
try:
    env.reset() # Randomizes the robot's orientation.
    # For deterministic tests we must set.

    qpos = env.data.qpos.copy()
    qvel = np.zeros_like(env.data.qvel)

    pitch_unit_tests = [
        ([1, 0, 0, 0], 0.0), 
        ([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], np.pi/2),
        ([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0], -np.pi/2),
        ([np.cos(np.pi / 12), 0, np.sin(np.pi / 12), 0], 0),
        ([np.cos(-np.pi / 12), 0, np.sin(-np.pi / 12), 0], 0),
        ([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], 0),
        ([np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)], 0),
        ([np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0], np.pi/6),
        ([0, 1, 0, 0], np.pi, -np.pi), # handle both
        ([np.cos(np.pi / 2), np.sin(np.pi / 2), 0, 0], np.pi, -np.pi),
        ([-np.cos(-np.pi / 4), 0, 0, -np.sin(-np.pi / 4)], 0),
        ([np.cos(np.deg2rad(185) / 2), np.sin(np.deg2rad(185) / 2), 0, 0], -35*np.pi/36),
        ([np.cos(np.deg2rad(175) / 2), np.sin(np.deg2rad(175) / 2), 0, 0], 35*np.pi/36)
    ]

    unit_test_pitch(qpos, qvel, pitch_unit_tests)

finally:
    env.close()
