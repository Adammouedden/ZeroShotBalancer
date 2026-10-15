from robot_environments.ControlAgentBaseEnv import ControlAgentBaseEnv


'''
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
