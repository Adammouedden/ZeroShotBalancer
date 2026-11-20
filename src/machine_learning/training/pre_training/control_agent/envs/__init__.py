from gymnasium.envs.registration import register

register(
    id="ControlAgentTrainingEnv",
    entry_point="envs.control_agent_training_environment:ControlAgentTrainingEnv",
    max_episode_steps=6000,
    reward_threshold=6000,
)
#Find the minimal training RL
register(
    id="MinimalTrainingEnv",
    entry_point="envs.minimal_training_environment:MinimalTrainingEnv",
    max_episode_steps=6000,
    reward_threshold=6000,
)