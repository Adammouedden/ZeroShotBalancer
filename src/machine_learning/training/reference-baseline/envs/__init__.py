from gymnasium.envs.registration import register

register(
    id="Env01-v3",
    entry_point="envs.env01_v3:Env01_v3",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env03-v2",
    entry_point="envs.env03_v2:Env03_v2",
    max_episode_steps=1200,
    reward_threshold=6000,
)
