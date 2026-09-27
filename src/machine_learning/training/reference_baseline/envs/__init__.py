from gymnasium.envs.registration import register

register(
    id="ReferenceBaselineTrainingEnv",
    entry_point="envs.reference_baseline_training_env:ReferenceBaselineTrainingEnv",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env03-v2",
    entry_point="envs.reference_baseline_fine_tuning:ReferenceBaselineFineTuning",
    max_episode_steps=1200,
    reward_threshold=6000,
)
