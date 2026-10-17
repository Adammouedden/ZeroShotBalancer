# Control Agent

This directory contains all code for creating our pre-trained policy, dubbed Control Agent as it will serve as a control variable for future testing. It takes the mantle of the "reference-baseline" agent as soon as it surpasses it in balance survival time, adherence to target velocity, and mean reward.

---

## Training:
From the control_agent directory, passing the Jira ticket this run belongs to:
```uv run python sb_rl.py -a PPO -out STR-106 train -e ControlAgentTrainingEnv -t <timesteps>```

`-out`/`--output_path` is required. Every artifact for the run — tensorboard logs, the best
model, checkpoints and training movies — is written to
[`experiments/<TICKET>/`](../../../experiments/README.md) instead of to shared `logs/`,
`models/` and `movies/` folders that the next run would overwrite.

Track with:
```tensorboard --logdir ../../../experiments/STR-106/training/logs/```

To compare several experiments in one tensorboard, point `--logdir` at `experiments/` itself.

Recent experiments have been running with 219136 timesteps, just because the first experiment we ever ran had 107 iterations which = 219136 steps with the default 2048 steps to iteration ratio