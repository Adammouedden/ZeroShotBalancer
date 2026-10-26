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

`--show-io <N>` adds a line every N steps showing what the agent sees and does. Nothing is
replaced or suppressed: the reward table still prints as usual and these lines interleave with it,
so the reward trend and the policy's behaviour are visible together:
```uv run python sb_rl.py -a PPO -out STR-106 train -e ControlAgentTrainingEnv -t <timesteps> --show-io 2000```

Each line is `step=<timestep> | <observations> -> <actions>`, with every value named:

```step=500 | pitch: +1.656  pitch_dot: +0.457  wheel_l: -0.243  wheel_r: +0.318  d_speed: +0.402  d_yaw: -1.409 -> left: +0.009  right: -0.209```

`d_speed` is the error against the target wheel speed and `d_yaw` the error against the target
heading. The names live in `OBS_LABELS` and `ACTION_LABELS` in `sb_rl.py` and their order has to
match `_get_obs()` and `_set_action_space()` in
[`ControlAgentBaseEnv`](../../robot_environments/ControlAgentBaseEnv.py), so keep them in sync if
the observation vector changes. The observations are the ones the policy received, after the
`NormalizeObservation` wrapper, and so are not the environment's raw state. The actions are the
policy's own output before the env clips them to the action space, so values outside [-1, 1] are
expected and show the policy saturating.

Off by default, and worth keeping N large: training runs at a few hundred steps a second, so a
small interval floods the stream and makes the reward table hard to pick out.

Track with:
```tensorboard --logdir ../../../experiments/STR-106/training/logs/```

To compare several experiments in one tensorboard, point `--logdir` at `experiments/` itself.

Recent experiments have been running with 219136 timesteps, just because the first experiment we ever ran had 107 iterations which = 219136 steps with the default 2048 steps to iteration ratio