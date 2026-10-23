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

## Calibrated retraining (STR-118)

Both STR-118 training runs and evaluations are complete. Control's checkpoint and
evaluation are retained as the baseline for future experiments; the
[reference instructions and results](../../reference_baseline/README.md#calibrated-training-str-118)
document the external comparison. The commands below describe the completed workflow
and do not need to be repeated.

Run the following commands from `src/machine_learning/`. Training now selects
`base_world_calibrated.xml`, which includes `robot_calibrated.xml`, and prints its
resolved path at startup. Omitting `-m` starts a fresh policy.

```powershell
uv run python training/pre_training/control_agent/sb_rl.py -a PPO -out STR-118 train -e ControlAgentTrainingEnv -t 219136
```

In a second terminal in the same directory:

```powershell
uv run tensorboard --logdir experiments/STR-118/training/logs --port 6009
```

Open http://localhost:6009. Models are saved in
`experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/`; videos have their
own `training/movies/ControlAgentTrainingEnv_PPO/` folder. A populated model folder
is rejected before training starts: archive an earlier run before repeating it.

The existing Gymnasium observation normalization is retained. Its mean, variance,
count and epsilon are embedded in **each** best/periodic/final model ZIP. Training
evaluation uses a separate simulator with a frozen copy of the current statistics.
`best_model.zip` is selected by evaluation reward; `final_model.zip` is saved after
the requested training finishes and may perform differently. Resuming with `-m`
requires a checkpoint containing these statistics, plus an empty destination model
folder. Old checkpoints created with normalization but without saved statistics
cannot be reliably recovered; historical unnormalized policies remain supported.

After completing the [reference training run](../../reference_baseline/README.md#calibrated-training-str-118), evaluate it:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.mp4
```

After control training, evaluate its best checkpoint:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip -e ControlAgentTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json
```

Both evaluations use STR-118. Each saves a video and JSON metrics under
`experiments/STR-118/evaluation/video/<ENVIRONMENT>/base_world_calibrated/`; the new
reference run uses the `calibrated_run2/` subdirectory. Repeating the same evaluation requires `--force` or
a different `-o` output path. Normalized policies automatically restore their saved
statistics, which remain frozen throughout evaluation.

To record a side-by-side comparison of the two calibrated policies:

```powershell
uv run python evaluation/compare.py -out STR-118 -a training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -b experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip --xml-a base_world_calibrated.xml --xml-b base_world_calibrated.xml -s mixed_schedule_hard.json
```

The comparison goes to `experiments/STR-118/evaluation/comparison/mixed_schedule_hard/`.
These video commands use one seed by default; they are an initial performance check,
not a multi-seed qualification. Cross-policy rewards and the existing yaw metrics
have different definitions; the comparison marks those metrics as not comparable.
The completed reference training artifacts remain in `training/reference_baseline/`;
passing STR-118 to evaluation does not move them.
