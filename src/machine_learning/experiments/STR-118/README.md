# STR-118: Calibrated reference and control policies

Retrain both policies using `base_world_calibrated.xml`, which includes the
physical robot's calibrated MJCF from PR #8. Reference provides an external
comparison; control is the baseline for future experiments. Implementation details
are in [PR #9](https://github.com/Adammouedden/ZeroShotBalancer/pull/9).

## Training configuration

Both fresh PPO `MlpPolicy` runs completed **219,136 steps** on CPU. This budget
matches the earlier control experiments (107 rollouts of 2,048 steps). It gives
both policies the same interaction budget, without establishing convergence.
Existing rewards and PPO settings were retained; PPO training seeds were unset.
Training and periodic evaluation use independent calibrated simulators.
Reference uses raw observations; control retains `NormalizeObservation` and saves
its statistics in each checkpoint for frozen preprocessing during evaluation.

| Policy | Best checkpoint step | Completed steps | Model directory |
| --- | ---: | ---: | --- |
| Reference | 20,000 | 219,136 | `training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/` |
| Control | 80,000 | 219,136 | `experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/` |

Paths in commands and tables are relative to `src/machine_learning/`.
`best_model.zip` was selected using each policy's own training-evaluation reward.
Both runs also saved `final_model.zip`; the recorded evaluations use the best
checkpoints, not the final checkpoints. Best-model evaluation occurs every
20,000 steps, and periodic checkpoints are saved every 40,000 steps.

## Results

Both best checkpoints were evaluated on the unchanged hard and smooth schedules
using the calibrated world, deterministic inference, seed 0, 60 simulated seconds,
and a one-second grace period after fall resets. Each video has 3,000 frames at
50 fps. Smooth targets are piecewise constant, not continuously interpolated.

| Schedule | Policy | Falls | Mean absolute wheel-speed error (rad/s) | Mean absolute pitch (degrees) |
| --- | --- | ---: | ---: | ---: |
| Hard | Reference | 9 | 24.861 | 2.333 |
| Hard | Control | 25 | 14.074 | 5.107 |
| Smooth | Reference | 2 | 10.730 | 1.218 |
| Smooth | Control | 0 | 7.738 | 1.207 |

Control has lower speed error in both recordings. Reference has fewer falls on
the hard schedule; control has zero falls on the smooth schedule. Neither meets
the hard-schedule zero-fall objective.

Control's mean heading error is 0.740 rad (42.40 degrees) on the hard schedule
and 0.146 rad (8.39 degrees) on the smooth schedule. Speed error and heading error
have different units and require agreed tolerances to rank tracking performance.
Reference's yaw metric uses wheel-speed differential, so it cannot be directly
compared with control's heading error. Reward definitions also differ.

Raw results:

- Reference: [hard](evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.json) and [smooth](evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/smooth_tracking_schedule.json).
- Control: [hard](evaluation/video/ControlAgentTrainingEnv/base_world_calibrated/mixed_schedule_hard.json) and [smooth](evaluation/video/ControlAgentTrainingEnv/base_world_calibrated/smooth_tracking_schedule.json).

## Completed-run commands

Run from `src/machine_learning/`. These commands document the completed runs.
Existing model folders and evaluation files are protected against overwriting.
For another training run, choose fresh output locations first; for another
evaluation, use a new `-o` destination to preserve these results.

Reference training:

```powershell
uv run python training/reference_baseline/reference_sb_rl.py -a PPO train -e ReferenceBaselineTrainingEnv -t 219136
```

Reference's script-relative outputs are `models/calibrated_run2/`,
`logs/calibrated_run2/`, and `movies/calibrated_run2/`. Model and movie roots contain
a `ReferenceBaselineTrainingEnv_PPO/` subdirectory. Its original tracked reference
checkpoint remains separate and unchanged.

Control training:

```powershell
uv run python training/pre_training/control_agent/sb_rl.py -a PPO -out STR-118 train -e ControlAgentTrainingEnv -t 219136
```

Control outputs live under this ticket's `training/` directory. Both policies can
share STR-118 because evaluation outputs are separated by environment and world.
Assigning a ticket to reference evaluation does not relocate reference training
outputs.

TensorBoard (run each command in a separate terminal):

```powershell
uv run tensorboard --logdir training/reference_baseline/logs/calibrated_run2 --port 6008
uv run tensorboard --logdir experiments/STR-118/training/logs --port 6009
```

Open `http://localhost:6008` for reference and `http://localhost:6009` for control.

Reference evaluations:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.mp4
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s smooth_tracking_schedule.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/smooth_tracking_schedule.mp4
```

Control evaluations:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip -e ControlAgentTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json --seed 0 -o experiments/STR-118/evaluation/video/ControlAgentTrainingEnv/base_world_calibrated/mixed_schedule_hard.mp4
uv run python evaluation/evaluate.py video -out STR-118 -m experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip -e ControlAgentTrainingEnv -x base_world_calibrated.xml -s smooth_tracking_schedule.json --seed 0 -o experiments/STR-118/evaluation/video/ControlAgentTrainingEnv/base_world_calibrated/smooth_tracking_schedule.mp4
```

Each evaluation writes JSON metrics beside its video and records results in
[metadata.json](metadata.json). Videos, reference training logs, and periodic/final
checkpoints remain local under the existing ignore rules. Both calibrated best
models, evaluation JSONs, and the control training log are tracked.

## Verification and limits

[AUDIT.json](AUDIT.json) records the completed artifact checks: 14 checkpoint
archives, both logs through 219,136 steps, best-checkpoint selection, frozen control
normalization, model loading/inference, and agreement between evaluation metadata,
JSON, and video duration. Source hashes describe files at audit time, not launch
snapshots. The audit remains a record of those checks as the source evolves.

- Hard targets reach +/-60 rad/s, exceeding the calibrated actuator command range
  of +/-20.6 rad/s. Smooth targets range from -10 to 20 rad/s; that alone does not
  establish physical feasibility during turns or transients.
- One training run and one evaluation seed per policy do not establish statistical
  superiority. Best checkpoints also come from different training steps and reward
  definitions.
- The inherited evaluator predicts an action before assigning the next scheduled
  target. Results reflect that timing convention and include fall-recovery periods.
- These are SB3 simulation results. Hardware inference and exported-model
  preprocessing have not been qualified by this experiment.
