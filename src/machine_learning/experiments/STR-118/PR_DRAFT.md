STR-118: Retrain reference and control policies on the calibrated robot

## Purpose and outcome

The physical robot now has a calibrated MuJoCo model, introduced in PR #8. The training entry points still selected the original world, so rerunning them would not automatically train policies for the calibrated robot. The control environment also contained an unfinished draft block that prevented import, and its observation-normalization statistics were not preserved with saved policies.

This change connects reference and control training to the calibrated scene, gives both policies independent training and evaluation simulators, preserves the control policy's preprocessing in its checkpoints, and repairs the evaluation commands. Both use a 219,136-step training budget and the same evaluation protocol on the hard and smooth schedules.

**Both policies completed 219,136 training steps and two 60-second evaluations.** On the hard schedule, reference fell 9 times and control fell 25 times. On the smooth schedule, reference fell 2 times and control had zero falls. Control had lower mean speed error on both schedules. These results establish a starting point for future experiments; the hard-schedule zero-fall objective remains unmet.

## Changes

### Reference training

- `training/reference_baseline/reference_sb_rl.py` passes `xml_file='base_world_calibrated.xml'` when creating its training environment.
- Training uses CPU on the current machine.
- New reference outputs go to `models/calibrated_run2`, `logs/calibrated_run2`, and `movies/calibrated_run2` within the reference-baseline directory, resolved independently of the launch directory.
- Periodic evaluation uses a separate simulator with the same calibrated XML, so it cannot reset or advance the live training simulator.
- Refuse populated model directories, save a final checkpoint when learning returns normally, embed the training XML and independent-evaluation setting in checkpoints, and close both simulators on completion or errors.
- The original tracked reference checkpoint remains intact. The calibrated reference checkpoint is stored separately.
- The reference policy retains its existing observations, reward function, PPO architecture, and training schedule.

### Control training

- `ControlAgentTrainingEnv` accepts an `xml_file` argument and forwards it to `ControlAgentBaseEnv`.
- `sb_rl.py` explicitly selects the calibrated world and prints its resolved path at startup.
- A separate evaluation environment replaces evaluation on the training simulator. Periodic evaluation therefore does not reset or advance the control policy's live training environment.
- Control model output is organized under `experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/`. Training refuses to overwrite a populated model folder.
- Training videos include the environment/algorithm in their directory name, preventing collisions between different policies sharing a ticket.
- Normal completion saves `final_model.zip` in addition to the reward-selected best checkpoint and periodic checkpoints.

### Checkpoint normalization

The existing STR-105 `Gymnasium NormalizeObservation` wrapper is retained. This change does not replace it with a different normalization algorithm or introduce reward normalization.

The new `visibility/observation_normalization.py` captures the observation mean, variance, sample count, epsilon, and a schema version as ordinary checkpoint metadata. Those statistics are embedded in the **same SB3 ZIP** as each best, periodic, and final model, avoiding an external statistics file that could be lost or paired with the wrong weights.

- Periodic saves snapshot statistics immediately before saving the policy.
- Training evaluation receives a frozen copy of the current training statistics; the best model saves that matching snapshot.
- Evaluation and the control script's interactive SB3 `test` command restore the saved statistics and do not update them during inference.
- Resume restores the statistics before attaching the training environment.
- Invalid statistics or missing statistics on a checkpoint marked as requiring them raise an error.
- Historical policies with no normalization metadata remain on the legacy raw-observation path, which supports the reference policy. An older *normalized* policy that never saved its statistics cannot have those statistics reconstructed retrospectively.

### Evaluation and shared-ticket outputs

- Merge the conflicting `video()` definitions in `evaluation/evaluate.py` so ticket, output, XML, and overwrite options reach one callback.
- Add the missing `xml_a` and `xml_b` callback parameters in `compare.py`, allowing the CLI to select each world correctly.
- Restore control checkpoint normalization during video evaluation, including scheduled steps that bypass the usual Gym wrapper chain.
- Save a JSON metrics file alongside each individual evaluation video, including the model path, environment, XML, schedule, evaluation seed, and normalization mode.
- Separate individual evaluation outputs by environment and world:

  ```text
  experiments/STR-118/evaluation/video/
    ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.{mp4,json}
    ControlAgentTrainingEnv/base_world_calibrated/mixed_schedule_hard.{mp4,json}
  ```

- Refuse to replace existing evaluation outputs unless `--force` is supplied.
- Close the video environment on errors as well as normal completion.

STR-118 can therefore contain both policies' evaluations without filename collisions. Reference training artifacts live in `training/reference_baseline/`; assigning STR-118 to evaluation does not relocate them. A ticket containing several training models should pass model paths explicitly to `compare.py`.

## Configuration and rationale

### Model and simulation

| Setting | Value | Reason and interpretation |
| --- | --- | --- |
| World MJCF | `base_world_calibrated.xml` | Loads the complete scene, floor and contact settings, and includes `robot_calibrated.xml`; the robot XML alone is not the complete training scene. |
| Calibrated robot | Existing model from PR #8 | Reused without changing its calibration values in this PR. Some quantities in that model are documented estimates; this retraining does not independently validate the calibration. |
| Physics timestep | 0.00002 s | Retained from the world XML. |
| Physics steps per action | 250 | Retained from the base environments; one training step is 0.005 s, or 200 control steps/s. |
| Training episode cap | 6,000 environment steps | Existing 30-second episode limit. A 60-second recorded evaluation is a separate protocol with resets after falls. |
| Observations/actions | 6 observations, 2 wheel actions | Preserves each policy's interface. The reference and control sixth observations have different yaw meanings. |
| Base observation scaling | `PITCH_MAX=0.25`, `PITCH_DOT_MAX=1`, `WHEEL_SPEED_MAX=170` | Existing base-environment scales are retained rather than retuned for the calibrated motor. Control additionally applies its saved running normalization. These constants are not claims about new hardware limits. |
| Action interval and increment | `[-1, 1]` per wheel, scaled by 4 | Existing wheel-speed-increment interface; actions are added to measured wheel velocity before commanding the velocity actuator. |
| Actuator velocity command limits | +/-20.6 rad/s | Inherited from the calibrated robot; relevant when interpreting the much larger hard-schedule targets. |
| Actuator force limits | +/-0.115 per wheel hinge actuator | Inherited calibrated torque limits, not newly tuned in this change. |

The calibrated model specifies chassis mass 0.825160 kg and wheel masses 0.060 kg each. Its explicit inertials and world configuration are reused. This PR changes which scene training loads; it does not create new CAD estimates or tune physical parameters.

### Training budget and PPO

| Setting | Value | Reason |
| --- | --- | --- |
| Algorithm/policy | PPO / `MlpPolicy` | Existing training approach for the six numerical observations; no architecture change. |
| Budget per run | **219,136 steps** | Matches the team's prior control experiments: 107 rollouts x 2,048 steps. Using it for both policies gives the same initial interaction budget. It is not a convergence guarantee. |
| Device | CPU | Matches the available machine. Device selection concerns execution, not the amount of training experience. |
| Rollout length | 2,048 | Retained PPO configuration and basis of the historical budget. |
| Batch size / optimization epochs | 64 / 10 | Existing PPO settings; not newly tuned. |
| Learning rate | 0.0003 | Existing PPO setting. |
| Discount / GAE lambda | 0.99 / 0.95 | Existing return/advantage settings. |
| Policy clip range | 0.2 | Existing PPO default; no change to the update constraint. |
| Entropy / value coefficients | 0.0 / 0.5 | Existing PPO settings. |
| Maximum gradient norm | 0.5 | Existing gradient-clipping setting. |
| Training evaluation frequency | 20,000 steps | Existing checkpoint-selection cadence; ten evaluations are scheduled within the budget. |
| Episodes per training evaluation | 5 | Retained `EvalCallback` default. |
| Periodic checkpoint frequency | 40,000 steps | Existing recovery cadence; periodic saves are scheduled through 200,000 steps. |
| Video recording frequency | Every 50 episodes | Existing monitoring cadence. |
| Normalization epsilon | `1e-8` | Retained numerical stabilizer in the observation standard-deviation denominator. |
| Legacy stopping callbacks | Reward threshold 6,000; no-improvement patience 5 after at least 10,000 evaluations | Retained, not newly tuned. The no-improvement callback cannot become active within this budget's ten scheduled evaluations. |
| PPO training seed | Unset (`null`) | Existing setting; no claim of fully reproducible or matched-seed training. |


### Reward weights and training targets

The reward weights are inherited rather than optimized for the new calibrated robot:

| Component | Reference | Control | Existing role |
| --- | ---: | ---: | --- |
| Alive | 0.6 | 0.6 | Per-step survival reward. |
| Pitch | 0.05 | 0.05 | Penalize absolute pitch. |
| Speed error | 0.15 | 0.15 | Penalize the normalized speed mismatch; existing `MAX_DV=40` clips/scales this term. |
| Lean shaping | 10.0 | 5.0 | Encourage leaning in the direction needed to change speed. Control retains the earlier STR-97 reduction. |
| Actual heading error | Not a separate term | 0.15 | Control penalizes wrapped target-heading minus actual-yaw error. |
| Wheel-yaw / uneven-wheel-speed term | 0.007 | 0.007 | Retains the existing `abs(target_yaw - get_wheel_yaw())` formulation. Its meaning differs from actual heading error and is not retuned here. |

Both training environments retain the existing changing speed schedule and +/-0.0349066-radian pitch-measurement offset. The speed schedule can scale its initial target by up to three, reaching magnitudes near 60 rad/s. Control retains one random heading target per episode and its environment-local target generator seeded with 42. That local generator does not seed PPO initialization, all reset randomness, or the whole training process.

### Evaluation protocol

| Setting | Value | Reason |
| --- | --- | --- |
| Schedules | `mixed_schedule_hard.json` and `smooth_tracking_schedule.json` | Both existing schedules are used unchanged. The smooth schedule uses smaller target changes, held constant between entries; it is not continuous interpolation. |
| World | `base_world_calibrated.xml` for both | Tests the new policies in the same calibrated scene. |
| Duration | 60 simulated seconds each | Last scheduled change at 58 seconds plus the evaluator's existing 2-second hold. |
| Evaluation seed | 0 | Used by all four completed evaluations. One seed is an initial check, not statistical qualification. |
| Policy inference | Deterministic | Uses the policy's deterministic action prediction during evaluation. |
| Reset grace | 1 second | Existing protocol temporarily holds targets at zero after a fall reset. Metrics include these recovery intervals. |
| Video rate | 50 frames/s | Existing evaluator setting, giving 3,000 frames for each 60-second recording. |

## Results

Both TensorBoard logs and final checkpoints reach 219,136 steps. The evaluated best checkpoints were selected by each policy's **own training-evaluation reward**:

| Evidence | Reference | Control |
| --- | ---: | ---: |
| Completed training steps | 219,136 | 219,136 |
| Selected best checkpoint step | 20,000 | 80,000 |
| Final-step checkpoint available | Yes, `final_model.zip` at 219,136 | Yes, `final_model.zip` at 219,136 |
| Evaluation duration / seed | 60 s / 0 | 60 s / 0 |

Both schedules use these same best checkpoints:

| Schedule | Policy | Falls | Mean absolute wheel-speed error (rad/s) | Mean absolute pitch (degrees) |
| --- | --- | ---: | ---: | ---: |
| Hard | Reference | 9 | 24.861 | 2.333 |
| Hard | Control | 25 | 14.074 | 5.107 |
| Smooth | Reference | 2 | 10.730 | 1.218 |
| Smooth | Control | **0** | **7.738** | 1.207 |

Control has lower mean speed error on both schedules. On the hard schedule, reference has fewer falls and smaller pitch deviations. On the smooth schedule, control has no falls, compared with two for reference, and their mean pitch errors are similar. One evaluation seed per schedule does not establish overall superiority or hardware readiness.

The JSON also contains yaw error and reward totals. Those must not be ranked across policies: the reference yaw metric measures a wheel-speed differential, while control measures wrapped heading error in radians; reward definitions also differ. The comparison tool already marks these cross-environment metrics as not comparable.

Control's mean heading error is 0.740 rad (42.40 degrees) on the hard schedule and
0.146 rad (8.39 degrees) on the smooth schedule. These figures do not establish that
control tracks speed better than heading: wheel-speed error and heading error are
different quantities with different units. Such a ranking would require agreed
speed/heading tolerances and measurements such as time within tolerance or settling
time. The earlier experiment log's description of good speed tracking was qualitative
and predates calibrated retraining.

## Verification

- Seven focused regression tests pass, covering normalization/checkpoint pairing, legacy handling, video/comparison CLI arguments, reference simulator isolation, cleanup on training errors, and protection of existing models.
- The reference integration check runs 32 disposable PPO steps on the calibrated MuJoCo world, triggers two real evaluations, and verifies that evaluation leaves training time, positions, velocities, and actuator controls unchanged. Best and final checkpoints reload with the expected metadata, and logs and training video are saved.
- The saved reference test checkpoint also passed a disposable 0.1-second run through the video evaluation CLI, producing both MP4 and JSON with the calibrated XML.
- Audited all 14 final-run checkpoint archives, both TensorBoard logs through 219,136 steps, and best-checkpoint selection at each run's maximum recorded training-evaluation reward. Reference's saved evaluation archive agrees with TensorBoard and contains ten evaluations of five episodes each.
- Loaded both production best checkpoints on CPU and verified finite parameters and predictions on a calibrated reset. All reference checkpoints identify the calibrated XML and independent evaluation; all control checkpoints contain matching normalization statistics, frozen during inference.
- Confirmed all four evaluation JSON files agree with STR-118 metadata. Each video contains 3,000 frames at 50 fps, covering 60 seconds. Smooth-schedule evaluations use the same verified best checkpoint hashes as the hard-schedule evaluations. The original reference model and completed control artifacts retain their verified hashes.
- Earlier disposable checks exercised the control training CLI, individual video evaluation, and a two-second spawned comparison with video stitching. These fixtures are not STR-118 performance results.

The completed artifact audit is recorded in [AUDIT.json](AUDIT.json). Verification inspected saved artifacts and performed reset/inference checks; it did not repeat production training or full evaluation. Source hashes describe files at audit time, not launch snapshots.

## Limitations and follow-up

1. **The hard schedule exceeds calibrated motor command limits.** It requests speeds up to +/-60 rad/s, while the calibrated velocity-actuator command range is +/-20.6 rad/s. The existing training target schedule also reaches this region. The smooth schedule uses speed targets from -10 to 20 rad/s, within that command range, but simultaneous turning and transients still require physical validation. These results do not identify the cause of every fall or establish hardware readiness.
2. **One training run and one evaluation seed per policy are insufficient for confidence intervals or general performance claims.** Training seeds are unset; equal budgets do not imply identical randomness or identical learning conditions.
3. **Checkpoint choice matters.** Best models are selected by separate reward definitions and come from different steps. All four recorded evaluations use best checkpoints; final checkpoints are available but were not evaluated in these recordings.
4. **The existing evaluator protocol is preserved.** For a scheduled evaluation, the action is predicted from the preceding observation before the next target is assigned. This timing convention and the differing yaw definitions should be reviewed before making stronger tracking claims.
5. **This is an SB3 simulation workflow.** ONNX/TFLite export and physical-hardware inference have not been updated or validated to carry the control normalization state. A model ZIP that evaluates correctly here is not by itself a validated hardware-deployment package.

These points do not invalidate the recorded files as results of the implemented procedure. They limit what can be concluded from them and should remain visible to reviewers.

## Artifacts and review scope

The proposed review scope includes the training/evaluation fixes, shared normalization helper, regression tests, documentation, both calibrated best checkpoints, and final STR-118 metadata/results and audit. The control TensorBoard log and reward-weight record are available under the ticket.

Existing ignore rules keep reference training logs, periodic/final checkpoints, and videos local. These ignored files are not automatically included in the PR. Unrelated pendulum/self-balancer validation work is outside STR-118.

## Commands

The commands below document the completed runs. No additional training or evaluation is required for these recorded results; populated model folders and existing evaluation files are protected against accidental overwrites.

Control training, from `src/machine_learning/`:

```powershell
uv run python training/pre_training/control_agent/sb_rl.py -a PPO -out STR-118 train -e ControlAgentTrainingEnv -t 219136
```

This command describes the completed run. Its populated output directory is now protected: archive the existing run before intentionally starting another.

Reference training, from `src/machine_learning/`:

```powershell
uv run python training/reference_baseline/reference_sb_rl.py -a PPO train -e ReferenceBaselineTrainingEnv -t 219136
```

Individual evaluations, from `src/machine_learning/`:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.mp4
uv run python evaluation/evaluate.py video -out STR-118 -m experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip -e ControlAgentTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json
```

Smooth-schedule evaluations, from the same directory:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s smooth_tracking_schedule.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/smooth_tracking_schedule.mp4
uv run python evaluation/evaluate.py video -out STR-118 -m experiments/STR-118/training/models/ControlAgentTrainingEnv_PPO/best_model.zip -e ControlAgentTrainingEnv -x base_world_calibrated.xml -s smooth_tracking_schedule.json --seed 0 -o experiments/STR-118/evaluation/video/ControlAgentTrainingEnv/base_world_calibrated/smooth_tracking_schedule.mp4
```

The recorded outputs are preserved. A deliberate rerun should use a new `-o` destination instead of replacing this evidence. The control README also documents the optional side-by-side comparison command; no full STR-118 comparison video is claimed by this PR description.
