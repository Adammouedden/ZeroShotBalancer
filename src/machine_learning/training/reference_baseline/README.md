# Reference Baseline

A PPO balancing policy trained with the environments and training script from
[lachlanhurst/balance-robot-mujoco-rl](https://github.com/lachlanhurst/balance-robot-mujoco-rl) (MIT).
It provides an external comparison; our control policy is the baseline for subsequent experiments.

Only the files needed to train the policy were copied:

- `reference_sb_rl.py`: the training and testing CLI
- `envs/`: the `ReferenceBaselineTrainingEnv` and `Env03-v2` MuJoCo environments and the files they depend on

## Calibrated training (STR-118)

**Completed:** 219,136 training steps, with the best checkpoint selected at step
20,000 and a final checkpoint at step 219,136. The selected best model's 60-second
hard-schedule evaluation (seed 0) recorded 9 falls, mean absolute wheel-speed error
24.861 rad/s, and mean absolute pitch 2.333 degrees. The same best checkpoint's
smooth-schedule evaluation (60 seconds, seed 0) recorded 2 falls, speed error
10.730 rad/s, and pitch 1.218 degrees. These are simulation results; reference did
not achieve zero falls on either schedule.

The commands below document the completed run. No restart is needed; the existing
model directory and evaluation files are protected against overwriting.

Run these commands from `src/machine_learning/`. Both the training simulator and
its independent evaluation simulator load `base_world_calibrated.xml`. Evaluation
does not reset or advance the training simulator. Reference's observations, reward,
PPO settings, and raw-observation preprocessing are retained.

Start a fresh CPU run with the same 219,136-step budget (omit `-m`):

```powershell
uv run python training/reference_baseline/reference_sb_rl.py -a PPO train -e ReferenceBaselineTrainingEnv -t 219136
```

The script prints both resolved XML paths. Outputs are resolved relative to this
directory, regardless of where the command is launched:

- Models: `models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/`
- TensorBoard: `logs/calibrated_run2/`
- Training videos: `movies/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/`

A populated model directory is rejected before training begins. `best_model.zip`
is selected by evaluation every 20,000 steps; periodic checkpoints are saved every
40,000 steps. `final_model.zip` is saved when learning returns normally, including
an early stop. These checkpoints record the training XML and independent-evaluation
setting. Both simulators are closed when training finishes or raises an error.

In a second terminal, also from `src/machine_learning/`:

```powershell
uv run tensorboard --logdir training/reference_baseline/logs/calibrated_run2 --port 6008
```

Open http://localhost:6008. After training completes, evaluate the selected best
checkpoint using the calibrated world and the same 60-second hard schedule:

```powershell
uv run python evaluation/evaluate.py video -out STR-118 -m training/reference_baseline/models/calibrated_run2/ReferenceBaselineTrainingEnv_PPO/best_model.zip -e ReferenceBaselineTrainingEnv -x base_world_calibrated.xml -s mixed_schedule_hard.json --seed 0 -o experiments/STR-118/evaluation/video/ReferenceBaselineTrainingEnv/base_world_calibrated/calibrated_run2/mixed_schedule_hard.mp4
```

The completed video and matching JSON metrics are in the explicit `calibrated_run2`
evaluation directory, with matching results in STR-118 metadata. Repeating an
evaluation requires a different output path or an intentional `--force` overwrite.
See the [final artifact audit](../../experiments/STR-118/AUDIT.json) for verification.

The completed smooth evaluation uses the same model and settings with
`-s smooth_tracking_schedule.json` and output filename `smooth_tracking_schedule.mp4`
in that same evaluation directory. Both command lines and the reference/control
comparison are recorded in the [STR-118 results](../../experiments/STR-118/PR_DRAFT.md#results).

## Earlier adaptation notes

- Earlier training used `cuda`; STR-118 uses `DEVICE = "cpu"` in `reference_sb_rl.py`.
- `train` has an optional `-t/--timesteps` cap. The rewards never reach the 6000 stop threshold,
  so without the cap training runs until you stop it (same as upstream).
- The environments are imported from the local `envs` package. Upstream uses an installed
  `balance_robot` package.
- `tensorflow`, `onnx` and `serial` are imported only inside the commands that use them.

## Setup

From `src/machine_learning/`, run:

```bash
uv sync                  # Python 3.13, torch (CUDA), SB3, MuJoCo, TensorBoard, ONNX
uv sync --extra tflite   # optional: TensorFlow, only needed for the test-tflite* commands
```

TensorFlow is an optional extra because loading it before PyTorch's `triton` causes a segfault.
When TensorFlow is installed, TensorBoard loads it during training, which triggers that crash.
Don't train with the `tflite` extra installed.

## Historical replication workflow

The following records the earlier two-stage workflow and its original output
layout. Use the calibrated commands above for STR-118; the current training script
selects the calibrated world and `calibrated_run2` output directories.

Run these commands from this directory. Models, logs and videos are written to `models/`, `logs/`
and `movies/`, which git ignores.

```bash
# 1. Train balancing from scratch
uv run python sb_rl.py -a PPO train -e ReferenceBaselineTrainingEnv -t 3000000

# 2. Fine-tune on Env03-v2 for robustness
uv run python sb_rl.py -a PPO -m ./models/ReferenceBaselineTrainingEnv_PPO/best_model.zip train -e Env03-v2 -t 2000000

# 3. Watch the result in the MuJoCo viewer (needs a display; WSLg works)
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.zip test -e Env03-v2
```

Stage 1 usually balances reliably within about 200k steps.

Track progress with `uv run tensorboard --logdir logs`. The policy balances reliably once
`ep_len_mean` levels off near the episode limit (6000 for ReferenceBaselineTrainingEnv, 1200 for Env03-v2). You can
stop training at that point: `best_model.zip` is updated at every evaluation (every 20k steps).

On an RTX 4060, training runs at about 200 steps/s, so stage 1 takes about 4 hours and stage 2
about 3 hours. PPO with a small MLP is often faster on CPU. To try it, set `DEVICE = "cpu"`.

## Evaluate

`evaluate.py` lives in `machine_learning/evaluation/` and runs without a display. The commands below are run from this folder:

```bash
# Survival rate on Env03-v2 with blocks from the front AND back (-m can be repeated to compare models)
uv run ../../evaluation/evaluate.py robustness -m models/Env03-v2_PPO/best_model.zip -n 50

# Replay the same test as a video (10x speed; use -n 5 --speed 1 for real time)
uv run ../../evaluation/evaluate.py blocks -m models/Env03-v2_PPO/best_model.zip -o movies/robustness.mp4

# Video with a readout on screen; resets whenever the robot falls
#   -s/--schedule: JSON of [start s, target speed, target yaw] entries (default: evaluation/mixed_schedule.json),
#                  or env01 for the env's own built-in schedule. Clip length = last entry + 2 s unless -d is given.
#   -e/--env:      ReferenceBaselineTrainingEnv (default) or ControlAgentTrainingEnv
uv run ../../evaluation/evaluate.py video -m models/Env03-v2_PPO/best_model.zip -o movies/balance_eval.mp4
```

Env03-v2 chooses which side blocks come from once, when the environment is created (`envs/reference_baseline_fine_tuning.py`).
So a single training run only ever sees blocks from one side, and `robustness` tests both.

## Export

To export a model to ONNX and test it:

```bash
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.zip convert -e Env03-v2
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.onnx test-onnx -e Env03-v2
```
