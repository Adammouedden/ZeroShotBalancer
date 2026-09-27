# Evaluation

This directory contains all code for evaluating the quality of RL trained policies that control the Zero-Shot Balancer.

Currently, we only have a very basic system of evaluations. One easy and solved evaluation which involves only linear movement
Another more difficult evaluation which is twice as long and contains more varied changes in speed and yaw.

Curiously, because we always test on the same time interval, speeds, and yaws, the reference-baseline model performs deterministically (starting from test-03).

### Developer Log:

Test-01: only mixed_schedule.json existed

Test-02: mixed_schedule_hard.json was created

Test-03: 
* evaluate.py was extended to give a 1 second grace period after every fall, since the agents would begin with a non-zero speed which caused rapid bursts of falls when the grace period was only 15 ms 
* compare.py was created, which gives a parallelized A\B test of any 2 models (in SB3 zip file format) 

The video evaluation is a test using a default MJCF environment file, with standard physical parameters. The test is a measure of falls for the duration of the evaluation as the agent must match target speeds and target yaws as they change dynamically. They change based on an input json schedule, one such example is mixed_schedule.json.

The format for this json is:

```json
(start time, target speed, target yaw)
```

Example:
Your control agent command, run from machine_learning/:
```bash
uv run evaluation/evaluate.py video -e ControlAgentTrainingEnv \
  -m training/pre_training/control_agent/models/ControlAgentTrainingEnv_PPO/best_model.zip \
  -o training/pre_training/control_agent/movies/attempt2/eval_mixed.mp4
```

## Comparing two models (`compare.py`)

`compare.py` records two models on the same schedule in parallel (one process per model), then saves both videos, a side-by-side video and a table of metrics. By default it compares the reference baseline against the control agent (both `best_model.zip`) on `mixed_schedule.json`.

Run from this folder (`evaluation/`):

```bash
# Default: reference_baseline vs control_agent best_model.zip, mixed_schedule.json
uv run compare.py

# Custom schedule (a bare file name is looked up in this folder)
uv run compare.py -s mixed_schedule_hard.json

# Any two models, e.g. an early checkpoint vs the best model
uv run compare.py \
  -a ../training/pre_training/control_agent/models/ControlAgentTrainingEnv_PPO/ControlAgentTrainingEnv_PPO_cp__40000_steps.zip \
  -b ../training/pre_training/control_agent/models/ControlAgentTrainingEnv_PPO/best_model.zip
```

| Option | Default | Description |
| --- | --- | --- |
| `-a`, `--model-a` | reference_baseline `best_model.zip` | First model `.zip` |
| `-b`, `--model-b` | control_agent `best_model.zip` | Second model `.zip` |
| `--env-a`, `--env-b` | inferred | Env to run each model in: `ReferenceBaselineTrainingEnv` or `ControlAgentTrainingEnv` |
| `-s`, `--schedule` | `mixed_schedule.json` | Schedule JSON |
| `--seed` | `0` | Seed for the env reset |

Each model runs in the env it was trained in. The env is read from the model's folder name (`models/<EnvId>_PPO/...`, which is where `sb_rl.py` saves models). If the folder name doesn't show the env, pass `--env-a` / `--env-b`.

The clip length is the last schedule entry's start time + 2 s (30 s for `mixed_schedule.json`).

### Output

Each run creates the next numbered folder, `comparison_testing/test-01/`, `test-02/`, and so on:

```
comparison_testing/test-NN/
├── model_a.mp4      # video of model A, with a speed / yaw / pitch readout
├── model_b.mp4
├── side_by_side.mp4 # model A (left) and model B (right) in sync, labelled at the bottom
├── schedule.json    # copy of the schedule used
└── results.json     # model path, env and metrics for each model
```

It also prints a table to the terminal:

```
                           model_a     model_b      better
  falls                          0           0         tie
  mean |speed error|          4.26        5.11     model_a
  mean |yaw error|            0.21        2.79        n/a*
  mean |pitch| (deg)          3.48        3.60     model_a
  max step reward            2.576       1.090        n/a*
  total reward               851.7       256.9        n/a*
```

| Metric | Better | Meaning |
| --- | --- | --- |
| `falls` | lower | Times the robot fell (pitch > 50°) and was reset |
| `mean_speed_error` | lower | Mean \|target wheel speed − wheel speed\| |
| `mean_yaw_error` | lower | Mean yaw tracking error: heading error in radians in `ControlAgentTrainingEnv`, wheel-yaw difference in `ReferenceBaselineTrainingEnv` |
| `mean_abs_pitch_deg` | lower | Mean \|pitch\| in degrees |
| `max_reward` | higher | Highest single-step reward |
| `total_reward` | higher | Reward summed over the whole clip |

When the two models run in **different envs**, the yaw error and reward rows show `n/a*` instead of a winner. Yaw error is measured differently in each env, and each env scores with its own reward function, so those numbers can't be compared directly. When both models share an env, every row is compared.
