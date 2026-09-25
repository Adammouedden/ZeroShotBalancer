# Reference Baseline

A PPO balancing policy trained with the environments and training script from
[lachlanhurst/balance-robot-mujoco-rl](https://github.com/lachlanhurst/balance-robot-mujoco-rl) (MIT).
We use it as the baseline for A/B testing our own policies.

Only the files needed to train the policy were copied:

- `sb_rl.py`: the training and testing CLI
- `envs/`: the `Env01-v3` and `Env03-v2` MuJoCo environments and the files they depend on

## Changes from upstream

- Training runs on `cuda` (the `DEVICE` constant in `sb_rl.py`).
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

## Replicate

Run these commands from this directory. Models, logs and videos are written to `models/`, `logs/`
and `movies/`, which git ignores.

```bash
# 1. Train balancing from scratch
uv run python sb_rl.py -a PPO train -e Env01-v3 -t 3000000

# 2. Fine-tune on Env03-v2 for robustness
#    (the upstream README says Env01-v2_PPO here, which is a typo)
uv run python sb_rl.py -a PPO -m ./models/Env01-v3_PPO/best_model.zip train -e Env03-v2 -t 2000000

# 3. Watch the result in the MuJoCo viewer (needs a display; WSLg works)
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.zip test -e Env03-v2
```

Or run both training stages back to back with `./train.sh [stage1_steps] [stage2_steps]`. It
defaults to 500k and 1M steps. Stage 1 usually balances reliably within about 200k steps.

Track progress with `uv run tensorboard --logdir logs`. The policy balances reliably once
`ep_len_mean` levels off near the episode limit (6000 for Env01-v3, 1200 for Env03-v2). You can
stop training at that point: `best_model.zip` is updated at every evaluation (every 20k steps).

On an RTX 4060, training runs at about 200 steps/s, so stage 1 takes about 4 hours and stage 2
about 3 hours. PPO with a small MLP is often faster on CPU. To try it, set `DEVICE = "cpu"`.

## Evaluate

`evaluate.py` runs without a display:

```bash
# Survival rate on Env03-v2 with blocks from the front AND back (-m can be repeated to compare models)
uv run python evaluate.py robustness -m models/Env03-v2_PPO/best_model.zip -n 50

# Replay the same test as a video (10x speed; use -n 5 --speed 1 for real time)
uv run python evaluate.py blocks -m models/Env03-v2_PPO/best_model.zip -o movies/robustness.mp4

# 30 s video in Env01-v3 with a readout on screen; resets whenever the robot falls
#   --schedule mixed: custom speed + yaw changes (default)
#   --schedule env01: Env01-v3's own speed-only schedule
uv run python evaluate.py video -m models/Env03-v2_PPO/best_model.zip -o movies/balance_eval.mp4
```

Env03-v2 chooses which side blocks come from once, when the environment is created (`envs/env03_v2.py`).
So a single training run only ever sees blocks from one side, and `robustness` tests both.

## Export

To export a model to ONNX and test it:

```bash
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.zip convert -e Env03-v2
uv run python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.onnx test-onnx -e Env03-v2
```
