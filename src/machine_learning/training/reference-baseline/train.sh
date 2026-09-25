#!/usr/bin/env bash
# Train the reference baseline: stage 1 (Env01-v3) then stage 2 (Env03-v2).
# Usage: ./train.sh [stage1_steps] [stage2_steps]
set -euo pipefail
cd "$(dirname "$0")"

STAGE1_STEPS=${1:-500000}
STAGE2_STEPS=${2:-1000000}

echo "Stage 1: Env01-v3 for $STAGE1_STEPS steps (log: train_env01.log)"
uv run python sb_rl.py -a PPO train -e Env01-v3 -t "$STAGE1_STEPS" > train_env01.log 2>&1

echo "Stage 2: Env03-v2 for $STAGE2_STEPS steps (log: train_env03.log)"
uv run python sb_rl.py -a PPO -m ./models/Env01-v3_PPO/best_model.zip \
    train -e Env03-v2 -t "$STAGE2_STEPS" > train_env03.log 2>&1

echo "Done. Final model: models/Env03-v2_PPO/best_model.zip"
