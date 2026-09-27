# Control Agent

This directory contains all code for creating our pre-trained policy, dubbed Control Agent as it will serve as a control variable for future testing. It takes the mantle of the "reference-baseline" agent as soon as it surpasses it in balance survival time, adherence to target velocity, and mean reward.

---

## Training:
From the control_agent directory:
```uv run python sb_rl.py -a PPO train -e ControlAgentTrainingEnv -t <timesteps>```
Track with:
```tensorboard --logdir logs/```