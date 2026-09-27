# This directory contains all code related to training a policy for the Zero-shot Balancer agent

[Reference Baseline](reference_baseline/) is a policy trained using code from https://github.com/lachlanhurst/balance-robot-mujoco-rl?tab=MIT-1-ov-file repository that will be used for initial A\B testing against our custom made pre-trained baseline

[Control Agent](pre_training/control_agent/) is a policy trained using our own reward function, but still leveraging the MJCF files and training code (sb_rl.py) from the reference repository.