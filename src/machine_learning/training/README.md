# This directory contains all code related to training a policy for the Zero-shot Balancer agent

[Reference Baseline](reference_baseline/) is a policy trained using code from https://github.com/lachlanhurst/balance-robot-mujoco-rl?tab=MIT-1-ov-file repository that will be used for initial A\B testing against our custom made pre-trained baseline

[Control Agent](pre_training/control_agent/) is a policy trained using our own reward function, but still leveraging the MJCF files and training code (sb_rl.py) from the reference repository.

## Experiments

Every training and evaluation run writes its artifacts to [`experiments/<TICKET>/`](../experiments/),
one folder per Jira ticket holding that experiment's training logs, models, movies and
evaluation results. The ticket is a required argument (`-out`/`--output_path`) on `sb_rl.py`,
`evaluate.py` and `compare.py`.

The hypothesis, change and results for each ticket are written up in the
[pre-training experiment log](pre_training/README.md).

| Experiment | Change | Artifacts |
| --- | --- | --- |
| STR-96 | Use actual yaw to calculate direction error | [experiments/STR-96](../experiments/STR-96) |
| STR-97 | Reduce the lean reward weight (10 to 5) | [experiments/STR-97](../experiments/STR-97) |
| STR-98 | Remove the wheel-speed difference penalty | [experiments/STR-98](../experiments/STR-98) |
| STR-99 | Include heading error in the observations | [experiments/STR-99](../experiments/STR-99) |
| STR-103 | Use a cosine-based heading-error penalty | not yet implemented |
| STR-105 | Normalize observation values | no run recorded |

Artifacts that could not be matched to a ticket are parked in
[experiments/_unmapped](../experiments/_unmapped). See
[experiments/README.md](../experiments/README.md) for the folder layout and how the
pre-existing artifacts were attributed.

Code implementing each experiment is tagged with its ticket in a comment, so `grep -rn "STR-"`
across `pre_training/` and `robot_environments/` shows where each change lives.