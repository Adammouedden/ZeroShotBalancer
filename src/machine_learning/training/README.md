# This directory contains all code related to training a policy for the Zero-shot Balancer agent

[Reference Baseline](reference_baseline/) is a policy trained using code from the [external reference repository](https://github.com/lachlanhurst/balance-robot-mujoco-rl). It provides an external point of comparison for our policies.

[Control Agent](pre_training/control_agent/) uses our team's reward and observation changes, building on the reference repository's training code and MJCF models. Control is the baseline for comparing subsequent experiments.

## STR-118 reference training scope

Both policies train and evaluate using independent simulator instances of
`base_world_calibrated.xml`. This keeps periodic evaluation from resetting or
advancing the simulator supplying the training observations. Reference retains its
existing observations, reward, policy architecture, and PPO settings; it does not
receive control's observation normalization.

Reference uses CPU and a 219,136-step budget. Its new models, logs, and training
videos go under `training/reference_baseline/{models,logs,movies}/calibrated_run2/`.
The script refuses to overwrite a populated model directory and saves a final
checkpoint as well as evaluation-selected best and periodic checkpoints. See the
[reference training instructions](reference_baseline/README.md#calibrated-training-str-118)
for training, TensorBoard, and evaluation commands. Both policies completed 219,136
steps and a 60-second calibrated hard-schedule evaluation: reference recorded 9
falls and control recorded 25. Control remains the baseline for subsequent
experiments. See the [STR-118 results and implementation summary](../experiments/STR-118/PR_DRAFT.md#results)
for the final comparison and its limits.

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
