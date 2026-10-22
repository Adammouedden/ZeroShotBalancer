# Machine Learning

Every run writes its artifacts to [`experiments/<JIRA-TICKET>/`](experiments/README.md), passed
as `-out`. Required when training, defaults to `_unmapped/` when evaluating.

Videos are not tracked in git, so post any movie worth keeping to the Jira ticket it belongs to.

## Train

From `training/pre_training/control_agent/`:

```bash
uv run python sb_rl.py -a PPO -out STR-106 train -e ControlAgentTrainingEnv -t 219136
```

`-a` algorithm, `-out` ticket number, `-e` environment. Drop `-t` to run until stopped.

## Evaluate one model

From `evaluation/`:

```bash
uv run evaluate.py video -out STR-106 -e ControlAgentTrainingEnv \
  -m ../experiments/STR-106/training/models/ControlAgentTrainingEnv_PPO/best_model.zip
```

`-m` model. Writes `experiments/STR-106/evaluation/video/<schedule>.mp4`.

## Compare two models

From `evaluation/`:

```bash
uv run compare.py -out STR-106
```

Reference baseline vs the model STR-106 trained. Add `-s mixed_schedule_hard.json` for the hard
schedule, or `-a <a.zip> -b <b.zip>` for any two models.
