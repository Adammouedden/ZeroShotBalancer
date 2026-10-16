# Experiments

One folder per Jira ticket, holding **every** artifact that ticket produced — training and
evaluation side by side. The experiment log in
[pre_training/README.md](../training/pre_training/README.md) links one folder per entry.

```
experiments/<TICKET>/
├── metadata.json                      what produced each run in this folder
├── training/
│   ├── logs/                          tensorboard runs (+ reward_weights.json)
│   ├── models/<EnvId>_<ALGO>/         best_model.zip and checkpoints
│   └── movies/                        RecordVideo clips captured during training
└── evaluation/
    ├── comparison/<schedule>/         compare.py: both videos, side_by_side, results.json
    ├── video/<schedule>.mp4           evaluate.py video
    └── robustness.json / .mp4         evaluate.py robustness / blocks
```

The ticket is `-out` / `--output_path`. It is **required** when training, since that is what
produces the artifacts an experiment is judged on. On the evaluation scripts it is optional and
defaults to `_unmapped/` (with a warning), because an evaluation is often ad-hoc - two arbitrary
checkpoints against each other - and training and evaluation are separate commands, so passing
the same ticket to both is what keeps a comparison tied to the run it judges.

```bash
# from training/pre_training/control_agent/
uv run python sb_rl.py -a PPO -out STR-106 train -e ControlAgentTrainingEnv -t 219136

# from evaluation/
uv run compare.py -out STR-106
uv run evaluate.py video -out STR-106 -e ControlAgentTrainingEnv -m <model.zip>
```

Why required rather than defaulted: the ticket is the experiment's identity in the log. Runs
used to write to fixed `logs/` / `models/` / `movies/` folders and auto-numbered
`comparison_testing/test-NN/` directories, so each run overwrote the last and had to be
matched back to a ticket by hand afterwards.

Evaluation runs are named after the schedule they ran rather than numbered, so one ticket can
hold an easy and a hard evaluation without collision. Re-running the same ticket and schedule
refuses to overwrite the recorded result unless you pass `--force`.

## What is committed

Not every artifact is tracked in git, since a long run writes a lot of large files:

| Artifact | Tracked? |
| --- | --- |
| `training/logs/` (tensorboard + reward_weights.json) | yes |
| `training/models/**/best_model.zip` | yes |
| `training/models/**/*_cp__*_steps.zip` (checkpoints) | no - local only |
| `metadata.json`, `evaluation/**/*.json` | yes |
| **all `*.mp4`** (training movies and evaluation videos) | **no - local only** |

Videos are not tracked at all: they were 286 MB of this repo's 288 MB, and one `compare.py` run
writes ~73 MB (both clips plus the side-by-side). They are still written locally as normal -
**post anything worth keeping to the Jira ticket for the experiment**, which is also where it is
easiest for the team to watch and comment on it.

Checkpoints are skipped for the same reason, at a smaller scale: ~440 KB per run, and they are
reproducible by re-running the ticket.

Shared layout code lives in
[`visibility/experiment_tracking.py`](../training/visibility/experiment_tracking.py); the
training and evaluation scripts both import it so they can't drift apart.

| Ticket | Training log | Model archive | Evaluation |
| --- | --- | --- | --- |
| STR-96 | `PPO_2` | not archived | `test-01` (mixed_schedule) |
| STR-97 | `PPO_3` | `attempt3` | `test-02` (mixed_schedule_hard) |
| STR-98 | `PPO_4` | `attempt4` | `test-03` (mixed_schedule_hard) |
| STR-99 | `PPO_5` | `attempt5` | `test-04`, `test-04-easy`, `test-04-smooth` |

### `_unmapped/`

Where artifacts go when they are not attributed to an experiment - either an evaluation run
without `-out`, or these historical runs whose ticket could not be established from the record:

- `smoke-test/` — training log `PPO_1`, a 13-event 20-second run the log calls a smoke test

