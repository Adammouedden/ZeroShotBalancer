"""
Shared artifact layout for experiment tracking (STR-91).

Every training and evaluation run belongs to a Jira ticket, and writes its artifacts under
`experiments/<TICKET>/`. The ticket is a required argument on the training and evaluation
entry points rather than something defaulted or auto-numbered, because the ticket is the
experiment's identity in the log (`training/pre_training/README.md`).

Before this, each run wrote to fixed `logs/`, `models/` and `movies/` folders and every
evaluation run took the next free `comparison_testing/test-NN/`, so a run's artifacts were
overwritten by the next run and had to be matched back to a ticket by hand. A trained model
was lost that way.

    experiments/<TICKET>/
    |-- metadata.json                     what produced each run in this folder
    |-- training/
    |   |-- logs/                         tensorboard runs (+ reward_weights.json)
    |   |-- models/<EnvId>_<ALGO>/        best_model.zip and checkpoints
    |   `-- movies/                       RecordVideo clips captured during training
    |                                     (local only, never tracked in git)
    `-- evaluation/
        |-- comparison/<schedule>/        compare.py: both videos, side_by_side, results.json
        |-- video/<schedule>.mp4          evaluate.py video
        `-- robustness.json               evaluate.py robustness

Evaluation runs are named after the schedule they ran rather than numbered, so the same
ticket can hold an easy and a hard evaluation without them colliding.
"""
import click
import json
import re

from datetime import datetime
from pathlib import Path

# this module lives in machine_learning/training/visibility/
MACHINE_LEARNING_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = MACHINE_LEARNING_DIR / "experiments"

# Jira issue key, eg; STR-106. Kept deliberately loose on the project key so this still works
# if the team ever files experiments under a second project.
TICKET_PATTERN = r"[A-Z][A-Z0-9]*-\d+"

# Holding pen for artifacts that are not attributed to an experiment: the historical runs whose
# ticket could not be established, and ad-hoc evaluation runs that are not part of an experiment.
# Training always names a ticket; evaluation defaults here (see TICKET_OPTION_HELP_OPTIONAL).
UNMAPPED_TICKET = "_unmapped"


class JiraTicket(click.ParamType):
    """A Jira issue key (eg; STR-106), used verbatim as the experiment folder name."""
    name = "ticket"

    def convert(self, value, param, ctx):
        ticket = str(value).strip()

        # the sentinel is a real folder name, not a ticket, so it skips the pattern check
        if ticket == UNMAPPED_TICKET:
            return ticket

        ticket = ticket.upper()
        if not re.fullmatch(TICKET_PATTERN, ticket):
            self.fail(
                f"{value!r} is not a Jira ticket id (expected something like STR-106)",
                param,
                ctx
            )
        return ticket


TICKET_OPTION_HELP = (
    "Jira ticket this run belongs to (eg; STR-106). Every artifact for the run is written to "
    "experiments/<TICKET>/, and the ticket is the experiment's identity in the log, so it is "
    "required rather than defaulted."
)

TICKET_OPTION_HELP_OPTIONAL = (
    "Jira ticket this evaluation belongs to (eg; STR-106), so its results land beside that "
    f"ticket's training artifacts. Defaults to {UNMAPPED_TICKET}/ for ad-hoc runs that are not "
    "part of an experiment - move them into a ticket folder if they turn out to matter."
)


def schedule_stem(schedule_name: str) -> str:
    """
    Folder / file name for a schedule: `mixed_schedule_hard.json` -> `mixed_schedule_hard`.
    """
    return Path(schedule_name).stem


class Experiment:
    """Where one ticket's artifacts live. See the module docstring for the layout."""

    def __init__(self, ticket: str):
        self.ticket = ticket
        self.root = EXPERIMENTS_DIR / ticket

        training = self.root / "training"
        self.logs = training / "logs"
        self.models = training / "models"
        self.movies = training / "movies"

        self.evaluation = self.root / "evaluation"

    def __str__(self) -> str:
        return self.ticket

    @property
    def is_unmapped(self) -> bool:
        """True when this run is not attributed to an experiment."""
        return self.ticket == UNMAPPED_TICKET

    def warn_if_unmapped(self):
        """
        Say so on stderr. Training and evaluation are separate commands, so the only thing
        keeping a comparison tied to the run it judges is the ticket being passed to both.
        """
        if self.is_unmapped:
            click.secho(
                f"warning: no ticket given, writing to experiments/{UNMAPPED_TICKET}/ - these "
                "results are not linked to any experiment. Pass -out <TICKET> to file them "
                "with that ticket's training artifacts.",
                fg="yellow",
                err=True
            )

    # -- training ------------------------------------------------------------------------
    def model_dir(self, environment: str, algorithm_name: str) -> Path:
        # compare.py infers the env from this folder name, so keep the <EnvId>_<ALGO> form
        return self.models / f"{environment}_{algorithm_name}"

    def make_training_dirs(self):
        for path in (self.logs, self.models, self.movies):
            path.mkdir(parents=True, exist_ok=True)

    def best_model(self) -> Path:
        """
        The model this ticket trained, so evaluating an experiment needs only its ticket.

        There is normally one <EnvId>_<ALGO> folder per ticket; if a ticket trained in more
        than one, the caller has to say which model it means.
        """
        found = sorted(self.models.glob("*/best_model.zip"))

        if not found:
            if self.is_unmapped:
                raise click.UsageError(
                    "no model to evaluate: pass -b/--model-b explicitly, or -out <TICKET> to "
                    "evaluate the model that ticket trained."
                )
            raise click.UsageError(
                f"no best_model.zip under {self.models} - train {self.ticket} first, "
                "or name a model explicitly."
            )
        if len(found) > 1:
            names = ", ".join(str(p.relative_to(self.models)) for p in found)
            raise click.UsageError(
                f"{self.ticket} has several trained models ({names}) - name one explicitly."
            )

        return found[0]

    # -- evaluation ----------------------------------------------------------------------
    def comparison_dir(self, schedule_name: str) -> Path:
        """Folder for one compare.py run, named after the schedule it ran."""
        return self.evaluation / "comparison" / schedule_stem(schedule_name)

    def video_path(self, schedule_name: str) -> Path:
        """Clip for one evaluate.py video run, named after the schedule it ran."""
        return self.evaluation / "video" / f"{schedule_stem(schedule_name)}.mp4"

    def robustness_path(self, suffix: str = ".json") -> Path:
        return self.evaluation / f"robustness{suffix}"

    # -- provenance ----------------------------------------------------------------------
    def write_metadata(self, **fields):
        """
        Append this run to metadata.json, so an artifact folder describes what produced it
        without having to cross-reference the experiment log.
        """
        path = self.root / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = json.loads(path.read_text()) if path.is_file() else {}
        metadata["ticket"] = self.ticket
        metadata.setdefault("runs", []).append({
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            **fields
        })

        path.write_text(json.dumps(metadata, indent=4) + "\n")


def claim_output_dir(path: Path, force: bool, flag: str = "--force") -> Path:
    """
    Create `path` for a run's artifacts, refusing to overwrite a run that is already there.

    Re-running the same ticket and schedule is usually a mistake rather than an intent to
    discard the recorded result, which is the failure mode the ticket folders exist to stop.
    """
    if path.exists() and any(path.iterdir()) and not force:
        raise click.UsageError(
            f"{path} already holds artifacts for this experiment. "
            f"Pass {flag} to overwrite it, or move the existing run aside."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
