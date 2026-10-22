"""
Record two models on the same schedule and compare them. Defaults to the reference baseline vs
the model the given ticket trained.

  uv run compare.py -out STR-106                      # -> experiments/STR-106/evaluation/comparison/mixed_schedule/
  uv run compare.py -out STR-106 -s my_schedule.json  # custom schedule (looked up in this folder if not found)
  uv run compare.py -out STR-106 -a a.zip -b b.zip    # any two models

The Jira ticket (-out/--output_path) files this comparison next to that ticket's training
artifacts, and -b defaults to the model that ticket trained. Without a ticket, results go to
experiments/_unmapped/ and -b has to be named explicitly. Comparisons are named after the
schedule rather than numbered, so one ticket can hold an easy and a hard evaluation.

Both models are recorded in parallel, then stitched into side_by_side.mp4 (A left, B right).
"""
import click
import imageio
import json
import multiprocessing
import numpy as np
import shutil

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw

from evaluate import TRAINING_DIR, VIDEO_ENVS, VIDEO_FPS, load_schedule, record_video, schedule_path
# STR-91: per-ticket artifact layout, replaces the auto-numbered comparison_testing/test-NN/
from visibility.experiment_tracking import (
    Experiment,
    JiraTicket,
    TICKET_OPTION_HELP_OPTIONAL,
    UNMAPPED_TICKET,
    claim_output_dir,
)

DEFAULT_MODEL_A = TRAINING_DIR / "reference_baseline/models/ReferenceBaselineTrainingEnv_PPO/best_model.zip"

# (key, label, format, higher is better, comparable across different envs)
METRICS = [
    ("falls", "falls", "{:d}", False, True),
    ("mean_speed_error", "mean |speed error|", "{:.2f}", False, True),
    # evaluate.yaw_error measures heading (rad) in ControlAgentTrainingEnv but wheel yaw in the reference env
    ("mean_yaw_error", "mean |yaw error|", "{:.2f}", False, False),
    ("mean_abs_pitch_deg", "mean |pitch| (deg)", "{:.2f}", False, True),
    # each env has its own reward function and weights
    ("max_reward", "max step reward", "{:.3f}", True, False),
    ("total_reward", "total reward", "{:.1f}", True, False),
]


def infer_env(model: Path, env_id: str | None, flag: str) -> str:
    """
    sb_rl.py saves models under models/<EnvId>_<ALGO>/, so the env can usually be read from the folder name.
    """
    if env_id:
        return env_id
    for candidate in VIDEO_ENVS:
        if model.parent.name.startswith(f"{candidate}_"):
            return candidate
    raise click.UsageError(f"can't tell which env {model} was trained in, pass {flag} ({' / '.join(VIDEO_ENVS)})")


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, img.height - 34, img.width, img.height], fill=(0, 0, 0))
    draw.text((10, img.height - 28), text, fill=(255, 255, 255), font_size=18)
    return np.asarray(img)


def side_by_side(video_a: Path, video_b: Path, labels: tuple[str, str], output: Path):
    with imageio.get_reader(video_a) as a, imageio.get_reader(video_b) as b, \
            imageio.get_writer(output, fps=VIDEO_FPS) as writer:
        for frame_a, frame_b in zip(a, b):
            writer.append_data(np.hstack([label_frame(frame_a, labels[0]), label_frame(frame_b, labels[1])]))


@click.command()
@click.option(
    '-out',
    '--output_path',
    'ticket',
    default=UNMAPPED_TICKET,
    show_default=True,
    type=JiraTicket(),
    help=TICKET_OPTION_HELP_OPTIONAL
)
@click.option('-a', '--model-a', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_MODEL_A, help="first model .zip (default: reference_baseline best_model.zip)")
@click.option('-b', '--model-b', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="second model .zip (default: the model this ticket trained)")
@click.option('--env-a', type=click.Choice(VIDEO_ENVS), default=None,
              help="env for model A (default: inferred from its models/<EnvId>_PPO folder)")
@click.option('--env-b', type=click.Choice(VIDEO_ENVS), default=None,
              help="env for model B (default: inferred from its models/<EnvId>_PPO folder)")
@click.option('-s', '--schedule', 'schedule_name', default="mixed_schedule.json",
              help="schedule JSON (looked up in this folder if not found)")
# Select the MuJoCo world XML for each model
@click.option('--xml-a', default='base_world_env.xml',help='World XML for original robot')
@click.option('--xml-b', default='base_world_calibrated.xml',help='World XML for calibrated robot')
@click.option('--seed', default=0)
@click.option('--force', is_flag=True, default=False,
              help="overwrite an existing comparison for this ticket and schedule")
def compare(ticket: str, model_a: Path, model_b: Path | None, env_a: str | None, env_b: str | None,
            schedule_name: str, seed: int, force: bool):
    experiment = Experiment(ticket)
    experiment.warn_if_unmapped()
    # STR-91: the ticket's own model is what the experiment is about, so it is model B by default
    model_b = model_b or experiment.best_model()

    runs = {
        "model_a": (model_a.resolve(), infer_env(model_a, env_a, "--env-a")),
        "model_b": (model_b.resolve(), infer_env(model_b, env_b, "--env-b")),
    }
    schedule = load_schedule(schedule_name)
    # STR-91: named after the schedule, so one ticket can hold an easy and a hard evaluation
    test_dir = claim_output_dir(experiment.comparison_dir(schedule_name), force)
    shutil.copy(schedule_path(schedule_name), test_dir / "schedule.json")

    # one process per model; spawn so each gets its own MuJoCo renderer
    with ProcessPoolExecutor(len(runs), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for name, (model, env_id) in runs.items():
            click.echo(f"Recording {name}: {model} in {env_id} ...")
            
            xml_file = xml_a if name == "model_a" else xml_b
            futures[name] = pool.submit(record_video,str(model),str(test_dir / f"{name}.mp4"),
                schedule,env_id,seed=seed,xml_file=xml_file
            )
            
        results = {name: {"model": str(runs[name][0]), "env": runs[name][1], **f.result()}
                   for name, f in futures.items()}

    click.echo("Stitching side_by_side.mp4 ...")
    side_by_side(test_dir / "model_a.mp4", test_dir / "model_b.mp4",
                 tuple(f"{n}: {model.parent.name}/{model.stem}" for n, (model, _) in runs.items()),
                 test_dir / "side_by_side.mp4")

    (test_dir / "results.json").write_text(json.dumps(results, indent=4))
    # STR-91: record this evaluation against the ticket
    experiment.write_metadata(
        command="compare",
        schedule=schedule_name,
        seed=seed,
        output=str(test_dir),
        results=results,
    )

    names = list(runs)
    same_env = runs["model_a"][1] == runs["model_b"][1]
    click.echo(f"\n{test_dir}  ({results['model_a']['duration_s']:.0f} s, schedule={schedule_name})")
    for name in names:
        click.echo(f"  {name}: {results[name]['model']} ({results[name]['env']})")
    click.echo(f"\n  {'':20s}" + "".join(f"{n:>12s}" for n in names) + f"{'better':>12s}")
    for key, label, fmt, higher_better, comparable in METRICS:
        values = [results[n][key] for n in names]
        if not (comparable or same_env):
            better = "n/a*"
        elif values[0] == values[1]:
            better = "tie"
        else:
            better = names[values.index(max(values) if higher_better else min(values))]
        click.echo(f"  {label:20s}" + "".join(f"{fmt.format(v):>12s}" for v in values) + f"{better:>12s}")
    if not same_env:
        click.echo("\n  * not compared across different envs: yaw error is measured differently (heading vs wheel yaw)"
                   "\n    and each env scores reward with its own reward function")


if __name__ == "__main__":
    compare()
