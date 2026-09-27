"""
Record two models on the same schedule and compare them. Defaults to the reference baseline vs the control agent.

  uv run compare.py                              # mixed_schedule.json -> comparison_testing/test-NN/
  uv run compare.py -s my_schedule.json          # custom schedule (looked up in this folder if not found)
  uv run compare.py -a path/a.zip -b path/b.zip  # any two models

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

from evaluate import EVAL_DIR, TRAINING_DIR, VIDEO_ENVS, VIDEO_FPS, load_schedule, record_video, schedule_path

OUTPUT_ROOT = EVAL_DIR / "comparison_testing"

DEFAULT_MODEL_A = TRAINING_DIR / "reference_baseline/models/ReferenceBaselineTrainingEnv_PPO/best_model.zip"
DEFAULT_MODEL_B = TRAINING_DIR / "pre_training/control_agent/models/ControlAgentTrainingEnv_PPO/best_model.zip"

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


def next_test_dir() -> Path:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    numbers = [int(p.name.split("-")[1]) for p in OUTPUT_ROOT.glob("test-*") if p.name.split("-")[1].isdigit()]
    test_dir = OUTPUT_ROOT / f"test-{max(numbers, default=0) + 1:02d}"
    test_dir.mkdir()
    return test_dir


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
@click.option('-a', '--model-a', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_MODEL_A, help="first model .zip (default: reference_baseline best_model.zip)")
@click.option('-b', '--model-b', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_MODEL_B, help="second model .zip (default: control_agent best_model.zip)")
@click.option('--env-a', type=click.Choice(VIDEO_ENVS), default=None,
              help="env for model A (default: inferred from its models/<EnvId>_PPO folder)")
@click.option('--env-b', type=click.Choice(VIDEO_ENVS), default=None,
              help="env for model B (default: inferred from its models/<EnvId>_PPO folder)")
@click.option('-s', '--schedule', 'schedule_name', default="mixed_schedule.json",
              help="schedule JSON (looked up in this folder if not found)")
@click.option('--seed', default=0)
def compare(model_a: Path, model_b: Path, env_a: str | None, env_b: str | None, schedule_name: str, seed: int):
    runs = {
        "model_a": (model_a.resolve(), infer_env(model_a, env_a, "--env-a")),
        "model_b": (model_b.resolve(), infer_env(model_b, env_b, "--env-b")),
    }
    schedule = load_schedule(schedule_name)
    test_dir = next_test_dir()
    shutil.copy(schedule_path(schedule_name), test_dir / "schedule.json")

    # one process per model; spawn so each gets its own MuJoCo renderer
    with ProcessPoolExecutor(len(runs), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for name, (model, env_id) in runs.items():
            click.echo(f"Recording {name}: {model} in {env_id} ...")
            futures[name] = pool.submit(record_video, str(model), str(test_dir / f"{name}.mp4"), schedule, env_id,
                                        seed=seed)
        results = {name: {"model": str(runs[name][0]), "env": runs[name][1], **f.result()}
                   for name, f in futures.items()}

    click.echo("Stitching side_by_side.mp4 ...")
    side_by_side(test_dir / "model_a.mp4", test_dir / "model_b.mp4",
                 tuple(f"{n}: {model.parent.name}/{model.stem}" for n, (model, _) in runs.items()),
                 test_dir / "side_by_side.mp4")

    (test_dir / "results.json").write_text(json.dumps(results, indent=4))

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
