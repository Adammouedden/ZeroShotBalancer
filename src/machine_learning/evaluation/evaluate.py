"""
Headless evaluation of trained balancing policies.

  robustness: run Env03-v2 episodes with blocks fired at the front AND back,
              report survival rate per side (training only ever sees one side).
  video:      record a continuous clip in ReferenceBaselineTrainingEnv (or ControlAgentTrainingEnv
              via --env) following a JSON speed / yaw schedule, resetting the robot whenever it falls.

Every command takes the Jira ticket it is evaluating for (-out/--output_path) and writes its
artifacts to experiments/<TICKET>/evaluation/, alongside that ticket's training artifacts.
Without one, artifacts go to experiments/_unmapped/ - fine for an ad-hoc run, but it breaks
the link between an evaluation and the training run it judges, so it warns.
Pass -o/--output to write somewhere else instead.
"""
import click
import gymnasium as gym
import imageio
import importlib.util
import json
import numpy as np
import stable_baselines3
import sys

from pathlib import Path
from PIL import Image, ImageDraw

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_DIR = EVAL_DIR.parent / "training"

# while not called directly, we need to import this so ReferenceBaselineTrainingEnv / Env03-v2 are registered
sys.path.insert(0, str(TRAINING_DIR / "reference_baseline"))
import envs

from robot_environments.base_world_env import BaseWorldEnv

# STR-91: per-ticket artifact layout, shared with the training scripts so the training and
# evaluation sides agree on where a ticket's artifacts go
from visibility.experiment_tracking import (
    Experiment,
    JiraTicket,
    TICKET_OPTION_HELP_OPTIONAL,
    UNMAPPED_TICKET,
)

# control_agent's env package is also named `envs`, so load its module by path and register it here
_control_env_path = TRAINING_DIR / "pre_training/control_agent/envs/control_agent_training_environment.py"
_spec = importlib.util.spec_from_file_location("control_agent_training_environment", _control_env_path)
_control_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_control_env)
gym.register(
    id="ControlAgentTrainingEnv",
    entry_point=_control_env.ControlAgentTrainingEnv,
    max_episode_steps=6000,
    reward_threshold=6000,
)

VIDEO_ENVS = ["ReferenceBaselineTrainingEnv", "ControlAgentTrainingEnv"]

CONTROL_DT = 0.005  # 250 physics steps x 0.02ms per env step
VIDEO_FPS = 50


def ticket_option(fn):
    """
    -out/--output_path, on every evaluation command (see visibility.experiment_tracking).

    STR-91: evaluation artifacts are filed under the ticket they evaluate, next to that
    ticket's training artifacts. Optional here, unlike sb_rl.py: training is what produces the
    artifacts an experiment is judged on, while an evaluation is often ad-hoc and has
    experiments/_unmapped/ to fall back on.
    """
    return click.option(
        '-out',
        '--output_path',
        'ticket',
        default=UNMAPPED_TICKET,
        show_default=True,
        type=JiraTicket(),
        help=TICKET_OPTION_HELP_OPTIONAL
    )(fn)


@click.group()
def cli():
    pass


@cli.command(help="Survival rate on Env03-v2 with blocks from the front and back")
@ticket_option
@click.option('-m', '--model', 'models', required=True, multiple=True, help="model .zip (repeat to compare)")
@click.option('-n', '--episodes', default=50, help="episodes per model (split evenly between sides)")
@click.option('-o', '--output', default=None, type=click.Path(path_type=Path),
              help="results JSON (default: experiments/<TICKET>/evaluation/robustness.json)")
@click.option('--seed', default=0)
def robustness(ticket: str, models: tuple[str, ...], episodes: int, output: Path | None, seed: int):
    experiment = Experiment(ticket)
    experiment.warn_if_unmapped()
    summary = {}

    for model_path in models:
        env = gym.make("Env03-v2")
        model = stable_baselines3.PPO.load(model_path, device="cpu")
        max_len = env.spec.max_episode_steps
        np.random.seed(seed)  # env uses the global RNG for block placement

        results = {"front": [], "back": []}
        for ep in range(episodes):
            side = "front" if ep % 2 == 0 else "back"
            env.unwrapped.attack_side_front = side == "front"
            obs, _ = env.reset(seed=seed + ep)
            length, ret, done = 0, 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                length += 1
                ret += reward
                done = terminated or truncated
            results[side].append((length, ret))

        click.echo(f"\n{model_path}")
        click.echo(f"  {'side':6s} {'survived':>10s} {'mean len':>10s} {'mean return':>12s}")
        summary[str(model_path)] = {"episodes": episodes, "seed": seed, "max_episode_steps": max_len}
        for side, rows in list(results.items()) + [("all", results["front"] + results["back"])]:
            lengths = np.array([r[0] for r in rows])
            returns = np.array([r[1] for r in rows])
            survived = (lengths >= max_len).mean() * 100
            click.echo(
                f"  {side:6s} {survived:9.0f}% {lengths.mean():7.0f}/{max_len} {returns.mean():12.1f}"
            )
            summary[str(model_path)][side] = {
                "survived_pct": float(survived),
                "mean_length": float(lengths.mean()),
                "mean_return": float(returns.mean()),
            }
        env.close()

    output = output or experiment.robustness_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=4) + "\n")
    click.echo(f"\nWrote {output}")


BUILTIN_SCHEDULE = "env01"
LAST_ENTRY_HOLD = 2.0  # seconds the final schedule entry is held before the clip ends
RESET_GRACE = 1.0  # seconds targets are held at 0 after a fall reset, like the start of a training episode


def schedule_path(name: str) -> Path:
    """
    Relative paths that don't exist from the current directory are looked up in this folder.
    """
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = EVAL_DIR / path
    return path


def load_schedule(name: str) -> list[list[float]]:
    """
    Schedule JSON: a list of [start time s, target wheel speed, target yaw], each held until the next entry.
    """
    path = schedule_path(name)
    schedule = sorted(json.loads(path.read_text()))
    if not schedule or any(len(entry) != 3 for entry in schedule):
        raise click.BadParameter(f"{path} must be a non-empty list of [start, speed, yaw]")
    return schedule


def schedule_target(schedule: list[list[float]], t: float) -> tuple[float, float]:
    speed, yaw = 0.0, 0.0
    for start, s, y in schedule:
        if t >= start:
            speed, yaw = s, y
    return speed, yaw


def yaw_error(robot, env_id: str) -> float:
    if env_id == "ControlAgentTrainingEnv":
        # target_yaw here is a heading (radians) tracked against get_yaw(), not a
        # wheel-speed differential like get_wheel_yaw() - wrap it the same way the
        # env's own direction_error reward term does, or the raw diff is meaningless
        return abs((robot.target_yaw - robot.get_yaw() + np.pi) % (2 * np.pi) - np.pi)
    return abs(robot.target_yaw - robot.get_wheel_yaw())


def yaw_status_line(robot, env_id: str) -> str:
    if env_id == "ControlAgentTrainingEnv":
        heading = np.degrees(robot.get_yaw())
        target_heading = np.degrees((robot.target_yaw + np.pi) % (2 * np.pi) - np.pi)
        return f"heading {heading:6.1f}  target {target_heading:6.1f} deg"
    return f"yaw    {robot.get_wheel_yaw():6.1f}  target {robot.target_yaw:6.1f}"


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 330, 22 * len(lines) + 12], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((10, 8 + 22 * i), line, fill=(255, 255, 255), font_size=18)
    return np.asarray(img)


@cli.command(help="Record a continuous clip following a speed / yaw schedule, resetting whenever the robot falls")
@ticket_option
@click.option('-m', '--model', required=True, help="model .zip")
@click.option('-o', '--output', default=None, type=click.Path(path_type=Path),
              help="output .mp4 (default: experiments/<TICKET>/evaluation/video/<schedule>.mp4)")
@click.option('-d', '--duration', type=float, default=None,
              help=f"clip length in simulated seconds (default: last schedule entry + {LAST_ENTRY_HOLD:.0f} s, "
                   f"or 30 s for {BUILTIN_SCHEDULE})")
@click.option('-s', '--schedule', 'schedule_name', default="mixed_schedule.json",
              help=f"schedule JSON (looked up in this folder if not found), or '{BUILTIN_SCHEDULE}' "
                   "for the env's own built-in schedule")
@click.option('-e', '--env', 'env_id', type=click.Choice(VIDEO_ENVS), default=VIDEO_ENVS[0],
              help="environment to record in (match the one the model was trained on)")
@click.option('--seed', default=0)
@click.option('--reset-grace', default=RESET_GRACE,
              help="seconds targets are held at 0 after a fall reset (0 = resume the schedule immediately)")
@click.option('-x', '--xml', 'xml_file', default="base_world_env.xml",
              help="world XML in training/robot_environments (e.g. base_world_calibrated.xml)")
def video(model: str, output: str, duration: float | None, schedule_name: str, env_id: str, seed: int,
          reset_grace: float, xml_file: str):
    schedule = None if schedule_name == BUILTIN_SCHEDULE else load_schedule(schedule_name)
    results = record_video(model, output, schedule, env_id, duration, seed, reset_grace, xml_file)
    click.echo(f"Wrote {output}: {results['duration_s']:.0f}s, {results['falls']} fall(s), schedule={schedule_name}, "
               f"xml={xml_file}")
@click.option('--force', is_flag=True, default=False, help="overwrite an existing clip for this ticket and schedule")
def video(ticket: str, model: str, output: Path | None, duration: float | None, schedule_name: str,
          env_id: str, seed: int, reset_grace: float, force: bool):
    experiment = Experiment(ticket)
    experiment.warn_if_unmapped()
    schedule = None if schedule_name == BUILTIN_SCHEDULE else load_schedule(schedule_name)

    if output is None:
        output = experiment.video_path(schedule_name)
        if output.exists() and not force:
            raise click.UsageError(
                f"{output} already exists for this experiment. Pass --force to overwrite it, "
                "or -o to write elsewhere."
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    results = record_video(str(model), str(output), schedule, env_id, duration, seed, reset_grace)

    experiment.write_metadata(
        command="evaluate video",
        model=str(model),
        environment=env_id,
        schedule=schedule_name,
        seed=seed,
        output=str(output),
        results=results,
    )
    click.echo(f"Wrote {output}: {results['duration_s']:.0f}s, {results['falls']} fall(s), schedule={schedule_name}")


def record_video(model: str, output: str, schedule: list[list[float]] | None, env_id: str,
                duration: float | None = None, seed: int = 0, reset_grace: float = RESET_GRACE,
                xml_file: str = "base_world_env.xml") -> dict:
    """
    Record a clip following `schedule` (None = the env's built-in schedule) and return tracking metrics.
    """
    if duration is None:
        duration = 30.0 if schedule is None else schedule[-1][0] + LAST_ENTRY_HOLD
    # Which world to load
    env = gym.make(env_id, render_mode="rgb_array", xml_file=xml_file)
    robot = env.unwrapped
    policy = stable_baselines3.PPO.load(model, device="cpu")
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    frame_every = round(1 / (VIDEO_FPS * CONTROL_DT))
    total_steps = round(duration / CONTROL_DT)
    falls = 0
    last_reset = -reset_grace  # no grace period at the start of the clip
    speed_errors, yaw_errors, pitches, rewards = [], [], [], []

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=VIDEO_FPS) as writer:
        for step in range(total_steps):
            t = step * CONTROL_DT
            action, _ = policy.predict(obs, deterministic=True)
            if schedule is not None:
                # bypass the env's built-in schedule in step(), drive the targets ourselves
                in_grace = t - last_reset < reset_grace
                robot.target_wheel_speed, robot.target_yaw = (0, 0) if in_grace else schedule_target(schedule, t)
                obs, reward, terminated, _, _ = BaseWorldEnv.step(robot, action)
            else:
                # ReferenceBaselineTrainingEnv schedule runs on episode time; ignore the 6000-step truncation
                obs, reward, terminated, _, _ = env.step(action)

            rewards.append(float(reward))
            speed_errors.append(abs(robot.target_wheel_speed - robot.get_wheel_speed()))
            yaw_errors.append(yaw_error(robot, env_id))
            pitches.append(abs(np.degrees(robot.get_pitch())))

            if step % frame_every == 0:
                writer.append_data(annotate(env.render(), [
                    f"t = {t:5.1f} s   falls: {falls}",
                    f"speed  {robot.get_wheel_speed():6.1f}  target {robot.target_wheel_speed:6.1f}",
                    yaw_status_line(robot, env_id),
                    f"pitch  {np.degrees(robot.get_pitch()):6.1f} deg",
                ]))

            if terminated:
                falls += 1
                obs, _ = env.reset()
                last_reset = t

    env.close()
    return {
        "duration_s": duration,
        "reset_grace_s": reset_grace,
        "falls": falls,
        "mean_speed_error": float(np.mean(speed_errors)),
        "mean_yaw_error": float(np.mean(yaw_errors)),
        "mean_abs_pitch_deg": float(np.mean(pitches)),
        # each env scores with its own reward function (see its _get_reward)
        "max_reward": max(rewards),
        "total_reward": sum(rewards),
    }


@cli.command(help="Record the robustness test (same seeds, alternating block sides) as a video")
@ticket_option
@click.option('-m', '--model', required=True, help="model .zip")
@click.option('-o', '--output', default=None, type=click.Path(path_type=Path),
              help="output .mp4 (default: experiments/<TICKET>/evaluation/robustness.mp4)")
@click.option('-n', '--episodes', default=50)
@click.option('--speed', default=10, help="playback speed multiplier (1 = real time)")
@click.option('--seed', default=0)
def blocks(ticket: str, model: str, output: Path | None, episodes: int, speed: int, seed: int):
    experiment = Experiment(ticket)
    experiment.warn_if_unmapped()
    output = output or experiment.robustness_path(suffix=".mp4")
    env = gym.make("Env03-v2", render_mode="rgb_array")
    robot = env.unwrapped
    policy = stable_baselines3.PPO.load(model, device="cpu")
    max_len = env.spec.max_episode_steps
    np.random.seed(seed)  # matches the robustness command

    frame_every = round(speed / (VIDEO_FPS * CONTROL_DT))
    survived = {"front": 0, "back": 0}
    played = {"front": 0, "back": 0}

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=VIDEO_FPS) as writer:
        for ep in range(episodes):
            side = "front" if ep % 2 == 0 else "back"
            robot.attack_side_front = side == "front"
            obs, _ = env.reset(seed=seed + ep)
            length, done = 0, False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                length += 1
                done = terminated or truncated
                if length % frame_every == 0 or terminated:
                    writer.append_data(annotate(env.render(), [
                        f"{speed}x   episode {ep + 1}/{episodes}   blocks: {side}",
                        f"t = {length * CONTROL_DT:4.1f} / {max_len * CONTROL_DT:.0f} s"
                        + ("   FELL" if terminated else ""),
                        f"front survived {survived['front']}/{played['front']}",
                        f"back  survived {survived['back']}/{played['back']}",
                    ]))
            played[side] += 1
            survived[side] += length >= max_len

    env.close()
    experiment.write_metadata(
        command="evaluate blocks",
        model=str(model),
        environment="Env03-v2",
        episodes=episodes,
        seed=seed,
        output=str(output),
        results={"survived": survived, "played": played},
    )
    click.echo(f"Wrote {output}: front {survived['front']}/{played['front']}, "
               f"back {survived['back']}/{played['back']} survived")


if __name__ == "__main__":
    cli()
