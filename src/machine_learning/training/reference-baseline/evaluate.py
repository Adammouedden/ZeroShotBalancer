"""
Headless evaluation of trained balancing policies.

  robustness: run Env03-v2 episodes with blocks fired at the front AND back,
              report survival rate per side (training only ever sees one side).
  video:      record a continuous clip in Env01-v3 with changing speed / yaw
              targets, resetting the robot whenever it falls.
"""
import click
import gymnasium as gym
import imageio
import numpy as np
import stable_baselines3

from PIL import Image, ImageDraw

# registers Env01-v3 / Env03-v2
import envs
from envs.env01_v1 import Env01

CONTROL_DT = 0.005  # 250 physics steps x 0.02ms per env step
VIDEO_FPS = 50


@click.group()
def cli():
    pass


@cli.command(help="Survival rate on Env03-v2 with blocks from the front and back")
@click.option('-m', '--model', 'models', required=True, multiple=True, help="model .zip (repeat to compare)")
@click.option('-n', '--episodes', default=50, help="episodes per model (split evenly between sides)")
@click.option('--seed', default=0)
def robustness(models: tuple[str, ...], episodes: int, seed: int):
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
        for side, rows in list(results.items()) + [("all", results["front"] + results["back"])]:
            lengths = np.array([r[0] for r in rows])
            returns = np.array([r[1] for r in rows])
            survived = (lengths >= max_len).mean() * 100
            click.echo(
                f"  {side:6s} {survived:9.0f}% {lengths.mean():7.0f}/{max_len} {returns.mean():12.1f}"
            )
        env.close()


# (start time s, target wheel speed, target yaw) - held until the next entry
MIXED_SCHEDULE = [
    (0, 0, 0),
    (2, 15, 0),
    (6, -15, 0),
    (10, 30, 0),
    (14, 0, 10),
    (18, 15, -10),
    (22, 0, 0),
    (25, -30, 0),
    (28, 0, 0),
]


def schedule_target(t: float) -> tuple[float, float]:
    speed, yaw = 0.0, 0.0
    for start, s, y in MIXED_SCHEDULE:
        if t >= start:
            speed, yaw = s, y
    return speed, yaw


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 330, 22 * len(lines) + 12], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((10, 8 + 22 * i), line, fill=(255, 255, 255), font_size=18)
    return np.asarray(img)


@cli.command(help="Record a continuous Env01-v3 clip, resetting whenever the robot falls")
@click.option('-m', '--model', required=True, help="model .zip")
@click.option('-o', '--output', default="movies/balance_eval.mp4")
@click.option('-d', '--duration', default=30.0, help="clip length in simulated seconds")
@click.option('--schedule', type=click.Choice(["mixed", "env01"]), default="mixed",
              help="mixed: custom speed + yaw changes; env01: Env01-v3's own speed-only schedule")
@click.option('--seed', default=0)
def video(model: str, output: str, duration: float, schedule: str, seed: int):
    env = gym.make("Env01-v3", render_mode="rgb_array")
    robot = env.unwrapped
    policy = stable_baselines3.PPO.load(model, device="cpu")
    np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    frame_every = round(1 / (VIDEO_FPS * CONTROL_DT))
    total_steps = round(duration / CONTROL_DT)
    falls = 0

    with imageio.get_writer(output, fps=VIDEO_FPS) as writer:
        for step in range(total_steps):
            t = step * CONTROL_DT
            action, _ = policy.predict(obs, deterministic=True)
            if schedule == "mixed":
                # bypass Env01_v3.step's built-in schedule, drive the targets ourselves
                robot.target_wheel_speed, robot.target_yaw = schedule_target(t)
                obs, _, terminated, _, _ = Env01.step(robot, action)
            else:
                # Env01-v3 schedule runs on episode time; ignore the 6000-step truncation
                obs, _, terminated, _, _ = env.step(action)

            if step % frame_every == 0:
                writer.append_data(annotate(env.render(), [
                    f"t = {t:5.1f} s   falls: {falls}",
                    f"speed  {robot.get_wheel_speed():6.1f}  target {robot.target_wheel_speed:6.1f}",
                    f"yaw    {robot.get_wheel_yaw():6.1f}  target {robot.target_yaw:6.1f}",
                    f"pitch  {np.degrees(robot.get_pitch()):6.1f} deg",
                ]))

            if terminated:
                falls += 1
                obs, _ = env.reset()

    env.close()
    click.echo(f"Wrote {output}: {duration:.0f}s, {falls} fall(s), schedule={schedule}")


@cli.command(help="Record the robustness test (same seeds, alternating block sides) as a video")
@click.option('-m', '--model', required=True, help="model .zip")
@click.option('-o', '--output', default="movies/robustness.mp4")
@click.option('-n', '--episodes', default=50)
@click.option('--speed', default=10, help="playback speed multiplier (1 = real time)")
@click.option('--seed', default=0)
def blocks(model: str, output: str, episodes: int, speed: int, seed: int):
    env = gym.make("Env03-v2", render_mode="rgb_array")
    robot = env.unwrapped
    policy = stable_baselines3.PPO.load(model, device="cpu")
    max_len = env.spec.max_episode_steps
    np.random.seed(seed)  # matches the robustness command

    frame_every = round(speed / (VIDEO_FPS * CONTROL_DT))
    survived = {"front": 0, "back": 0}
    played = {"front": 0, "back": 0}

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
    click.echo(f"Wrote {output}: front {survived['front']}/{played['front']}, "
               f"back {survived['back']}/{played['back']} survived")


if __name__ == "__main__":
    cli()
