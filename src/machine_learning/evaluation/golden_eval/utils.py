"""
Utilities for the golden evaluation schedules: load one, integrate it into a path,
and draw it.

A schedule JSON is a list of [start time s, target wheel speed, target yaw], each entry held
until the next one starts (the same contract evaluate.py's load_schedule uses). Both target
columns are in the units the policy reads, not the units the geometry is sampled in:

    target wheel speed - rad/s, matched against RobotBaseEnv.get_wheel_speed()
    target yaw         - an absolute heading in RADIANS, wrapped to +/- pi

get_wheel_speed() is the mean of the two wheels' angular velocities, which for a differential
drive is the body's forward speed divided by the wheel radius. So the body speed of segment i
is speed_i * WHEEL_RADIUS, and the robot, tracking the schedule perfectly, walks

    p_(i+1) = p_i + (speed_i * WHEEL_RADIUS) * dt_i * (cos yaw_i, sin yaw_i),  dt_i = t_(i+1) - t_i

A negative speed is a reversal: the robot keeps facing yaw_i and drives backwards, so the
displacement flips while the heading does not.

starting from the origin at the centre of the test area. That is the path drawn here:
the *commanded* path, not a simulated one - it is what the schedule asks for, so it is
what tells you whether the schedule stays inside the boundary and how hard it swerves.

Usage:
    python utils.py visualize                        # schedules/test.json, 25 m^2 area
    python utils.py visualize -s test.json -a 25     # explicit
    python utils.py visualize --labels --show        # annotate times, open a window
    python utils.py check -s test.json -a 25         # boundary / speed report, no plot

Area is given the way generate_eval.py takes it - in square metres - and the boundary is
drawn as the square of that area centred on the origin, i.e. +/- sqrt(area) / 2 per axis.
Schedules written before the units were settled can still be read with --units mps and/or
--yaw-degrees.
"""
import click
import json
import math

from pathlib import Path

GOLDEN_EVAL_DIR = Path(__file__).resolve().parent
SCHEDULES_DIR = GOLDEN_EVAL_DIR / "schedules"
PLOTS_DIR = GOLDEN_EVAL_DIR / "plots"

WHEEL_RADIUS = 0.0325  # 65 mm wheels, from notes.md - converts the rad/s targets into metres
# robot_calibrated.xml caps both velocity actuators at +/- 20.6 rad/s, so a target past this is
# uncommandable: the error can never close and the robot just leans into it for the whole segment
ACTUATOR_LIMIT = 20.6

# Palette: one blue sequential ramp for speed magnitude (light -> dark = slow -> fast),
# recessive greys for the boundary and grid, and text in ink rather than in series colour.
SPEED_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BOUNDARY_COLOUR = "#e34948"  # the hard wall - the one thing the path must never cross
THRESHOLD_COLOUR = "#8a8984"
GRID_COLOUR = "#e4e3df"


def schedule_path(name: str) -> Path:
    """Resolve a schedule by bare name inside schedules/, or by an explicit path."""
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    for guess in (SCHEDULES_DIR / name, SCHEDULES_DIR / f"{name}.json"):
        if guess.is_file():
            return guess
    raise click.BadParameter(f"no schedule {name!r} in {SCHEDULES_DIR} (or as a path)")


def load_schedule(name: str) -> list[list[float]]:
    """Load and validate a schedule JSON, sorted by start time."""
    path = schedule_path(name)
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if not schedule or any(len(entry) != 3 for entry in schedule):
        raise click.BadParameter(f"{path} must be a non-empty list of [start, speed, yaw]")
    return sorted((list(map(float, entry)) for entry in schedule), key=lambda entry: entry[0])


def integrate_schedule(schedule: list[list[float]], units: str = "rad",
                       yaw_degrees: bool = False, wheel_radius: float = WHEEL_RADIUS) -> list[dict]:
    """
    Walk a schedule into waypoints.

    Returns one dict per entry: the position the robot is at when that entry starts, plus the
    segment it commands from there. Each waypoint keeps the commanded wheel speed in rad/s
    (`speed`, what the file says and the policy reads) alongside the body speed in m/s
    (`body_speed`, what the geometry is in), so nothing downstream has to re-derive either.

    The last entry has duration 0 - it is the arrival, with nothing after it to hold.
    """
    waypoints, x, y = [], 0.0, 0.0
    for index, (start, speed, yaw) in enumerate(schedule):
        # the file is rad/s of wheel; the path is in metres of body travel
        wheel_speed = speed if units == "rad" else speed / wheel_radius
        body_speed = wheel_speed * wheel_radius
        heading = math.radians(yaw) if yaw_degrees else yaw
        duration = schedule[index + 1][0] - start if index + 1 < len(schedule) else 0.0
        distance = body_speed * duration  # signed: a negative speed drives backwards

        waypoints.append({
            "index": index,
            "time": start,
            "x": x,
            "y": y,
            "speed": wheel_speed,
            "body_speed": body_speed,
            "yaw": heading,  # always radians from here on
            "heading": heading,
            "duration": duration,
            "distance": distance,
            "reverse": distance < 0,
            "dx": distance * math.cos(heading),
            "dy": distance * math.sin(heading),
        })
        x += distance * math.cos(heading)
        y += distance * math.sin(heading)

    return waypoints


def summarize(waypoints: list[dict], area: float, boundary_threshold: float = 0.0,
              actuator_limit: float = ACTUATOR_LIMIT) -> dict:
    """Path statistics plus any boundary breach, for the caption and the check command."""
    max_distance = math.sqrt(area) / 2
    limit = max_distance - boundary_threshold

    # a reversal has negative distance and is still a moving segment, so test against zero
    moving = [point for point in waypoints if point["distance"] != 0]
    path_length = sum(abs(point["distance"]) for point in moving)
    duration = waypoints[-1]["time"] - waypoints[0]["time"] if waypoints else 0.0
    # heading change between consecutive moving segments, wrapped to +/- pi, reported in degrees
    turns = [
        abs(math.degrees((later["yaw"] - earlier["yaw"] + math.pi) % (2 * math.pi) - math.pi))
        for earlier, later in zip(moving, moving[1:])
    ]

    breaches = [
        point for point in waypoints
        if abs(point["x"]) > limit + 1e-9 or abs(point["y"]) > limit + 1e-9
    ]
    # targets the calibrated actuators physically cannot reach
    over_limit = [point for point in moving if abs(point["speed"]) > actuator_limit + 1e-9]

    return {
        "area": area,
        "max_distance": max_distance,
        "limit": limit,
        "segments": len(moving),
        "reversals": sum(1 for point in moving if point["reverse"]),
        "duration": duration,
        "path_length": path_length,
        "max_speed": max((abs(point["speed"]) for point in moving), default=0.0),
        "mean_speed": (sum(abs(point["speed"]) * point["duration"] for point in moving) / duration
                       if duration else 0.0),
        "max_body_speed": max((abs(point["body_speed"]) for point in moving), default=0.0),
        "mean_body_speed": path_length / duration if duration else 0.0,
        "actuator_limit": actuator_limit,
        "over_limit": over_limit,
        "max_abs_x": max((abs(point["x"]) for point in waypoints), default=0.0),
        "max_abs_y": max((abs(point["y"]) for point in waypoints), default=0.0),
        "max_turn": max(turns, default=0.0),
        "mean_turn": sum(turns) / len(turns) if turns else 0.0,
        "settle_time": next((point["time"] for point in waypoints if point["distance"] != 0), 0.0),
        "breaches": breaches,
    }


def draw_schedule(waypoints: list[dict], stats: dict, title: str, output: Path | None = None,
                  labels: bool = False, show: bool = False, dpi: int = 200):
    """
    Draw the boundary square and the commanded path as one arrow per schedule segment,
    shaded light -> dark by commanded speed. Saves to `output` unless it is None.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # headless by default; a window is opt-in via --show

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    speed_ramp = LinearSegmentedColormap.from_list("speed", SPEED_RAMP)
    # a pure-balancing schedule commands no motion at all - that is a valid schedule, so draw
    # the boundary and say so rather than failing
    moving = [point for point in waypoints if point["distance"] != 0]
    # colour encodes how hard the segment is driven, so a reversal shades by its magnitude;
    # the direction of travel is already carried by the arrow itself
    speeds = [abs(point["speed"]) for point in moving]
    # floor the ramp a little above white so the slowest arrow still reads on the surface
    norm = (Normalize(vmin=min(speeds) - 0.15 * (max(speeds) - min(speeds) or 1), vmax=max(speeds))
            if moving else Normalize(vmin=0, vmax=1))

    figure, axes = plt.subplots(figsize=(8.5, 8.5), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)

    extent = stats["max_distance"]
    axes.add_patch(Rectangle((-extent, -extent), 2 * extent, 2 * extent, fill=False,
                             edgecolor=BOUNDARY_COLOUR, linewidth=2, zorder=1,
                             label=f"boundary  {stats['area']:g} m²  ({2 * extent:g} × {2 * extent:g} m)"))
    if stats["limit"] < extent:
        inset = stats["limit"]
        axes.add_patch(Rectangle((-inset, -inset), 2 * inset, 2 * inset, fill=False,
                                 edgecolor=THRESHOLD_COLOUR, linewidth=1, linestyle=(0, (4, 3)),
                                 zorder=1, label=f"threshold  {extent - inset:g} m margin"))

    # the path itself: a thin spine for continuity, then a vector arrow per segment
    axes.plot([point["x"] for point in waypoints], [point["y"] for point in waypoints],
              color=GRID_COLOUR, linewidth=2, solid_capstyle="round", zorder=2)
    if moving:
        axes.quiver(
            [point["x"] for point in moving], [point["y"] for point in moving],
            [point["dx"] for point in moving], [point["dy"] for point in moving],
            speeds, cmap=speed_ramp, norm=norm,
            angles="xy", scale_units="xy", scale=1,
            width=0.005, headwidth=4.5, headlength=5.5, headaxislength=5,
            zorder=3,
        )
    else:
        axes.annotate("no motion commanded - the schedule holds the robot at rest",
                      xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=10,
                      color=TEXT_SECONDARY, zorder=5)

    # a surface-coloured halo keeps every label readable where it lands on top of an arrow
    halo = {"boxstyle": "round,pad=0.18", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.82}

    # reversals: same arrow, but ring the tail so a backwards segment is visible at a glance
    reversals = [point for point in moving if point["reverse"]]
    if reversals:
        axes.plot([point["x"] for point in reversals], [point["y"] for point in reversals],
                  "o", markersize=7, markerfacecolor="none", markeredgecolor=TEXT_SECONDARY,
                  markeredgewidth=1.4, linestyle="none", zorder=4,
                  label=f"reversal  ({len(reversals)} of {len(moving)})")

    start, end = waypoints[0], waypoints[-1]
    # start and end can sit almost on top of each other - push their labels opposite ways
    for point, name, marker, offset in ((start, "start", "o", (12, -16)), (end, "end", "s", (12, 12))):
        axes.plot(point["x"], point["y"], marker, markersize=9, markerfacecolor=SURFACE,
                  markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.8, zorder=7)
        axes.annotate(f"{name}  t={point['time']:g}s", (point["x"], point["y"]),
                      textcoords="offset points", xytext=offset, fontsize=9,
                      color=TEXT_PRIMARY, fontweight="bold", bbox=halo, zorder=8)
    if stats["settle_time"]:
        axes.annotate(f"settle {stats['settle_time']:g}s at origin", (start["x"], start["y"]),
                      textcoords="offset points", xytext=(12, -30), fontsize=8.5,
                      color=TEXT_SECONDARY, bbox=halo, zorder=8)

    if labels:
        for point in moving[1:]:
            axes.annotate(f"{point['time']:.1f}s", (point["x"], point["y"]),
                          textcoords="offset points", xytext=(6, 6), fontsize=7.5,
                          color=TEXT_SECONDARY, bbox=halo, zorder=5)

    # every breach gets a marker, but only the first few get text - past that the labels
    # overlap into an unreadable smear and the count in the caption is the useful number
    MAX_BREACH_LABELS = 5
    for index, breach in enumerate(stats["breaches"]):
        axes.plot(breach["x"], breach["y"], "x", markersize=12, markeredgewidth=2.4,
                  color=BOUNDARY_COLOUR, zorder=6)
        if index >= MAX_BREACH_LABELS:
            continue
        # label towards the middle of the plot, so a breach on the right edge does not
        # run its text off the figure and into the colourbar
        inward = -1 if breach["x"] > 0 else 1
        axes.annotate(f"outside at t={breach['time']:.1f}s", (breach["x"], breach["y"]),
                      textcoords="offset points", xytext=(8 * inward, -12), fontsize=8,
                      ha="left" if inward > 0 else "right",
                      color=BOUNDARY_COLOUR, fontweight="bold", bbox=halo, zorder=6)

    if moving:
        bar = figure.colorbar(axes.collections[0], ax=axes, fraction=0.035, pad=0.02)
        bar.set_label("commanded wheel speed (rad/s)", color=TEXT_SECONDARY, fontsize=9)
        bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        bar.outline.set_visible(False)

    # keep the view square and centred, but widen it if the path escapes the boundary so a
    # breach is never cropped out of the picture
    reach = max(extent, stats["max_abs_x"], stats["max_abs_y"])
    margin = 0.08 * reach
    axes.set_xlim(-reach - margin, reach + margin)
    axes.set_ylim(-reach - margin, reach + margin)
    axes.set_aspect("equal", adjustable="box")
    axes.set_box_aspect(1)
    axes.set_xlabel("x (m)", color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel("y (m)", color=TEXT_SECONDARY, fontsize=9)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axes.grid(True, color=GRID_COLOUR, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)

    # wrapped: one long line forces bbox_inches="tight" to stretch the whole canvas sideways
    caption = "\n".join(filter(None, [
        f"{stats['segments']} segments over {stats['duration']:g} s   ·   "
        f"{stats['reversals']} reversal(s)   ·   path {stats['path_length']:.2f} m",

        f"wheel speed mean {stats['mean_speed']:.1f} / max {stats['max_speed']:.1f} rad/s "
        f"(max {stats['max_body_speed']:.2f} m/s)   ·   "
        f"turn mean {stats['mean_turn']:.0f}° / max {stats['max_turn']:.0f}°",

        f"reach |x| {stats['max_abs_x']:.2f} m, |y| {stats['max_abs_y']:.2f} m "
        f"against a {stats['limit']:.2f} m limit",

        (f"{len(stats['over_limit'])} of {stats['segments']} segments exceed the "
         f"{stats['actuator_limit']:g} rad/s actuator limit - those targets are unreachable")
        if stats["over_limit"] else "",

        (f"{len(stats['breaches'])} waypoint(s) outside the boundary")
        if stats["breaches"] else "",
    ]))
    caption_lines = caption.count("\n") + 1
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold",
                   loc="left", pad=14 + 12 * caption_lines)
    axes.annotate(caption, xy=(0, 1.015), xycoords="axes fraction", fontsize=8.5,
                  color=TEXT_SECONDARY, va="bottom", linespacing=1.5)

    legend = axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False,
                         fontsize=8.5, labelcolor=TEXT_SECONDARY, ncols=3)
    legend.set_zorder(5)

    figure.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output


def report(stats: dict) -> str:
    """The same numbers as the plot caption, as text."""
    lines = [
        f"area              {stats['area']:g} m²  ->  ±{stats['max_distance']:.3f} m per axis",
        f"usable limit      ±{stats['limit']:.3f} m",
        f"segments          {stats['segments']}  ({stats['reversals']} reversal(s), "
        f"settle {stats['settle_time']:g} s)",
        f"duration          {stats['duration']:g} s",
        f"path length       {stats['path_length']:.3f} m",
        f"wheel speed       mean {stats['mean_speed']:.3f}  max {stats['max_speed']:.3f} rad/s",
        f"body speed        mean {stats['mean_body_speed']:.3f}  max {stats['max_body_speed']:.3f} m/s",
        f"heading change    mean {stats['mean_turn']:.1f}°  max {stats['max_turn']:.1f}°",
        f"furthest reach    |x| {stats['max_abs_x']:.3f} m   |y| {stats['max_abs_y']:.3f} m",
    ]
    if stats["over_limit"]:
        lines.append(f"ACTUATOR          {len(stats['over_limit'])} of {stats['segments']} segments "
                     f"exceed ±{stats['actuator_limit']:g} rad/s and cannot be reached:")
        lines += [f"                  t={point['time']:.3f}s  {point['speed']:+.3f} rad/s"
                  for point in stats["over_limit"][:8]]
        if len(stats["over_limit"]) > 8:
            lines.append(f"                  ... and {len(stats['over_limit']) - 8} more")
    else:
        lines.append(f"actuator          ok - every target within ±{stats['actuator_limit']:g} rad/s")
    if stats["breaches"]:
        lines.append(f"BOUNDARY          {len(stats['breaches'])} waypoint(s) outside the limit:")
        lines += [f"                  t={point['time']:.3f}s  "
                  f"({point['x']:+.3f}, {point['y']:+.3f})" for point in stats["breaches"]]
    else:
        lines.append("boundary          ok - every waypoint inside the limit")
    return "\n".join(lines)


schedule_option = click.option('-s', '--schedule', 'schedule_name', default="test.json",
                               help="schedule JSON: a name inside schedules/, or a path")
area_option = click.option('-a', '--area', default=25.0, type=float,
                           help="area of the test boundary in m² (drawn as a square)")
threshold_option = click.option('-bt', '--boundary-threshold', default=0.0, type=float,
                                help="safety margin in m, drawn as an inner dashed square")
units_option = click.option('-u', '--units', type=click.Choice(["rad", "mps"]), default="rad",
                            help="speed column units: wheel rad/s (default), or legacy body m/s")
yaw_option = click.option('--yaw-degrees', is_flag=True, default=False,
                          help="read the yaw column as degrees (legacy schedules); default is radians")
wheel_option = click.option('-wr', '--wheel-radius', default=WHEEL_RADIUS, type=float,
                            help="wheel radius in m, converting wheel rad/s to body m/s")
limit_option = click.option('-al', '--actuator-limit', default=ACTUATOR_LIMIT, type=float,
                            help="flag targets past this wheel speed in rad/s as unreachable")


@click.group(help=__doc__.split("Usage:")[0].strip())
def cli():
    pass


@cli.command(help="Draw the boundary and the path the schedule commands, as vector arrows")
@schedule_option
@area_option
@threshold_option
@units_option
@yaw_option
@wheel_option
@limit_option
@click.option('-o', '--output', type=click.Path(path_type=Path), default=None,
              help="output image (default: plots/<schedule>.png)")
@click.option('--labels', is_flag=True, default=False, help="annotate each waypoint with its start time")
@click.option('--show', is_flag=True, default=False, help="open an interactive window as well")
@click.option('--dpi', default=200, type=int, help="output resolution")
def visualize(schedule_name: str, area: float, boundary_threshold: float, units: str,
              yaw_degrees: bool, wheel_radius: float, actuator_limit: float,
              output: Path | None, labels: bool, show: bool, dpi: int):
    path = schedule_path(schedule_name)
    schedule = load_schedule(schedule_name)
    waypoints = integrate_schedule(schedule, units=units, yaw_degrees=yaw_degrees,
                                   wheel_radius=wheel_radius)
    stats = summarize(waypoints, area, boundary_threshold, actuator_limit=actuator_limit)

    if output is None:
        output = PLOTS_DIR / f"{path.stem}.png"

    title = f"{path.stem} - commanded path in a {math.sqrt(area):g} × {math.sqrt(area):g} m area"
    draw_schedule(waypoints, stats, title, output=output, labels=labels, show=show, dpi=dpi)

    click.echo(report(stats))
    if stats["breaches"]:
        click.secho(f"\nwarning: schedule leaves the boundary at "
                    f"{len(stats['breaches'])} waypoint(s)", fg="red")
    if stats["over_limit"]:
        click.secho(f"warning: {len(stats['over_limit'])} segment(s) command more than "
                    f"±{stats['actuator_limit']:g} rad/s", fg="yellow")
    click.echo(f"\nwrote {output}")


@cli.command(help="Report the path statistics and boundary check without drawing anything")
@schedule_option
@area_option
@threshold_option
@units_option
@yaw_option
@wheel_option
@limit_option
def check(schedule_name: str, area: float, boundary_threshold: float, units: str,
          yaw_degrees: bool, wheel_radius: float, actuator_limit: float):
    waypoints = integrate_schedule(load_schedule(schedule_name), units=units,
                                   yaw_degrees=yaw_degrees, wheel_radius=wheel_radius)
    stats = summarize(waypoints, area, boundary_threshold, actuator_limit=actuator_limit)
    click.echo(report(stats))
    if stats["breaches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
