"""Containment of the upright calibrated robot on the ideal golden schedule path."""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from evaluation.golden_eval.generate_eval import varying_time_generator
from evaluation.golden_eval.utils import SCHEDULES_DIR, integrate_schedule


@pytest.fixture(scope="module")
def footprint():
    """Return (forward, left, radius) for collision corners and the round chassis brim."""
    root = Path(__file__).resolve().parents[2]
    body = ET.parse(root / "training/robot_environments/robot_calibrated.xml").find(".//body[@name='robot_body']")
    points = []
    for geom in body.findall("geom"):
        if geom.get("type") == "mesh":  # visual-only CAD mesh
            continue
        x, y, _ = map(float, geom.attrib["pos"].split())
        size = list(map(float, geom.attrib["size"].split()))
        if geom.get("type") == "cylinder":  # upright brim projects to a circle
            points.append((y, -x, size[0]))
        else:
            assert geom.get("type") == "box"
            points.extend((y + dy, -x + dx, 0) for dy in (-size[1], size[1]) for dx in (-size[0], size[0]))
    for name in ("l_wheel", "r_wheel"):
        wheel = body.find(f"body[@name='{name}']")
        x, y, _ = map(float, wheel.attrib["pos"].split())
        radius, half_width = map(float, wheel.find("geom").attrib["size"].split())
        # The axle is lateral. An upright wheel projects to a rectangle on the floor.
        points.extend((y + dy, -x + dx, 0) for dy in (-radius, radius) for dx in (-half_width, half_width))
    return points


def assert_inside_area(schedule, area, footprint, source):
    half_side = math.sqrt(area) / 2
    reach = max(math.hypot(fwd, left) + radius for fwd, left, radius in footprint)
    previous_heading = 0.0
    for point in integrate_schedule(schedule):
        x, y, heading = point["x"], point["y"], point["heading"]
        turn = math.remainder(heading - previous_heading, math.tau)
        start_heading = previous_heading
        previous_heading = heading
        if max(abs(x), abs(y)) + reach <= half_side:
            continue  # Every orientation fits here.
        # The square is convex, so checking both ends covers each straight move.
        # During the in-place turn each corner traces an arc. Its extreme x/y
        # occurs at an endpoint or at a multiple of pi/2 along that arc.
        for fwd, left, radius in footprint:
            length = math.hypot(fwd, left)
            start = start_heading + math.atan2(left, fwd)
            lo, hi = sorted((start, start + turn))
            quarter = math.pi / 2
            angles = [lo, hi]
            angles.extend(k * quarter for k in range(math.ceil(lo / quarter), math.floor(hi / quarter) + 1))
            for angle in angles:
                edge = max(abs(x + length * math.cos(angle)), abs(y + length * math.sin(angle))) + radius
                assert edge <= half_side + 1e-9, (
                    f"{source}, t={point['time']} s: robot edge {edge:.5f} m exceeds wall {half_side:g} m"
                )


@pytest.mark.parametrize("path", sorted(SCHEDULES_DIR.glob("*.json")), ids=lambda path: path.name)
def test_saved_schedule_inside_area(path, footprint):
    # The JSON rows carry commands only. Supply the test arena for each saved example.
    areas_m2 = {"test.json": 25, "pure_balancing.json": 25}
    assert path.name in areas_m2, f"Specify the test area in m² for {path.name}"
    assert_inside_area(json.loads(path.read_text()), areas_m2[path.name], footprint, path.name)


@pytest.mark.parametrize("area_m2", [4, 25, 56.25], ids=["2x2m", "5x5m", "7.5x7.5m"])
def test_generated_schedules_inside_area(footprint, area_m2):
    breaches = []
    for seed in range(1000):
        schedule = varying_time_generator(area=area_m2, total_time=60, max_speed=1, seed=seed)
        try:
            assert_inside_area(schedule, area_m2, footprint, f"seed {seed}")
        except AssertionError as error:
            breaches.append(str(error).splitlines()[0])
    assert not breaches, f"{len(breaches)} of 1000 schedules leave the area. First breach: {breaches[0]}"


def test_turn_can_cross_wall_with_both_end_poses_inside(footprint):
    # In a 2 m square, move to x=0.90 m. Both forward and backward headings
    # fit, but a wheel crosses the x=1 m wall during the half turn.
    approach = [[0, 0.90 / 0.0325, 0], [1, 0, 0]]
    assert_inside_area(approach, 4, footprint, "approach")
    with pytest.raises(AssertionError, match="exceeds wall"):
        assert_inside_area(approach + [[2, 0, math.pi]], 4, footprint, "turn")
