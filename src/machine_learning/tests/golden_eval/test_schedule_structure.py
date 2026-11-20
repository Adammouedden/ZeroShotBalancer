"""Check golden schedule rows in their original JSON order."""

import json
import math
from pathlib import Path

import pytest

from evaluation.golden_eval.generate_eval import varying_time_generator


SCHEDULES = Path(__file__).resolve().parents[2] / "evaluation/golden_eval/schedules"


def assert_schedule_structure(schedule, source):
    assert isinstance(schedule, list) and schedule, f"{source}: expected a nonempty array"
    previous_time = None
    for row_number, row in enumerate(schedule, start=1):
        location = f"{source}, row {row_number}"
        assert isinstance(row, list) and len(row) == 3, f"{location}: expected [time, speed, heading]"
        for field, value in zip(("time", "speed", "heading"), row):
            assert type(value) in (int, float) and math.isfinite(value), (
                f"{location}: {field} must be a finite number, got {value!r}"
            )
        start = row[0]
        assert start >= 0, f"{location}: negative start time {start}"
        assert previous_time is None or start > previous_time, (
            f"{location}: start time {start} must follow {previous_time}"
        )
        previous_time = start


@pytest.mark.parametrize("path", sorted(SCHEDULES.glob("*.json")), ids=lambda path: path.name)
def test_saved_schedule_structure(path):
    assert_schedule_structure(json.loads(path.read_text(encoding="utf-8")), path.name)


def test_generated_schedule_structure():
    for seed in range(1000):
        schedule = varying_time_generator(area=25, total_time=60, max_speed=1, seed=seed)
        assert_schedule_structure(schedule, f"seed {seed}")
