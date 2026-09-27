"""Independently audit saved evidence without rerunning a simulation."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from model_adapter import AXES, PROJECT, digest, write_json
from runner import selected_directory


def verify(directory):
    directory = Path(directory).resolve()
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    config, manifest, provenance, status = (load(n) for n in ("config.json", "manifest.json", "provenance.json", "status.json"))
    assert status["state"] == "completed", status["state"]
    assert load("reference_check.json")["passed"]
    baseline_rows = load("baseline_results.json")
    assert sorted(r["seed"] for r in baseline_rows) == sorted(config["seeds"])
    assert all(r["balance"] and r["numerical"] == "valid_observed" and not r["review_flags"] for r in baseline_rows)
    assert digest(directory / "policy.zip") == provenance["policy"]["checkpoint_sha256"]
    current_source_changes = []
    for relative, expected in provenance["source_hashes"].items():
        assert digest(directory / "sources" / relative) == expected, relative
        if not (PROJECT / relative).exists() or digest(PROJECT / relative) != expected:
            current_source_changes.append(relative)
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_keys = {(case["id"], seed) for case in manifest["cases"] if "xml" in case for seed in config["seeds"]}
    keys = [(r["case_id"], r["seed"]) for r in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys
    assert len(rows) == status["completed_trials"] == status["expected_trials"]
    cases = {c["id"]: c for c in manifest["cases"]}
    baseline = mujoco.MjModel.from_xml_path(str(directory / "baseline.xml"))
    zero_matches = []
    for case in manifest["cases"]:
        if "xml" not in case:
            assert case["physical"]["status"] in {"known_infeasible", "invalid_input"}
            continue
        path = directory / "variants" / case["xml"]
        assert digest(path) == case["sha256"]
        if config["property"] == "center_of_mass":
            actual = mujoco.MjModel.from_xml_path(str(path))
            expected = baseline.body_ipos.copy()
            for direction, value in case["values"].items():
                expected[baseline.body("robot_body").id, AXES[direction]] += value
            np.testing.assert_allclose(actual.body_ipos, expected, rtol=1e-13, atol=1e-14)
            for field in ("body_mass", "body_inertia", "body_iquat", "geom_size", "geom_pos", "dof_damping", "actuator_forcerange"):
                np.testing.assert_allclose(getattr(actual, field), getattr(baseline, field), rtol=1e-13, atol=1e-14)
            if all(abs(v) < 1e-12 for v in case["values"].values()):
                for seed in config["seeds"]:
                    a = directory / "baseline_worker" / f"trace_seed{seed}.jsonl"
                    b = selected_directory(directory / "tests" / case["id"]) / f"trace_seed{seed}.jsonl"
                    assert digest(a) == digest(b), "Zero-offset trace differs from neutral baseline"
                    zero_matches.append(seed)
    trace_rows = 0
    for row in rows:
        assert cases[row["case_id"]]["physical"]["status"] not in {"known_infeasible", "invalid_input"}
        if row["numerical"] == "valid_observed":
            assert math.isclose(row["simulated_seconds"], config["seconds_per_seed"], abs_tol=1e-9)
        path = selected_directory(directory / "tests" / row["case_id"]) / f"trace_seed{row['seed']}.jsonl"
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                assert entry["target_speed"] == entry["target_yaw"] == 0
                assert np.isfinite(entry["qpos"]).all() and np.isfinite(entry["qvel"]).all()
                count += 1
        if row["numerical"] == "valid_observed":
            assert count == round(config["seconds_per_seed"] / .005)
        trace_rows += count
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows) + sum("xml" not in c for c in cases.values())
    result = {"passed": True, "trials": len(rows), "trace_steps_checked": trace_rows,
        "numerically_valid": sum(r["numerical"] == "valid_observed" for r in rows),
        "balance_successes": sum(r["balance"] for r in rows), "review_flagged_trials": sum(bool(r["review_flags"]) for r in rows),
        "excluded_cases": sum("xml" not in c for c in cases.values()), "zero_offset_trace_matching_seeds": zero_matches,
        "simulated_seconds": sum(r["simulated_seconds"] for r in rows),
        "physical_case_counts": dict(collections.Counter(c["physical"]["status"] for c in cases.values())),
        "current_source_changes_since_snapshot": current_source_changes,
        "checks": ["source snapshots", "checkpoint hash", "variant hashes", "CoM compiled mutations", "neutral zero targets", "complete unique seed coverage", "CSV/JSONL consistency", "baseline qualification", "finite trace states"]}
    write_json(directory / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
