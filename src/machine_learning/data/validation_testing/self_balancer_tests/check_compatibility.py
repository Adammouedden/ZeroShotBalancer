"""Compare current baseline/selected cases with a saved run; never overwrite it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco

from bounds import validate_config
from model_adapter import (HERE, WORLD, compare_models, compile_root, configured_baseline,
                           digest, explicit_baseline, save_xml, write_json)
from runner import execute_job, read_rows, selected_directory, snapshot
from variants import mutate


def check(previous, output, case_ids):
    """Compare current code with saved baseline and selected case trajectories.

    Args:
        previous: Path to an earlier complete run with policy, models and traces.
        output: New Path for this check's snapshots, logs and results.
        case_ids: Unique admitted case IDs from the earlier manifest; the
            baseline is always checked, so [] requests baseline-only comparison.

    Returns:
        Summary dictionary also written to compatibility.json on success.
        Fail on policy/model differences, incomplete trials or unequal traces.

    This function DOES run new simulations and subprocesses, including paired
    reference checks. It never rewrites previous. It checks selected cases only,
    not the entire historical sweep or a continuous property range.
    """
    previous, output = previous.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    load = lambda name: json.loads((previous / name).read_text(encoding="utf-8"))
    config = validate_config(load("config.json"))
    manifest = load("manifest.json")
    cases = {c["id"]: c for c in manifest["cases"]}
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case IDs")
    write_json(output / "config.json", config)
    policy = snapshot(output, config)
    if policy["checkpoint_sha256"] != digest(previous / "policy.zip"):
        raise ValueError("Policy changed; historical trace comparison requires the same weights")
    # Prove saved workers import the saved shared package, not the live editable one.
    entry = output / "sources/data/validation_testing/self_balancer_tests"
    probe = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); import model_adapter, robot_environments; "
        "from pathlib import Path; assert Path(robot_environments.__file__).resolve().is_relative_to(model_adapter.PROJECT); "
        "print(robot_environments.__file__)", str(entry)], capture_output=True, text=True, check=True)
    (output / "snapshot_import.log").write_text(probe.stdout, encoding="utf-8")
    root, _ = explicit_baseline()
    save_xml(root, output / "converted_reference.xml")
    reference = execute_job({"type": "reference", "source": str(WORLD),
        "converted": str(output / "converted_reference.xml"), "policy": policy,
        "config": config, "response": str(output / "reference_check.json")},
        output / "reference_worker", max(180, config["timeout_seconds"]))
    if reference["status"] != "completed":
        raise AssertionError(f"Reference check failed: {reference}")
    root = configured_baseline(root, config["model"])
    comparisons = []
    for case_id in ["baseline", *case_ids]:
        if case_id == "baseline":
            current_root = root
            old_xml = previous / "baseline.xml"
            old_worker = previous / "baseline_worker"
        else:
            case = cases[case_id]
            current_root, _ = mutate(root, config, case["values"])
            old_xml = previous / "variants" / case["xml"]
            if digest(old_xml) != case["sha256"]:
                raise AssertionError("Historical XML hash changed")
            old_worker = selected_directory(previous / "tests" / case_id)
        differences = compare_models(mujoco.MjModel.from_xml_path(str(old_xml)), compile_root(current_root))
        if differences:
            raise AssertionError(f"Compiled model changed for {case_id}: {differences}")
        scene = output / (case_id + ".xml")
        save_xml(current_root, scene)
        directory = output / case_id
        status = execute_job({"xml": str(scene), "xml_sha256": digest(scene), "policy": policy,
            "config": config, "case_id": case_id}, directory, config["timeout_seconds"])
        rows = read_rows(directory)
        if status["status"] != "completed" or sorted(r["seed"] for r in rows) != sorted(config["seeds"]):
            raise AssertionError(f"Incomplete check: {case_id}: {status}")
        matches = []
        for row in rows:
            seed = row["seed"]
            trace = f"trace_seed{seed}.jsonl"
            if digest(directory / trace) != digest(old_worker / trace):
                raise AssertionError(f"Historical trace changed: {case_id}, seed {seed}")
            if row["numerical"] != "valid_observed" or row["review_flags"]:
                raise AssertionError(f"Diagnostic failure: {case_id}, seed {seed}")
            if case_id == "baseline" and not row["balance"]:
                raise AssertionError("Baseline failed to balance")
            matches.append(seed)
        comparisons.append({"case_id": case_id, "matching_trace_seeds": matches,
                            "balance_successes": sum(r["balance"] for r in rows)})
        print(f"{case_id}: exact historical trace matches on {len(matches)} seeds", flush=True)
    result = {"passed": True, "previous_run": str(previous), "comparisons": comparisons,
              "reference_check_passed": True, "saved_shared_package_import_passed": True,
              "scope": "Selected cases only; no full sweep or continuous-range claim"}
    write_json(output / "compatibility.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.previous_run, args.output, args.cases), indent=2))
    except Exception as error:
        # Keep failed evidence visible without touching historical results.
        print(f"Compatibility check failed: {error}", file=sys.stderr)
        sys.exit(1)
