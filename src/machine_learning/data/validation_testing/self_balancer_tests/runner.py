"""Isolated frozen-policy evaluation and reproducible baseline qualification."""
from __future__ import annotations
import concurrent.futures
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import mujoco
import numpy as np

from model_adapter import (HERE, PROJECT, REFERENCE, ROBOT_ENVIRONMENTS, WORLD, POLICY,
                           ReferenceBaselineTrainingEnv, RobotBaseEnv, DEFAULT_REWARD_WEIGHTS, NeutralEnv,
                           compile_root, configured_baseline, digest, explicit_baseline,
                           save_xml, write_json)
from variants import generate, make_case

# Reuse the existing frozen SB3 loader, including hash and complete space checks.
sys.path.insert(0, str(HERE.parent / "inverted_pendulum_tests"))
from pretrained_policy import SB3Controller


def policy_metadata(checkpoint):
    return {"algorithm": "PPO", "checkpoint_path": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": digest(checkpoint), "deterministic": True, "device": "cpu",
            "normalization": "RobotBaseEnv observation scaling; no VecNormalize"}


def metrics_valid(env, observation, action, previous_time):
    flags = []
    for name, value in (("qpos", env.data.qpos), ("qvel", env.data.qvel), ("qacc", env.data.qacc),
                        ("observation", observation), ("action", action), ("actuator_force", env.data.actuator_force)):
        if not np.isfinite(value).all():
            flags.append("nonfinite_" + name)
    if not math.isfinite(env.data.time) or not math.isclose(env.data.time - previous_time, env.dt, rel_tol=1e-7, abs_tol=1e-9):
        flags.append("unexpected_time_advance")
    if np.any(env.data.warning.number):
        flags.append("mujoco_warning")
    return flags


def trial(env, controller, config, seed, trace_path):
    duration = config["seconds_per_seed"]
    total_steps = round(duration / env.dt)
    if total_steps < 1 or not math.isclose(total_steps * env.dt, duration, abs_tol=1e-9):
        raise ValueError("seconds_per_seed must be an integer number of control steps")
    thresholds = {"max_abs_qvel": 1000., "max_abs_qacc": 1e6, "max_contact_force_n": 1000., "max_penetration_m": .01}
    supplied = config.get("diagnostics", {})
    if set(supplied) - set(thresholds) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in supplied.values()):
        raise ValueError("Unknown or invalid diagnostic threshold")
    thresholds.update(supplied)
    resets, flags, review = 0, set(), set()
    first_fall = None
    longest = episode_steps = completed = saturated = 0
    peaks = {"pitch_deg": 0., "true_tilt_deg": 0., "abs_qvel": 0., "abs_qacc": 0.,
             "contact_force_n": 0., "penetration_m": 0., "travel_m": 0., "actuator_force_nm": 0.}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, Path(trace_path).open("w", encoding="utf-8") as trace:
        warnings.simplefilter("always")
        obs, _ = env.reset(seed=seed)
        initial_observation = obs.tolist()
        for step in range(total_steps):
            if not np.isfinite(obs).all():
                flags.add("nonfinite_observation")
                break
            action = controller.predict(obs)
            if not np.isfinite(action).all() or not env.action_space.contains(action):
                flags.add("invalid_policy_action")
                break
            previous = float(env.data.time)
            obs, reward, terminated, truncated, _ = env.step(action)
            completed += 1
            episode_steps += 1
            flags.update(metrics_valid(env, obs, action, previous))
            if flags:
                break
            pitch = float(env.get_pitch())
            rotation = env.data.body("robot_body").xmat.reshape(3, 3)
            tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
            force, penetration = 0., 0.
            for contact in range(env.data.ncon):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, contact, wrench)
                force = max(force, float(np.linalg.norm(wrench[:3])))
                penetration = max(penetration, -float(env.data.contact[contact].dist))
            if not math.isfinite(force):
                flags.add("nonfinite_contact_force")
                break
            current = {"pitch_deg": abs(math.degrees(pitch)), "true_tilt_deg": math.degrees(tilt),
                       "abs_qvel": float(np.max(np.abs(env.data.qvel))), "abs_qacc": float(np.max(np.abs(env.data.qacc))),
                       "contact_force_n": force, "penetration_m": penetration,
                       "travel_m": float(np.linalg.norm(env.data.qpos[:2])),
                       "actuator_force_nm": float(np.max(np.abs(env.data.actuator_force)))}
            for key in peaks:
                peaks[key] = max(peaks[key], current[key])
            for metric, threshold in (("abs_qvel", "max_abs_qvel"), ("abs_qacc", "max_abs_qacc"),
                                      ("contact_force_n", "max_contact_force_n"), ("penetration_m", "max_penetration_m")):
                if current[metric] > thresholds[threshold]:
                    review.add(threshold)
            limits = np.max(np.abs(env.model.actuator_forcerange), axis=1)
            saturated += int(np.any(np.abs(env.data.actuator_force) >= limits * .999))
            fell = bool(terminated or tilt > math.radians(50))
            trace.write(json.dumps({"step": completed, "elapsed_s": completed * env.dt, "episode": resets,
                "observation": obs.tolist(), "action": action.tolist(), "qpos": env.data.qpos.tolist(),
                "qvel": env.data.qvel.tolist(), "ctrl": env.data.ctrl.tolist(), "actuator_force": env.data.actuator_force.tolist(),
                "target_speed": env.target_wheel_speed, "target_yaw": env.target_yaw,
                "contact_count": int(env.data.ncon), "contact_force_n": force, "penetration_m": penetration,
                "pitch_deg": math.degrees(pitch), "true_tilt_deg": math.degrees(tilt), "fell": fell}, allow_nan=False) + "\n")
            longest = max(longest, episode_steps)
            if fell or truncated:
                if first_fall is None:
                    first_fall = completed * env.dt
                if step + 1 < total_steps:
                    resets += 1
                    episode_steps = 0
                    obs, _ = env.reset(seed=seed + resets * 1000003)
        python_warnings = sorted(set(str(w.message) for w in caught))
        if python_warnings:
            review.add("python_warning")
    return {"seed": seed, "numerical": "anomaly" if flags else ("valid_observed" if completed == total_steps else "incomplete"),
            "flags": sorted(flags), "review_flags": sorted(review), "python_warnings": python_warnings,
            "balance": first_fall is None and completed == total_steps and not flags,
            "simulated_seconds": completed * env.dt, "first_fall_s": first_fall, "resets": resets,
            "longest_episode_s": longest * env.dt, "saturation_fraction": saturated / max(1, completed),
            "peaks": peaks, "initial_observation": initial_observation,
            "wall_seconds": time.perf_counter() - started, "thresholds": thresholds,
            "diagnostic_coverage": "control-step endpoints (250 physics substeps); MuJoCo warning counters checked each action"}


class LegacyCopy(ReferenceBaselineTrainingEnv):
    def __init__(self, scene):
        # Match the renamed reference initialization, injecting only the XML path.
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        self.reward_terms = {}
        RobotBaseEnv.__init__(self, str(Path(scene).resolve()), render_mode=None)
        self.delay_target_speed = self.delay_target_yaw = self.pitch_offset = 0.


def reference_check(job):
    rows = []
    with warnings.catch_warnings(record=True) as setup_warnings:
        warnings.simplefilter("always")
        source, converted = LegacyCopy(job["source"]), LegacyCopy(job["converted"])
    try:
        controller = SB3Controller(job["policy"], source)
        for seed in job["config"]["seeds"]:
            np.random.seed(seed)
            left, _ = source.reset(seed=seed)
            np.random.seed(seed)
            right, _ = converted.reset(seed=seed)
            resets, max_difference = 0, 0.
            first_fall = None
            for step in range(round(job["config"]["seconds_per_seed"] / source.dt)):
                np.testing.assert_allclose(left, right, atol=1e-9, rtol=1e-9)
                a, b = controller.predict(left), controller.predict(right)
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
                previous = source.data.time
                left, _, done1, _, _ = source.step(a)
                right, _, done2, _, _ = converted.step(b)
                if metrics_valid(source, left, a, previous) or metrics_valid(converted, right, b, previous):
                    raise AssertionError("Legacy reference numerical failure")
                np.testing.assert_allclose(source.data.qpos, converted.data.qpos, atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(source.data.qvel, converted.data.qvel, atol=1e-9, rtol=1e-9)
                max_difference = max(max_difference, float(np.max(np.abs(source.data.qpos - converted.data.qpos))))
                assert done1 == done2
                if done1:
                    if first_fall is None:
                        first_fall = (step + 1) * source.dt
                    resets += 1
                    np.random.seed(seed + resets * 1000003)
                    left, _ = source.reset(seed=seed + resets * 1000003)
                    np.random.seed(seed + resets * 1000003)
                    right, _ = converted.reset(seed=seed + resets * 1000003)
            rows.append({"seed": seed, "max_qpos_difference": max_difference, "first_fall_s": first_fall,
                         "simulated_seconds_each": job["config"]["seconds_per_seed"], "passed": True})
            print(f"reference seed {seed}: matched", flush=True)
    finally:
        source.close()
        converted.close()
    return {"passed": True, "rows": rows, "setup_warnings": sorted(set(str(w.message) for w in setup_warnings))}


def worker(job):
    if job.get("type") == "reference":
        write_json(job["response"], reference_check(job))
        return
    directory = Path(job["directory"])
    if digest(job["xml"]) != job["xml_sha256"]:
        raise ValueError("XML hash mismatch")
    env = NeutralEnv(job["xml"], job["config"]["initial_pitch_rad"], job["config"]["clearance_m"])
    try:
        controller = SB3Controller(job["policy"], env)
        with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
            for seed in job["config"]["seeds"]:
                result = trial(env, controller, job["config"], seed, directory / f"trace_seed{seed}.jsonl")
                result.update(case_id=job["case_id"], policy_compatibility_notes=controller.compatibility_notes)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(f"{job['case_id']} seed {seed}: {result['numerical']}, balance={result['balance']}", flush=True)
    finally:
        env.close()


def execute_job(job, directory, timeout, on_update=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    job = {**job, "directory": str(directory.resolve())}
    request = directory / "request.json"
    write_json(request, job)
    started = time.monotonic()
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(HERE / "validate_self_balancer.py"), "_worker", str(request)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
        try:
            while proc.poll() is None:
                if time.monotonic() - started > timeout:
                    proc.kill()
                    proc.wait()
                    return {"status": "timeout", "seconds": timeout}
                if on_update:
                    on_update()
                time.sleep(.25)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return {"status": "completed" if proc.returncode == 0 else "worker_error", "exit_code": proc.returncode}


def selected_directory(directory):
    directory = Path(directory).resolve()
    selection = directory / "selected_attempt.json"
    if selection.exists():
        target = (directory / json.loads(selection.read_text(encoding="utf-8"))["directory"]).resolve()
        if not target.is_relative_to(directory) or target == directory:
            raise ValueError("Invalid retry attempt path")
        return target
    return directory


def read_rows(directory):
    path = selected_directory(directory) / "results.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a worker may be writing its last line
    return rows


def snapshot(directory, config):
    target = directory / "sources"
    target.mkdir()
    sources = [*HERE.glob("*.py"), *REFERENCE.joinpath("envs").glob("*.py"),
               *ROBOT_ENVIRONMENTS.glob("*.py"), WORLD, WORLD.parent / "robot-02.xml",
               HERE.parent / "inverted_pendulum_tests/pretrained_policy.py"]
    fingerprints = {}
    for path in sources:
        relative = path.relative_to(PROJECT)
        copy_path = target / relative
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy_path)
        fingerprints[str(relative)] = digest(path)
    checkpoint = Path(config.get("checkpoint", POLICY))
    if not checkpoint.is_absolute():
        checkpoint = PROJECT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"Reference PPO checkpoint not found: {checkpoint}. Supply checkpoint in the JSON request; models are not included in Git. Do not substitute the Control Agent checkpoint for a reference-policy comparison.")
    destination = directory / "policy.zip"
    shutil.copyfile(checkpoint, destination)
    metadata = policy_metadata(destination)
    metadata["original_checkpoint"] = str(checkpoint)
    write_json(directory / "provenance.json", {"source_hashes": fingerprints, "policy": metadata,
        "python": sys.version, "packages": {k: importlib.metadata.version(k) for k in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy", "scipy")},
        "reset_profile": "neutral_v1: seed-local RNG, correct wxyz, +/-0.02 rad pitch by default, random yaw, 2mm support clearance by default",
        "step_profile": "inherited BaseWorldEnv, 250 substeps; unchanged observations/action scaling"})
    return metadata


def run_experiment(config, output, workers=2):
    from reporting import summarize, refinement_candidates
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    policy = snapshot(directory, config)
    root, original = explicit_baseline()
    save_xml(root, directory / "converted_reference.xml")
    reference_job = {"type": "reference", "source": str(WORLD), "converted": str(directory / "converted_reference.xml"),
                     "policy": policy, "config": config, "response": str(directory / "reference_check.json")}
    status = execute_job(reference_job, directory / "reference_worker", max(180, config["timeout_seconds"]))
    if status["status"] != "completed":
        write_json(directory / "setup_error.json", status)
        (directory / "REPORT.md").write_text(f"# Self-balancer evaluation\n\nReference equivalence failed: {status}. See reference_worker/worker.log. No sweep ran.\n", encoding="utf-8")
        return directory
    root = configured_baseline(root, config["model"])
    save_xml(root, directory / "baseline.xml")
    manifest = generate(root, config, directory / "variants")
    write_json(directory / "manifest.json", manifest)
    rows, statuses = [], {}
    baseline_job = {"xml": str(directory / "baseline.xml"), "xml_sha256": digest(directory / "baseline.xml"),
                    "policy": policy, "config": config, "case_id": "baseline"}
    statuses["baseline"] = execute_job(baseline_job, directory / "baseline_worker", config["timeout_seconds"])
    baseline = read_rows(directory / "baseline_worker")
    write_json(directory / "baseline_results.json", baseline)
    passed = (statuses["baseline"]["status"] == "completed" and len(baseline) == len(config["seeds"]) and
              all(r["numerical"] == "valid_observed" and r["balance"] and not r["review_flags"] for r in baseline))
    if not passed:
        summarize(directory, rows, manifest, config, statuses, "baseline_failed")
        return directory
    print("Neutral baseline qualified; starting cases", flush=True)

    def evaluate(case):
        job = {"xml": str(directory / "variants" / case["xml"]), "xml_sha256": case["sha256"],
               "policy": policy, "config": config, "case_id": case["id"]}
        return execute_job(job, directory / "tests" / case["id"], config["timeout_seconds"])

    (directory / "tests").mkdir()
    def run_cases(cases):
        # Parent refreshes aggregate files after each flushed per-seed result.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate, case): case for case in cases if "xml" in case}
            previous_count = -1
            while futures:
                done, _ = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    case = futures.pop(future)
                    statuses[case["id"]] = future.result()
                    print(f"{case['id']}: {statuses[case['id']]['status']}", flush=True)
                current = []
                for case in manifest["cases"]:
                    current.extend(read_rows(directory / "tests" / case["id"]))
                if len(current) != previous_count or done:
                    rows[:] = current
                    summarize(directory, rows, manifest, config, statuses, "running")
                    previous_count = len(current)
    run_cases(manifest["cases"])
    remaining = config["max_refinement_cases"]
    while remaining > 0:
        proposals = refinement_candidates(rows, manifest, config)[:min(remaining, max(1, workers))]
        if not proposals:
            break
        cases = [make_case(root, config, value, "refinement", directory / "variants", len(manifest["cases"]) + i) for i, value in enumerate(proposals)]
        manifest["cases"].extend(cases)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "variants/manifest.json", manifest)
        run_cases(cases)
        remaining -= len(cases)
    summarize(directory, rows, manifest, config, statuses, "completed")
    return directory
