"""Retry incomplete cases with their saved source code; retain every attempt."""
import argparse
import concurrent.futures
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from model_adapter import digest, write_json
from reporting import summarize
from runner import read_rows


def retry_one(directory, case, status, seeds):
    original = directory / "tests" / case["id"]
    job = json.loads((original / "request.json").read_text(encoding="utf-8"))
    index = 2
    while (original / f"attempt_{index}").exists():
        index += 1
    attempt = original / f"attempt_{index}"
    attempt.mkdir()
    job["directory"] = str(attempt)
    request = attempt / "request.json"
    write_json(request, job)
    # Invoke the preserved CLI: its own directory and source tree supply the
    # exact adapter, observation/action code and loader used for the original run.
    entry = directory / "sources/data/validation_testing/self_balancer_tests/validate_self_balancer.py"
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if provenance["python"] != sys.version or any(importlib.metadata.version(name) != version for name, version in provenance["packages"].items()):
        raise ValueError("Python/dependency versions changed; use a new experiment instead of mixing environments")
    for relative, expected in provenance["source_hashes"].items():
        if digest(directory / "sources" / relative) != expected:
            raise ValueError("Preserved source hash mismatch: " + relative)
    started = time.monotonic()
    with (attempt / "worker.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen([sys.executable, str(entry), "_worker", str(request)], stdout=stream, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=job["config"]["timeout_seconds"])
            result = {"status": "completed" if proc.returncode == 0 else "worker_error", "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result = {"status": "timeout", "seconds": job["config"]["timeout_seconds"]}
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    actual = read_rows(attempt)
    if result["status"] == "completed" and sorted(r["seed"] for r in actual) != sorted(seeds):
        result["status"] = "incomplete_worker_output"
    record = {"attempt": index, "wall_seconds": time.monotonic()-started, "result": result,
              "source_entry_sha256": digest(entry), "uses_original_source_snapshot": True}
    write_json(attempt / "attempt_status.json", record)
    if result["status"] == "completed":
        write_json(original / "selected_attempt.json", {"directory": attempt.name})
    return {**result, "previous_attempt": status, "retry": record}


def retry(directory, workers=2):
    directory = Path(directory).resolve()
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    status, config, manifest = load("status.json"), load("config.json"), load("manifest.json")
    if status["state"] not in {"completed", "completed_with_errors"}:
        raise ValueError("Wait for the original batch to finish before retrying")
    cases = [c for c in manifest["cases"] if "xml" in c and
             (status["workers"].get(c["id"], {}).get("status") != "completed" or
              sorted(r["seed"] for r in read_rows(directory / "tests" / c["id"])) != sorted(config["seeds"]))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(retry_one, directory, c, status["workers"].get(c["id"], {}), config["seeds"]): c for c in cases}
        for future in concurrent.futures.as_completed(jobs):
            case = jobs[future]
            status["workers"][case["id"]] = future.result()
            rows = [r for c in manifest["cases"] for r in read_rows(directory / "tests" / c["id"])]
            summarize(directory, rows, manifest, config, status["workers"], "completed")
            print(case["id"], status["workers"][case["id"]]["status"], flush=True)
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    print("Cases retried:", retry(args.directory, args.workers))
