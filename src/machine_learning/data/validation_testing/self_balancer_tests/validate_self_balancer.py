"""CLI for this self-balancer's boundary value and data mutation testing."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from bounds import calculate, validate_config
from model_adapter import compile_root, configured_baseline, describe, explicit_baseline, save_xml, write_json
from variants import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "calculate", "generate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=name in {"generate", "run"})
        if name == "run":
            command.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    hidden = commands.add_parser("_worker")
    hidden.add_argument("request", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        from runner import worker
        worker(json.loads(args.request.read_text(encoding="utf-8")))
        return 0
    if args.command == "report":
        from reporting import summarize
        from runner import read_rows
        d = args.directory
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        rows = [r for c in manifest["cases"] for r in read_rows(d / "tests" / c["id"])]
        summarize(d, rows, manifest, manifest["config"], status["workers"], status["state"])
        return 0
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if args.command == "run":
        from runner import run_experiment
        directory = run_experiment(config, args.output, args.workers)
        print(directory / "REPORT.md")
        status_file = directory / "status.json"
        return 0 if status_file.exists() and json.loads(status_file.read_text())["state"] == "completed" else 2
    root, _ = explicit_baseline()
    root = configured_baseline(root, config["model"])
    model = compile_root(root)
    if args.command == "generate":
        manifest = generate(root, config, args.output)
        save_xml(root, args.output / "baseline.xml")
        write_json(args.output / "bounds.json", manifest["bounds"])
        print(f"Generated {len(manifest['cases'])} cases in {args.output}")
    else:
        value = describe(model) if args.command == "inspect" else calculate(model, config)
        if args.output:
            write_json(args.output, value)
        print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
