"""Run the observation, action, and calibrated workflow test suites."""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    suites = parser.add_mutually_exclusive_group()
    suites.add_argument(
        "--test-observations",
        dest="suite",
        action="store_const",
        const="observations",
        help="Run only the observation unit tests.",
    )
    suites.add_argument(
        "--test-actions",
        dest="suite",
        action="store_const",
        const="actions",
        help="Run only the action unit tests.",
    )
    suites.add_argument(
        "--test-calibrated-workflow",
        dest="suite",
        action="store_const",
        const="calibrated-workflow",
        help="Run only the calibrated workflow tests.",
    )
    suites.add_argument(
        "--test-all",
        dest="suite",
        action="store_const",
        const="all",
        help="Run all three test suites (default).",
    )
    parser.set_defaults(suite="all")
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments after --, for example: -- -v -x",
    )
    args = parser.parse_args(argv)

    test_directory = Path(__file__).resolve().parent
    test_files = {
        "observations": test_directory / "robot_environments/test_env_observations.py",
        "actions": test_directory / "robot_environments/test_env_actions.py",
        "calibrated-workflow": test_directory / "workflows/test_calibrated_workflow.py",
    }
    # Control and reference training both import a package named envs; keep them isolated.
    if args.suite == "all":
        test_groups = [
            [test_files["observations"], test_files["actions"]],
            [test_files["calibrated-workflow"]],
        ]
    else:
        test_groups = [[test_files[args.suite]]]
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    # A fresh module invocation avoids tests/robot_environments shadowing the
    # application's robot_environments package on this script's import path.
    return_code = 0
    for group in test_groups:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *(str(path) for path in group), *pytest_args],
            check=False,
        )
        return_code = return_code or result.returncode
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
