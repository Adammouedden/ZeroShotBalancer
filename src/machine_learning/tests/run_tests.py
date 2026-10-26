"""Run the control-agent observation tests, action tests, or both."""

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
        "--test-all",
        dest="suite",
        action="store_const",
        const="all",
        help="Run both test suites (default).",
    )
    parser.set_defaults(suite="all")
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments after --, for example: -- -v -x",
    )
    args = parser.parse_args(argv)

    test_directory = Path(__file__).resolve().parent / "robot_environments"
    selected_suites = ("observations", "actions") if args.suite == "all" else (args.suite,)
    test_files = [
        str(test_directory / f"test_env_{suite}.py")
        for suite in selected_suites
    ]
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    # A fresh module invocation avoids tests/robot_environments shadowing the
    # application's robot_environments package on this script's import path.
    return subprocess.run(
        [sys.executable, "-m", "pytest", *test_files, *pytest_args],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
