# Unit Tests

With uv installed, start from the repository root:

```bash
cd src/machine_learning
uv sync --locked
uv run python tests/run_tests.py                              # All tests (default)
uv run python tests/run_tests.py --test-observations           # Observations only
uv run python tests/run_tests.py --test-actions                # Actions only
uv run python tests/run_tests.py --test-calibrated-workflow    # Calibrated workflows only
uv run python tests/run_tests.py --test-actions -- -v          # Verbose pytest output
```

The default and `--test-all` run the observation/action tests together, the calibrated workflow tests in a separate process, and all tests under `tests/golden_eval/`. This keeps the control and reference training imports isolated. The runner returns a nonzero exit code if any group fails.

Pass pytest options after `--`; they apply to each pytest process. `-v` and `--verbose` enable verbose output, `-vv` adds more detail, and `-x` stops each process after its first failure. Combine them with `uv run python tests/run_tests.py --test-actions -- -v -x`. Use `uv run python tests/run_tests.py --help` for runner options.

For setup instructions, see the [developer quickstart](/docs/developer_guide/README.md#developer-quickstart). For test cases and scope, see the [unit test guide](/docs/unit_tests.md).

For the full CI checks and headless rendering setup, see [Continuous integration](/docs/ci.md).
