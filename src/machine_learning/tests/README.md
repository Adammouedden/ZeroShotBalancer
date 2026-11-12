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

`--test-all` explicitly selects all three suites. The default and `--test-all` run the observation/action tests together, then `workflows/test_calibrated_workflow.py` in a separate pytest process to keep the control and reference training imports isolated. The runner returns a nonzero exit code if either group fails.

Pass pytest options after `--`; they apply to each pytest process. `-v` and `--verbose` enable verbose output, `-vv` adds more detail, and `-x` stops each process after its first failure. Combine them with `uv run python tests/run_tests.py --test-actions -- -v -x`. Use `uv run python tests/run_tests.py --help` for runner options.

For setup instructions, see the [developer quickstart](/docs/developer_guide/README.md#developer-quickstart). For test cases and scope, see the [unit test guide](/docs/unit_tests.md).
