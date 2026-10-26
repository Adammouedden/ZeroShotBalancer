# Unit Tests

With uv installed, start from the repository root:

```bash
cd src/machine_learning
uv sync --locked
uv run python tests/run_tests.py                              # All tests (default)
uv run python tests/run_tests.py --test-observations           # Observations only
uv run python tests/run_tests.py --test-actions                # Actions only
uv run python tests/run_tests.py --test-actions -- -v          # Verbose pytest output
```

`--test-all` explicitly selects all tests. Pass pytest options after `--`; `-v` and `--verbose` enable verbose output, `-vv` adds more detail, and `-x` stops the run after the first failure. Combine them with `uv run python tests/run_tests.py --test-actions -- -v -x`. Use `uv run python tests/run_tests.py --help` for runner options.

For setup instructions, see the [developer quickstart](/docs/developer_guide/README.md#developer-quickstart). For test cases and scope, see the [unit test guide](/docs/unit_tests.md).
