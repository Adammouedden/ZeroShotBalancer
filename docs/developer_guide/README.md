# Developer Guide

This directory contains onboarding instructions, best practices, and development guides for Zero-Shot Balancer. See [Python and uv](/docs/developer_guide/python-uv.md) for the full package-management guide.

## Developer Quickstart

### 1. Install uv

Use the [official installer](https://docs.astral.sh/uv/getting-started/installation/) for your platform.

**macOS / Linux (bash):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal after installation, then check that `uv --version` works.

### 2. Set up the Python project

From the cloned repository's root:

```bash
cd src/machine_learning
uv python install 3.13
uv sync --locked
```

The project selects Python 3.13 in [`.python-version`](/src/machine_learning/.python-version). It already contains [`pyproject.toml`](/src/machine_learning/pyproject.toml) and [`uv.lock`](/src/machine_learning/uv.lock), so initialization is complete; `uv init` is for creating a new project.

`uv sync --locked` creates the project's `.venv` and installs the locked dependencies, including development tools such as pytest. It reports an error if the lockfile needs updating. Run it again after pulling dependency changes. See [uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/) for details.

### 3. Run tests and scripts

Keep the terminal in `src/machine_learning` and use `uv run` to run commands in the project's environment. Manual virtual-environment activation is unnecessary.

Run the observation and action tests:

```bash
uv run --locked pytest -v tests/robot_environments/test_control_agent_base_env.py
```

View the control-agent command-line options:

```bash
uv run --locked python training/pre_training/control_agent/sb_rl.py --help
```

`uv run` automatically checks and syncs the environment before running a command. The `--locked` flag keeps the lockfile unchanged. See [Unit tests](/docs/unit_tests.md) for test scope and cases, and the [Control Agent README](/src/machine_learning/training/pre_training/control_agent/README.md) for training instructions.

### 4. Manage dependencies

Run these commands from `src/machine_learning`, replacing `<package>` with the package name:

| Task | Command |
| --- | --- |
| Add a runtime dependency | `uv add <package>` |
| Add a development dependency | `uv add --dev <package>` |
| Resolve dependencies after manually editing `pyproject.toml` | `uv lock` |
| Sync the environment with the updated lockfile | `uv sync --locked` |

`uv add` updates the project configuration, lockfile, and environment. Commit changes to `pyproject.toml` and `uv.lock` together so other developers can reproduce the environment.
