# Continuous integration

## Purpose and scope

The [GitHub Actions workflow](../.github/workflows/ci.yml) validates the Python project on Ubuntu 24.04 with Python 3.13. It installs locked dependencies, compiles project sources, builds a source distribution and wheel, and runs the observation, action, and calibrated workflow tests. Firmware builds, full training runs, optional TensorFlow conversion, and deployment are outside its scope.

## Workflow configuration

- **Triggers:** pull requests targeting `main`, pushes to `main`, and manual dispatch. All paths trigger checks, including documentation changes.
- **Execution:** one `Python CI` job with a 30-minute timeout and read-only repository permissions. New runs cancel older runs for the same PR or branch.
- **Dependencies:** uv is pinned to `0.12.6`; Python is selected by [`.python-version`](../src/machine_learning/.python-version). `uv sync --locked --dev` installs runtime and test dependencies and fails if the manifest and lockfile disagree. The uv cache tracks `uv.lock`.
- **Compilation and packaging:** `compileall` checks project source directories without traversing third-party code in `.venv`. `uv build` writes temporary distributions outside the source tree; artifacts are not published.
- **Tests:** the [test runner](../src/machine_learning/tests/run_tests.py) executes observations/actions together and calibrated workflows in a separate process. This prevents the trainers' identically named `envs` packages from sharing Python's import cache. Failure in either group fails CI.

The calibrated workflow tests generate video, requiring rendering and encoding without a display. The runner installs `libosmesa6` and `ffmpeg` and sets:

| Variable | Value | Purpose |
| --- | --- | --- |
| `MUJOCO_GL` | `osmesa` | Render MuJoCo frames on the CPU without a display. |
| `PYOPENGL_PLATFORM` | `osmesa` | Use the same backend in PyOpenGL. |
| `IMAGEIO_FFMPEG_EXE` | `/usr/bin/ffmpeg` | Encode test videos using the installed FFmpeg executable. |
| `MPLBACKEND` | `Agg` | Precautionary noninteractive backend if Matplotlib is introduced; it is not currently in the lockfile. |

## Reproduce locally

From the repository root on Ubuntu 24.04, with uv 0.12.6 installed:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y libosmesa6 ffmpeg
cd src/machine_learning
uv python install
uv sync --locked --dev
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export MPLBACKEND=Agg
export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
uv run --locked python -m compileall -q main.py data evaluation tests training utilities
uv build --out-dir /tmp/zeroshotbalancer-dist
uv run --locked python tests/run_tests.py --test-all -- -v
```

These checks run without a physical robot, display server, or GPU. See the [test README](../src/machine_learning/tests/README.md) for individual suite commands.

## Maintenance

- Commit `pyproject.toml` and `uv.lock` together when changing dependencies.
- Update the workflow's uv pin intentionally; upgrading a local installation does not change CI. Review checkout/setup-uv action versions and the Ubuntu runner version periodically.
- Keep `.python-version` consistent with the project's supported Python versions. CI reads this file automatically.
- Extend the compilation command when adding new top-level Python source locations, and update the reproduction command above.
- Add new suites to the test runner explicitly: `--test-all` runs its listed suites rather than discovering every future test file. Preserve process isolation while both trainers import `envs`.
- Recheck system packages and rendering/encoding variables when changing simulation or recording dependencies.

## References

- [GitHub continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration)
- [uv with GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [MuJoCo rendering backends](https://gymnasium.farama.org/environments/mujoco/#rendering-backend)
- [ImageIO environment variables](https://imageio.readthedocs.io/en/stable/user_guide/envvariables.html)
