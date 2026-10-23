import pathlib
import sys

# STR-90: sb_rl.py is a script, not part of the installed packages (see
# [tool.hatch.build.targets.wheel] in pyproject.toml), so its directory goes on the path to let
# tests import the observation and action labels it prints with.
CONTROL_AGENT_DIR = pathlib.Path(__file__).parents[1] / "training" / "pre_training" / "control_agent"
sys.path.insert(0, str(CONTROL_AGENT_DIR))
