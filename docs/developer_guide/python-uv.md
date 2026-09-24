# Python Guide to uv 
## Motivations
uv is an extremely fast Python package and project manager that is written in Rust. 
It replaces the `pip` command, being anywhere from 10 to 100x faster, and removes the need to create and source a virtual environment.

## Installation
First, you can install uv with the official installer.  
For **bash**:

    curl -LsSf https://astral.sh/uv/install.sh | sh

For **powershell**:
    
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"


Additionally, you can also use `pip`, though the above methods are preferred.  
```
pip install uv
```

## Projects
When in a project for the first time, you must first initialize uv in order to use it.
- For a current directory, like being in a cloned Git repo:  
```
uv init
```

- For a new directory:  
```
uv init my-project
```

## Adding Packages (replacing `pip`)
```
uv add <package>
```

For example, if running my python script requires `torch`, I can use:  
```
uv add torch
```

## Running a file
With uv, you no longer need to run a python script with `python ./main.py`/`python3 ./main.py` and a pre-activated virtual environment.  
Instead, you can use:  
```
uv run main.py
```

Doing so will automatically create a `.venv` folder if it doesn't already exist (no more `python -m venv .venv`!).

The convention for running a script with uv is the same as doing so with a normal python script, so running working code could also look like:  
```
uv run src/machine_learning/data/validation_testing/inverted_pendulum_tests/pendulum.py
```

## Lock and Sync
To lock down the entire dependency tree, primarily to ensure that future installations are deterministic and reproducible (similar to pip freeze), run:  
```
uv lock
``` 

This resolves the project dependencies in your `pyproject.toml` and pins the package versions into a cross-platform lockfile `uv.lock`.

Similarly, when wanting to match your project config to the previously-created `uv.lock` lockfile, run:  
```
uv sync
```

This creates a virtual environment, installs missing packages, updates changed ones, and removes unneeded ones so your environment matches the lockfile precisely.

It should be noted the above commands are typically automatically ran with many usual commands such as `uv run`.

## Keeping things clean
It may be inevitable that we run into Python version mismatches. `uv` helps keep our environments and dependency trees clean by managing these for you.

### Python Version Management
If you need a specific version of Python for a project, you can initialize it pinned to that version (so others know to use it too):
    
    uv init --python 3.11 my-project
To just install a specific Python version to your system:

    uv python install 3.11

### Development Dependencies
If needing to write validation tests, you may want to avoid shipping these to production or otherwise bloating the primary dependency tree.

To add development-specific packages, use the `--dev` flag:

    uv add --dev pytest ruff
This ensures your `pyproject.toml` keeps development tools strictly separated from our core ML dependencies.

## Addendum
The development of this guide is tracked by this [Jira ticket](https://ucf-team-xx2ob1z2.atlassian.net/jira/software/projects/STR/boards/3/timeline?selectedIssue=STR-69).

For more information, and the source of this guide, visit the official uv installation documentation: https://docs.astral.sh/uv/#installation
