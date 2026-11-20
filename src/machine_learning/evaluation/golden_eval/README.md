# Golden Evaluation Data Generation

generate_eval.py creates schedules, example usage:
``` uv run generate_eval.py .... ```

A schedule JSON is `[start time s, target wheel speed, target yaw]` per entry, held until the
next entry starts. Both target columns are in the units the policy reads:

| column | unit | read against |
| --- | --- | --- |
| target wheel speed | **rad/s** (mean of the two wheels) | `RobotBaseEnv.get_wheel_speed()` |
| target yaw | **radians**, absolute heading wrapped to ±π | `ControlAgentBaseEnv.get_yaw()` |

Body speed in m/s is `wheel rad/s × 0.0325` (the 65 mm wheel radius). A negative speed is a
reversal: the robot keeps its heading and drives backwards.

utils.py contains many things, one of which is a visualizer for schedules
```
cd src/machine_learning/evaluation/golden_eval
python utils.py visualize                          # schedules/test.json, 25 m² → plots/test.png
python utils.py visualize -s test.json -a 25 -bt 0.3 --labels --show
python utils.py check -s test.json -a 9            # numbers only, exits 1 if the path escapes
```

It integrates the *commanded* path (perfect tracking, from the origin) and draws the boundary
square plus one vector arrow per segment, shaded by commanded wheel speed. It reports boundary
breaches and targets past the actuator limit, so a schedule can be checked before it is run.

| flag | default | what it does |
| --- | --- | --- |
| `-a, --area` | `25` | boundary area in m², drawn as a square of `±sqrt(area)/2` |
| `-bt, --boundary-threshold` | `0` | safety margin in m, drawn as an inner dashed square |
| `-wr, --wheel-radius` | `0.0325` | converts wheel rad/s into body m/s |
| `-al, --actuator-limit` | `20.6` | flags targets past this as unreachable (`robot_calibrated.xml` ctrlrange) |
| `-u, --units` | `rad` | set `mps` for legacy schedules whose speed column is body m/s |
| `--yaw-degrees` | off | for legacy schedules whose yaw column is in degrees |
| `--labels` / `--show` / `--dpi` | | annotate waypoint times / open a window / output resolution |
