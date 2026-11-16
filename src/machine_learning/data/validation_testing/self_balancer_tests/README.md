# Self-balancer boundary value and data mutation testing

This tool calculates conditional bounds and evaluates property changes for the **two-wheel self-balancer** in Adam's Env01 world, using the existing frozen PPO. It supports different dimensions and masses of this topology. It does not infer universal physical or controller limits from dimensions alone.

Run from `src/machine_learning` with the existing uv environment. The commands below do not train a policy or modify the training files. Generated files and results are locally ignored; choose a new output directory for each experiment.

## Quick start

Inspect the actual compiled configuration:

```powershell
uv run python data/validation_testing/self_balancer_tests/validate_self_balancer.py inspect --config data/validation_testing/self_balancer_tests/configs/com_fore_aft.json
```

Calculate forward/backward CoM bounds without simulation:

```powershell
uv run python data/validation_testing/self_balancer_tests/validate_self_balancer.py calculate --config data/validation_testing/self_balancer_tests/configs/com_fore_aft.json
```

Generate XMLs, boundary exclusions and a manifest without policy trials:

```powershell
uv run python data/validation_testing/self_balancer_tests/validate_self_balancer.py generate --config data/validation_testing/self_balancer_tests/configs/com_fore_aft.json --output data/validation_testing/self_balancer_tests/generated/my_com_cases
```

Run baseline qualification, cases and bounded transition refinement:

```powershell
uv run python data/validation_testing/self_balancer_tests/validate_self_balancer.py run --config data/validation_testing/self_balancer_tests/configs/com_fore_aft.json --output data/validation_testing/self_balancer_tests/results/my_com_run --workers 2
```

Use `com_vertical.json`, `com_joint.json` or `motor_strength.json` for the other supplied requests. A joint grid includes all selected combinations, including the nominal-coordinate slices; it does not assume independent passing intervals form a passing rectangle. Up to four worker processes can run concurrently. Each case worker runs all requested seeds, with a per-case timeout; per-seed logs survive a later worker error.

`resized_com_fore_aft.json` supplies an illustrative second dimensions/masses configuration. Its saved evaluation qualifies only the resized neutral baseline; running the full request performs its nonzero CoM sweep as a new experiment.

Rebuild a report from saved logs:

```powershell
uv run python data/validation_testing/self_balancer_tests/validate_self_balancer.py report data/validation_testing/self_balancer_tests/results/my_com_run
```

## Inputs and supported calculations

The named `env01_neutral_compiled` profile selects `training/robot_environments/base_world_env.xml`, its included `robot-02.xml`, and `training/reference_baseline/models/Env01-v3_PPO/best_model.zip`. The profile name remains unchanged so existing requests continue to work. The adapter inherits Adam's renamed `BaseWorldEnv`; reference conversion checks use `ReferenceBaselineTrainingEnv`. Paths in an optional `checkpoint` override are relative to `src/machine_learning`, unless absolute. The default has six observations, two wheel-velocity-increment actions, 250 physics steps per action and a 5 ms control interval.

Run `uv sync` after pulling the shared-package changes. Reference checkpoints are now ignored and are not supplied by a fresh Git clone. Obtain the original PPO artifact or set `checkpoint` explicitly; our historical artifact has SHA256 `9bb72b08f45be7e043d000327cf79a67bb45914826685aa7380a5afd2c620f82`. A newly trained policy defines a new experiment. The new Control Agent is not a drop-in replacement for the six-observation reference policy.

| Property name | Input / interpretation |
| --- | --- |
| `center_of_mass` | Chassis only. Set `direction` to `fore_aft` (body Y), `left_right` (X), or `vertical` (Z). Alternatively set `directions` to two axes with `sampling: "grid"`. Only `inertia_mode: "fixed_central"` is currently supported. |
| `motor_strength` | Torque-cap fraction of the reference +/-0.65 N m. Default search is 0 to 1, an exploratory interval. Zero disables the servo gain and velocity feedback, producing zero motor force. |
| `mass_scale` | Positive chassis-mass factor; scale central inertia by the same factor. Supply `search_range`. |
| `friction` | Nonnegative wheel-floor sliding friction; updates both explicit contact-pair tangential coefficients. Supply `search_range`. |
| `damping` | Nonnegative damping on both wheel hinges in N m s/rad. Supply `search_range`. |
| `chassis_width`, `chassis_depth`, `chassis_height` | Full dimensions in meters. Recompute uniform box inertia at fixed mass and retain chassis bottom height. Width's upper clearance bound follows the wheel spacing; depth/height need a declared search interval. |
| `wheel_radius` | Radius in meters; moves the wheel axle with the radius, updates cylinder inertia at fixed mass, and recalculates reset placement. Supply `search_range`. |

For example, add this to a request to calculate bounds for a resized reference robot:

```json
"model": {
  "chassis_size_m": [0.11, 0.05, 0.20],
  "chassis_mass_kg": 0.75,
  "wheel_radius_m": 0.04,
  "wheel_thickness_m": 0.026,
  "wheel_mass_kg": 0.10,
  "track_width_m": 0.16
}
```

These are illustrative inputs, not hardware recommendations. Dimensions are full chassis sizes and full wheel thickness; MJCF half-sizes are handled internally. Resizing assumes uniform body geometry, keeps mass unless supplied, and checks lateral chassis/wheel clearance. This simple topology check is not a universal collision or manufacturing feasibility proof. Every resized reference must qualify with the frozen policy before its sweep can run.

Other useful settings:

- `points`: at least 11 per axis. For CoM, `step_mm` can instead impose a maximum initial spacing; endpoints and zero are included. Joint requests can specify per-axis step dictionaries.
- `seconds_per_seed`: cumulative simulation duration; defaults to 10. Must be an integer multiple of the 5 ms control interval.
- `seeds`: defaults to `[0, 1, 2, 3, 4]`. Repeated episodes use `seed + episode_index * 1000003`.
- `boundary_epsilon`: neighbour spacing in the property's base units (meters for CoM). Boundary neighbours surround geometry/moment exclusion edges or scalar search/domain edges; invalid candidates are logged without a rollout.
- `max_cases`: generation budget, including boundary cases; defaults to 400. `max_refinement_cases` separately caps extra evaluations, defaults to zero.
- `refine_to_mm` (CoM) or `refine_to` (other properties): desired bracket width, subject to the refinement budget. Refinement follows every sampled change in seed outcomes, rather than assuming one monotonic threshold.
- `timeout_seconds`: worker budget for one complete case, including policy loading and all seeds; defaults to 180 wall seconds.
- `initial_pitch_rad`, `clearance_m`: neutral reset parameters; defaults to +/-0.02 rad and 2 mm above geometric floor contact. Changing these defines a different evaluation protocol.
- `diagnostics`: optional positive review thresholds `max_abs_qvel` (1000), `max_abs_qacc` (1e6), `max_contact_force_n` (1000), `max_penetration_m` (0.01). These defaults are coarse, explicitly uncalibrated heuristics. They do not establish real hardware safety.

Unknown fields, unsupported body/distribution modes, invalid values and ambiguous directions fail explicitly. Calculation returns `requires_input` where a finite upper search bound cannot be inferred. Medium/wind effects, movable-component mass distributions, arbitrary shapes, and assembly-wide CoM mutation are not implemented.

## Scientific interpretation

The original world uses `inertiafromgeom="true"`. Our self-contained copies use **complete explicit inertials** with `auto`, preserving the actual compiled mass/CoM/inertia initially. The chassis therefore starts at about 0.633 kg, rather than silently switching to its unused written 0.514 kg. Compilation checks prove that requested CoM offsets apply while mass, central inertia and geometry remain fixed.

CoM geometry provides an outer support interval. Given mass `m` and the central inertia tensor `I` in body coordinates, the mass covariance is `C = trace(I)/(2m) * identity - I/m`. A necessary box-support condition is `C_ii <= (upper_i - com_i) * (com_i - lower_i)` for each coordinate. Violations prove exclusion under those assumptions. Passing these checks does **not** prove that a nonnegative physical mass distribution realizes the requested full moments.

Accordingly, only the unchanged CoM has `nominal_geometry_derived` status. Nonzero CoM points are either `known_infeasible` (excluded) or `unverified`. A simulation that balances is not thereby physically validated. Modeling an actual battery movement requires a component model that updates inertia too.

One-axis CoM ranges hold other coordinates fixed. Joint runs evaluate each tuple and save a balance map. Percentages use the nominal geometric half-size for each axis, not the nominal CoM coordinate. Simulations report sampled behavior and transition brackets, not exact continuous bounds or hardware guarantees.

## Evaluation protocol and outputs

Before each sweep, paired legacy runs compare original versus converted model states, observations and PPO actions on the requested seeds/duration. The neutral adapter is then independently qualified: all baseline seeds must survive the full duration without numerical or review flags. A failed baseline stops property rollouts and is reported, without retraining or silently switching policies.

The neutral adapter inherits Adam's observations and action/step methods, with zero targets, no artificial pitch bias, corrected quaternion ordering and reset derivative history. Falls use the existing 50-degree pitch limit plus true body tilt above 50 degrees. Balance here means full-duration survival; position drift is measured, not classified as a fall.

Finite-state checks, warning counters and contact diagnostics operate at **control-step endpoints**, not every physics substep. Unexpected time changes and nonfinite values are numerical anomalies; large finite measurements are separate review flags. Falls trigger immediate seeded resets to complete the time budget.

Each run preserves XMLs, the PPO, source snapshots/hashes and package versions; writes immediate per-test JSONL, worker logs and full action/state/contact traces; and updates `summary.csv` and `REPORT.md`. `transitions.json` stores conditional transition brackets. Joint CoM runs include `balance_grid.svg`. Timeout/crash rows remain visible as untested rather than successful trials.

## Regression checks

After a rename or integration change, rerun regression checks and compare selected trajectories with a saved run before deciding whether full sweeps need repeating. The following checks the reference conversion, the five-seed neutral baseline, a passing horizontal case, and a failing combined CoM case against the saved joint evaluation:

```powershell
uv run python data/validation_testing/self_balancer_tests/check_compatibility.py --previous-run data/validation_testing/self_balancer_tests/results/com_joint_run2 --output data/validation_testing/self_balancer_tests/results/my_integration_check --cases case_0016 case_0012
```

Choose a new output directory. This saves current sources, policy, logs, traces and `compatibility.json`; historical results remain untouched. It also verifies that saved workers import their saved shared robot package rather than the live editable installation. Case IDs belong to the supplied previous run; omit `--cases` for baseline-only qualification. Exact trace matches support compatibility for these cases, not a new exhaustive evaluation. Changed physics, policy, observation/action semantics, or unexplained mismatches require investigating and repeating affected evaluations.

```powershell
uv run python -m unittest discover -s data/validation_testing/self_balancer_tests -p test_self_balancer.py -v
```

The tests cover compiled equivalence, applied mutations, analytical box bounds across dimensions, contact/actuator/passive-force effects, invalid inputs, seeded reset conventions, cumulative duration after falls, anomaly checks, timeout classification and nonmonotonic refinement. Test-only zero-action fixtures exercise failure paths; actual study runs always use the supplied frozen PPO.

Audit a completed run independently, without rerunning physics:

```powershell
uv run python data/validation_testing/self_balancer_tests/verify_evaluation.py data/validation_testing/self_balancer_tests/results/my_com_run
```

This checks artifact hashes, compiled CoM mutations, full seed/time coverage, finite traces, zero targets and CSV/JSONL agreement. An unchanged CoM sample must reproduce the neutral baseline traces exactly. `verification.json` also identifies current source files that differ from the preserved run snapshot; historical results retain the implementation that produced them.

After a batch ends, retry only incomplete or interrupted cases with:

```powershell
uv run python data/validation_testing/self_balancer_tests/retry_evaluation.py data/validation_testing/self_balancer_tests/results/my_com_run --workers 2
```

Retries execute the original saved source tree and reject changed Python/dependency versions. They create new attempt directories and preserve earlier logs. `selected_attempt.json` selects a successful replacement for aggregation and auditing; completed unaffected cases are not rerun. Retries do not extend the declared refinement budget.

The [implementation plan](IMPLEMENTATION_PLAN.md) records the broader design and deferred extensions. Evaluation findings are in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).
