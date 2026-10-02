# Self-balancer implementation and evaluation

Status: **implemented and evaluated**, 2026-09-27. All four final studies completed and passed independent artifact audits.

## Compatibility after Adam's environment reorganization

Updated the adapter to `training/robot_environments/base_world_env.xml`, `BaseWorldEnv`, and `training/reference_baseline/envs/reference_baseline_training_env.py`. Reference checks initialize the new modular reward fields. Source snapshots now include the shared robot package, and saved workers load that snapshot ahead of the editable live package. The original six-input PPO checkpoint is retained; the Control Agent is not substituted. Existing JSON requests and historical results are unchanged.

**No full sweep was repeated.** Code/XML comparison found unchanged robot/world physics and unchanged headless control/observation behavior; upstream changes affect names, video annotations and reference reward bookkeeping. All 20 current self-balancer regression tests and 18 pendulum tests passed (38 total). The historical counts below remain the counts at the original evaluation date.

Targeted checks in adam_integration_check1 (`results/adam_integration_check1/REPORT.md`) passed: five original/converted reference comparisons, five neutral baseline trials, five passing horizontal-CoM trials and five failing joint-CoM trials, each for ten seconds. All 15 neutral-protocol trial traces matched their historical counterparts byte-for-byte on seeds 0-4. Compiled selected models matched, all 15 trials were numerically valid with no review flags, and saved-package import isolation passed. Baseline and horizontal case balanced 5/5; the joint case failed 0/5, exactly as before. Evidence is saved in `compatibility.json`, worker logs, source/policy snapshots and traces.

This supports retaining the earlier evaluations for this integration without repeating 725 property trials. It is a targeted compatibility check, not a new full range study. Changes to physics, policy weights, observations/actions, reset or evaluation criteria require renewed qualification and affected sweeps. A source rename alone does not require them when behavior is preserved and checked.

The implemented tool calculates conditional bounds for this self-balancer, generates boundary neighbours and incremental mutations, and evaluates permitted cases with the frozen Env01-v3 PPO. See the [usage guide](README.md) for configuration and commands.

## Completed checks

- 19 self-balancer regression tests passed, covering explicit-inertia preservation, effective CoM/force/contact mutations, resized-box bounds, reproducible neutral resets, cumulative duration after falls, numerical anomalies, timeouts/retries, exact nominal grid values and refinement across multiple transitions.
- All 18 existing inverted-pendulum regression tests passed. Their original commands and historical results remain intact.
- Original and converted legacy trajectories matched on five seeds before each started sweep. The neutral profile separately balanced all five baseline seeds for ten seconds each, with no numerical or review flags.
- All four final runs passed independent saved-artifact audits: checkpoint/source/XML hashes, complete seed coverage, finite traces, zero commanded targets and CSV/JSONL agreement. Each CoM study's zero-offset traces matched the neutral baseline byte-for-byte on all five seeds.
- A resized configuration (`results/resized_qualification1/REPORT.md`) also passed five ten-second neutral baseline trials. Its chassis is 11 x 5 x 20 cm at 0.75 kg, with 4 cm radius / 0.10 kg wheels and 16 cm track width. The calculated necessary fore-aft moment bound changes to +/-20.412 mm. This verifies the configuration path and reference policy qualification for that example; its nonzero CoM offsets were not swept.

## Results

All trials use ten cumulative simulated seconds, seeds 0-4, the same frozen PPO and a flat contact world. Ordinary falls trigger resets to finish the time budget. Full-duration balance requires no fall.

| Study | Completed trials | Numerically valid | Full-duration balance | Excluded configurations |
| --- | ---: | ---: | ---: | ---: |
| Motor torque capacity (`results/motor_run1/REPORT.md`) | 80 | 80 | 70 | 1 negative input |
| Forward/backward CoM (`results/com_fore_aft_run1/REPORT.md`) | 65 | 65 | 65 | 8 physically excluded |
| Vertical CoM (`results/com_vertical_run2/REPORT.md`) | 75 | 75 | 60 | 8 physically excluded |
| Combined CoM (`results/com_joint_run2/REPORT.md`) | 505 | 505 | 415 | 52 physically excluded |

Across these studies, **725/725 trials were numerically valid**, with **610 full-duration balance successes**, no review flags, and 7,250 simulated seconds in property trials. Reference qualification and the resized-baseline check are additional runs. These counts include only the selected final attempts.

The trial counts include extra boundary-neighbour and refinement cases, so they are not simply 11 samples times 5 seeds. Each excluded configuration is logged once without a policy rollout.

Motor capacity zero and 0.1% failed all five seeds, while the smallest tested all-seed success was 2.575%. That leaves an observed fail/pass bracket of 0.1%-2.575%; it is not an exact motor requirement. The requested refinement tolerance was 2.5 percentage points. Values at the upper search boundary and its neighbour (100.1%) are exploratory model tests, not hardware ratings.

For forward/backward CoM, all 13 permitted offsets balanced, including boundary points near +/-15.105 mm. The geometric outer extent is +/-18.5 mm; the fixed-inertia necessary moment bound excludes offsets beyond about +/-15.105 mm. The tests did not find a balance transition among the permitted sampled offsets. Passing the necessary bound does not establish a realizable mass distribution.

For vertical CoM at zero horizontal offset, the refined lower fail/pass bracket is **-69.062 to -68.731 mm** relative to the nominal height: all five seeds failed at the lower endpoint and all five balanced at the upper endpoint. This bracket is approximately 0.331 mm wide. The sampled positive offsets up to the necessary upper bound balanced. The completed vertical run passed the independent artifact audit, including exact zero-offset trace agreement on all five seeds.

The joint study found **(-14.8 mm fore-aft, -68.4 mm vertical) failed all five seeds**, although each offset passed all five when tested individually with the other coordinate nominal. This demonstrates why separate passing intervals cannot be combined into a guaranteed rectangle. Its 11-by-11 initial grid, boundary neighbours and 12 additional refinement cases leave 14 observed transition brackets; unresolved regions remain unknown. The refinement budget does not establish a complete continuous boundary.

The combined CoM balance map is saved locally as `results/com_joint_run2/balance_grid.svg`.

The map shows balance, not physical feasibility certification: green cells can still be unverified mass distributions. Grey X cells were mathematically excluded before simulation.

## Physical interpretation

The baseline is the **actual geometry-derived simulation**, converted to explicit inertials: chassis mass approximately 0.633 kg, whole robot approximately 0.822 kg. It does not silently adopt the unused 0.514 kg chassis inertial entry or claim a match to CAD/hardware.

For the current uniform box's fixed central inertia, necessary coordinate bounds are about 81.65% of each geometric half-size: +/-15.105 mm fore-aft and +/-69.810 mm vertically, relative to the nominal CoM. These are conditional mathematical exclusion bounds. Only zero offset is labeled `nominal_geometry_derived`; all admitted nonzero offsets remain `unverified`. No nonzero physically established CoM range is claimed.

Individual coordinate results hold the other coordinates fixed. A joint grid is required to test combinations, and its finite samples do not certify all intervening positions. Physical screening, numerical checks and balance outcomes remain independent.

## Scope and practical limits

- Implemented: inspection, calculation, generation, policy evaluation, separate reports, bounded refinement, joint CoM grid, configurable dimensions/masses, motor torque capacity, uniform chassis mass scaling, wheel-floor friction and wheel damping.
- CoM uses the chassis body frame: X left/right, Y forward/backward, Z vertical. It holds mass, central inertia and geometry fixed. It does not model moving a physical battery or payload.
- Motor controls retain Adam's velocity-servo semantics, speed limits and 200 Hz timing. Neutral evaluation removes changing speed commands and artificial pitch bias, corrects reset quaternion ordering, and resets derivative history. The inherited observation and action methods are reused.
- Numerical diagnostics inspect control-step endpoints and warning counters. They do not observe every intermediate physics state. Review thresholds are coarse heuristics, not hardware-calibrated limits.
- Balance is a ten-second survival result under the declared initial disturbances and fall criterion, not precision position holding or a guarantee for different policies, terrain, durations or hardware.
- Movable-component distributions, arbitrary robot types, assembly-wide CoM mutation and aerodynamic effects remain deferred. Mass/material/motor limits that geometry cannot supply require engineering inputs or explicitly declared search caps.

## Evidence and reproduction

This repository includes the evaluation summary, not the generated run directories or policy weights. Paths under `results/` below identify local artifacts and are not downloadable from this commit. Repeating an experiment requires the documented reference checkpoint; compatibility comparisons also require the earlier run artifacts.

Each run directory contains its config, typed bounds, manifest, copied XMLs, frozen policy, source snapshots and hashes, package versions, reference/baseline results, per-worker JSONL and full traces, summary CSV, transition brackets and report. Completed audits add `verification.json`. Joint runs also produce a balance-grid SVG.

The first vertical/joint attempts are preserved separately as `com_vertical_run1` and `com_joint_run1`. An independent audit found that floating-point grid construction produced a -1.39e-17 m offset in their intended zero-Z sample. The generator now snaps machine-scale zero roundoff before deduplication, with a regression check requiring exact zero and unchanged compiled CoM. The corrected evaluations use `run2`; the first attempts are not the final evidence for exact nominal reproduction.

Three corrected joint-study workers hit their wall-clock timeout. Their original partial logs are retained; new attempts used the original saved source tree, identical checkpoint and unchanged Python/dependency versions. All three retries completed successfully. The aggregate selects one complete attempt per case/seed, and the final audit checks all 505 traces. Later retry/report-reader changes are recorded as current-source differences from the immutable original snapshots; they do not replace the saved evaluation implementation.

Use a new output directory to repeat an experiment; existing run directories are never overwritten. Exact CLI commands and supported settings are in [README.md](README.md). The evaluation work performed no policy training or physical hardware actions.
