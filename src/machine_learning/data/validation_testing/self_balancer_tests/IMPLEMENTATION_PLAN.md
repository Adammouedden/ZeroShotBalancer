# Self-balancer boundary and mutation testing plan

Status: **core implementation evaluated**, 2026-09-27. The user authorized implementation and simulation evaluations; all four final studies and their artifact audits completed. See [README.md](README.md) for the implemented interface and supported scope, and [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the findings. The design below also describes deferred extensions, especially component/payload distributions and medium effects. Original training files and earlier pendulum results remain unchanged.

## 1. Recommended baseline

Historical planning notes below describe the pre-implementation layout. The implementation is complete; see README.md and EVALUATION_REPORT.md for current commands and results. After Adam's rename, `env01_v1.xml` is `training/robot_environments/base_world_env.xml`, shared robot code lives in that directory, and reference training code/checkpoints live under `training/reference_baseline/`. The neutral adapter now inherits `BaseWorldEnv`; the same original Env01-v3 PPO is retained.

Original selection: use the **flat world in `env01_v1.xml` together with its included `robot-02.xml`**, and the existing **Env01-v3 PPO checkpoint**. Build a neutral evaluation adapter using the shared Env01 behavior with zero speed/yaw targets. Preserve the original training files.

| Component | Selected source, relative to `src/machine_learning` | Reason |
| --- | --- | --- |
| World | `training/reference-baseline/envs/env01_v1.xml` | Flat floor, Earth gravity, no thrown objects; already includes the robot and wheel material. |
| Robot | `training/reference-baseline/envs/robot-02.xml` | Two-wheel self-balancer used by the available policy's environment. |
| Control/observation implementation | `training/reference-baseline/envs/RobotBaseEnv.py` and `env01_v1.py` | Provides the six policy inputs and two wheel-speed-increment actions. |
| Frozen policy | `training/reference-baseline/models/Env01-v3_PPO/best_model.zip` | Actual local PPO artifact; loads on CPU and matches the environment spaces. |

The directories inspected were `data/mjcf` and `training/reference-baseline/envs`; there is no separate `training/envs` model directory in the current checkout.

**Neutral means a documented nominal configuration, not zeroing all physical properties.** Keep flat ground, nominal wheel friction and damping, Earth gravity, original motor limits, no wind, and zero aerodynamic density/viscosity. Set commanded speed and yaw to zero and disable the artificial pitch measurement offset. Retain small, seeded initial disturbances so balancing is meaningfully tested.

### Why not the other models?

| Candidate | Inspection finding | Decision |
| --- | --- | --- |
| `data/mjcf/balancing_starter.xml` | Compiles as a complete robot/world. It has different body/joint names, wheel axes, torque motors and timestep. A stray `]` after the compiler element is present but did not prevent compilation. | Useful separate starter, but no demonstrated observation/action mapping to the supplied policy. Do not use it for the first policy-based validation baseline. |
| `env03_v1.xml`, used by Env03-v2 | Includes the robot plus a free moving block; its environment throws the block at the robot. It also lacks Env01's explicit wheel-floor contact-pair settings. | A later disturbance experiment, not the neutral baseline. Removing the block alone would not reproduce Env01 contact settings. |
| `robot-02.xml` by itself | Does not compile standalone: it refers to `wheel_material` supplied by the world. It also has no floor. | Treat it as an include, not an independent simulation scene. |

The checkpoint is PPO for this robot, not the Farama SAC policy from the inverted-pendulum study. No Env03-v2 checkpoint or separate observation-normalization statistics were found in the inspected models directory.

## 2. Findings that must be addressed first

### A. The compiled mass properties differ from the written inertial values

`env01_v1.xml` sets `inertiafromgeom="true"`. Compilation therefore derives mass, CoM and inertia from geometry, overriding the robot file's inertial elements. The following values were measured from the compiled model:

| Property | Written in the robot's inertial element | Used by the compiled scene |
| --- | --- | --- |
| Chassis mass | 0.514 kg | 0.6327 kg |
| Each wheel mass | 0.032 kg | about 0.0944 kg |
| Chassis local CoM height | 0.057 m | 0.0995 m |
| Total robot mass | 0.578 kg if those written masses were used | about 0.8215 kg |

**Decision for the first baseline:** preserve the actual compiled physics of the current reference scene. Do not silently switch to the written masses. A CAD-based or intended-mass model is a separate baseline that needs its own policy qualification.

For later inertial mutations, generate a separate copy with explicit full-precision mass, inertial position, principal inertia and quaternion for every moving body, and use `inertiafromgeom="auto"`. Complete all inertial definitions before switching modes: the source elements omit inertia values. Prove the unmodified conversion preserves compiled physics and deterministic trajectories before perturbing it. See [MuJoCo compiler documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html#compiler-inertiafromgeom).

User confirmed that adaptations belong in our validation folder. The evaluation adapter and generated world/robot copies live under `self_balancer_tests/`; the training originals remain source references. The generator uses complete explicit inertial definitions for moving bodies and checks compiled mutations. With complete explicit inertials, `auto` does not replace them with geometry-derived values. This approval resolves where to make the changes; it does not select new CAD masses in place of the current compiled baseline.

### B. The policy has a specific control contract

| Item | Inspected behavior |
| --- | --- |
| Observations | Six normalized values: pitch, finite-difference pitch rate, left/right wheel speed, speed-target error and wheel-speed-difference yaw-target error. |
| Actions | Two values in [-1, 1], converted to target wheel velocities as current wheel velocity plus action times 4 rad/s. They are not direct motor torques. |
| Actuation | Velocity servos with gain 4, target limits about +/-78.54 rad/s and torque limits +/-0.65 N m. |
| Physics/control timing | Physics step 20 microseconds; 250 substeps per action; control interval 5 milliseconds (200 Hz). |
| Axes | X is across the axle, Y is fore-aft, Z is vertical. The code's balance pitch is rotation about X. Left/right hinge signs differ. |
| Existing fall rule | Absolute reported pitch above 50 degrees. A stricter operational tilt criterion must be separately named and logged. |

Do not transplant the pendulum's X-axis meaning, four-observation interface, one action, gear-based strength sweep or 0.04 s control interval. Do not increase the robot timestep to speed up testing without a separate convergence and policy-compatibility check.

Loading and space checks passed, and the checkpoint returned finite in-range actions after reset in both Env01-v3 and the unscheduled Env01 base class. **Ten-second balance and robustness have not yet been qualified.** The checkpoint records 180,000 training timesteps; its name and successful loading do not establish its performance.

### C. Training conditions need an explicit neutral evaluation mode

Env01-v3 changes target speed after 1, 3, 4.5 and 5.5 seconds and randomizes a pitch offset of about +/-2 degrees. Those should not vary during an isolated physical-property test. Use the Env01 base step/observation contract with zero targets and no injected pitch bias; do not use the demo video's changing command schedule.

The reset implementation also needs an audit:

- It mixes `self.np_random` and global `np.random`; an environment seed alone does not reproduce all reset randomness.
- It writes SciPy's default `(x,y,z,w)` quaternion directly into MuJoCo's `(w,x,y,z)` state. The reset comments and intended rotations therefore do not describe the implemented convention reliably.
- Pitch-rate observations keep `last_time` and `last_pitch`; define and clear this history at every reset. Calling the observation builder more than once at the same timestamp can also change the derivative observation, so logging must use cached observations.
- The original reset pins root Z to zero, while the floor is at -0.02 m. A configurable geometry needs a computed, nonpenetrating initial placement and a declared clearance/settling protocol.
- Environment construction emits float32 Box-bound precision warnings. Correct bound-array dtypes in the adapter or classify those exact setup warnings separately; do not suppress arbitrary runtime warnings.

Keep an exact legacy-reference mode to reproduce the existing environment, seeding both RNGs. Establish the neutral/corrected reset protocol in a separately named profile and requalify the same frozen policy. Do not silently alter the reference and attribute behavior changes to a physical parameter.

The environment hardcodes `env01_v1.xml`, so the adapter must accept an explicit scene path. Preferred approach: a small self-balancer evaluation subclass reusing RobotBaseEnv and Env01 control/observation methods, with an explicit constructor and reset protocol. Avoid copying a second independent implementation of the policy observation/action mapping.

## 3. What the script will calculate

The deliverable is specific to this two-wheel self-balancer topology: a box-like chassis, two cylindrical wheels, their joints, motors and contact world. Different dimensions/configurations of that robot are supported. Arbitrary robot types or arbitrary shape import are out of scope.

Separate four types of result:

1. **Input-domain limits:** finite positive dimensions/mass where required, valid units and structurally valid model data.
2. **Geometry/inertia bounds:** analytically derived conditions, labeled as sufficient constructions or necessary exclusions. A necessary bound does not make its entire interior feasible.
3. **Component/design limits:** supplied motor ratings, material constraints and user-chosen search caps. These cannot generally be inferred from width, height and mass.
4. **Observed behavior limits:** seed-, policy- and horizon-dependent transitions found by simulation, reported as brackets at a chosen resolution.

Every bound records its units, coordinate frame, inequality, endpoint inclusion, assumptions, source and status. Return `requires_input`, `unbounded` or `necessary_only` where appropriate; do not invent a finite maximum just to generate a test grid. Treat numerical instability and ordinary loss of balance as different outcomes.

### Input configuration

Use one JSON configuration plus the source scene/checkpoint paths. Include:

- Full chassis width (X), depth (Y), height (Z), body/geom offsets; wheel radius, full thickness, track width and attachment locations. Convert explicitly to MJCF half-sizes.
- Body masses and an inertia mode: geometry-derived uniform density, specified mass plus derived inertia, explicit measured inertia, or an explicitly defined component/payload distribution. Masses and centers must identify a body or the whole assembly.
- Motor speed/torque limits and servo gain; contact-pair friction, joint damping, gravity and medium settings.
- Parameters to test, units, percentage denominator, finite search caps where needed, absolute/relative mutation steps, refinement tolerance and test budget.
- Policy path, reference profile, seeds, simulated duration, timeout and predeclared diagnostic/balance criteria.

The parser should validate model topology, names, dimensions and coordinate conventions rather than inferring a universal robot from width/length/mass alone.

### User-facing request: CoM range for the current configuration

Use a named profile so the user does not retype dimensions already available in the model. The implemented `env01_neutral_compiled` profile resolves the selected world/includes, local PPO, neutral protocol and compiled baseline values. Its source paths resolve relative to the project, not the caller's working directory. Resolved paths and hashes are recorded in each run. See the README for complete commands, which also require a fresh output directory for generation/evaluation.

For example, save a request as `com_request.json`:

```json
{
  "profile": "env01_neutral_compiled",
  "property": "center_of_mass",
  "body": "robot_body",
  "direction": "fore_aft",
  "inertia_mode": "fixed_central",
  "search_extent": "geometry",
  "step_mm": 1.0,
  "refine_to_mm": 0.1,
  "max_refinement_cases": 20,
  "seconds_per_seed": 10,
  "seeds": [0, 1, 2, 3, 4]
}
```

This means: explore the chassis CoM relative to its current nominal position along body-local Y, keeping its other CoM coordinates, mass, central inertia, geometry and other physical settings fixed. `fore_aft`, `left_right` and `vertical` map explicitly to this model's Y, X and Z axes. The body selector matters: chassis CoM and whole-robot CoM are different quantities. An assembly request must specify how component mass properties change; there is no single whole-robot inertial field to edit.

`search_extent: geometry` derives the outer candidate interval from the support geometry through the nominal CoM, rather than requiring the user to supply the answer. Apply additional inertia/feasibility screening before rollouts; known-infeasible points are recorded as excluded unless an explicitly requested stress mode allows them. Fixed-central-inertia edits remain abstract perturbations, with unverified feasibility where only necessary tests pass. A uniformly filled box cannot develop an offset CoM while remaining uniformly filled with unchanged geometry. To claim realizable changes, request a component/payload mode and supply its masses, dimensions and allowed travel; then recompute both CoM and inertia.

For this centered chassis, the fore-aft geometric extent is +/-18.5 mm. That is an outer geometric bound, not a feasible or balanced interval. Report signed displacement in millimeters and percent of the nominal half-depth: 1.85 mm is 10%. Never use a percentage of the nominal Y coordinate, which is zero. For asymmetric geometry or noncentered nominal CoM, show the fixed normalization length and both directional distances explicitly.

With the proposed entry point available in the working directory, the interface would be:

```text
uv run python validate_self_balancer.py calculate --config com_request.json
uv run python validate_self_balancer.py run --config com_request.json --output results/my_com_run
```

`calculate` returns analytical bounds, assumptions, exclusions and unresolved feasibility without a policy rollout. `run` additionally generates cases, qualifies the baseline, simulates permitted cases and reports observed transitions. A 0.1 mm refinement tolerance is search resolution, not a guarantee of hardware accuracy. Keep geometric bounds, physically established/excluded/unknown cases, numerically valid observations and policy balance outcomes separate in both outputs. These example step sizes are configurable design choices, not measured limits.

### Individual and combined CoM tests

CoM is one vector property with three independently addressable coordinate components. Treat its components as separate sweep parameters, with a joint-testing option. Avoid an ambiguous `horizontal` field: for this robot `fore_aft` means body-local Y, `left_right` means body-local X, and `vertical` means body-local Z.

Recommended progression:

1. Calculate/sweep fore-aft CoM at nominal vertical and lateral CoM. Report a conditional one-dimensional result: "fore-aft offsets with Z and X held at their baseline values."
2. Calculate/sweep vertical CoM at nominal fore-aft and lateral CoM. Report that condition explicitly. Add a separate lateral sweep when left/right sensitivity is in scope.
3. Calculate/sweep a two-dimensional fore-aft/vertical grid, holding lateral CoM fixed. Each candidate specifies both offsets and passes geometry/full-inertia screening before any permitted rollout. Report a map of sampled combinations and transition brackets, with feasibility, numerical status and balance results separate.

The one-axis results are slices through a joint region. Do not multiply the passing intervals into a rectangle and claim every combination passes. A horizontal offset that balances at nominal height may fail at another height; combinations can also compensate, so a joint exploration should not be restricted automatically to the individually successful intervals. Similarly, independent necessary physical bounds do not establish joint feasibility. When a mathematical construction proves a region, record its assumptions and proof separately from finite simulation evidence.

Extend the proposed input schema with mutually exclusive single `direction` or multiple `directions`. A joint request would replace `"direction": "fore_aft"` with `"directions": ["fore_aft", "vertical"]` and declare `"sampling": "grid"`; support explicit per-direction steps and refinement tolerances. A grid changes both coordinates across combinations, not just equal offsets along one diagonal. Share the unchanged baseline evidence and deduplicate identical samples within the same profile/protocol. Limit joint sampling/refinement by a declared budget and retain untested cells as unknown.

Keep percentage denominators axis-specific and fixed to the declared nominal geometry. A joint result can be a horizontal-range-versus-height table or heatmap; it is not one universal percentage. All observed balance limits remain conditional on the body/configuration, mass-distribution assumption, frozen policy, world, seeds and duration.

### Initial property coverage and dependencies

| Property | Calculation/input required | Mutation and interpretation |
| --- | --- | --- |
| Chassis dimensions and wheel geometry | Positive sizes plus collision, clearance and attachment constraints; preserve topology. | Recompute shape inertia and dependent placements under an explicit mass-vs-density rule. One independent input can legitimately change several derived fields. |
| Chassis/assembly CoM | Specify which body or assembly and coordinate frame; use support geometry and full central inertia. Component distributions can provide constructive feasible cases. | Keep abstract fixed-inertia offsets separate from a realizable movable-payload model that changes inertia consistently. No inherited 71.4% capsule rule. |
| Mass | Positive moving-body mass plus supplied density/payload/component bounds; no geometry-only universal maximum. | For fixed shape/uniform density, scale central inertia proportionally. Do not change mass alone while claiming the same uniform material distribution. |
| Motor authority | Existing +/-0.65 N m limit supplies the reference; additional engineering ratings needed for other designs. | For a torque-capacity sweep, vary `forcerange` symmetrically while keeping velocity limits/gain fixed. Zero authority is a labeled failure case; do not copy the pendulum's gear sweep. |
| Wheel-floor friction | Nonnegative coefficients; finite physical range needs material data or a declared exploratory cap. | Env01's explicit contact pairs override the intended wheel-floor values. Change the pair coefficients and verify compiled contact settings, not just geom friction. |
| Hinge damping | Passive damping has a nonnegative input domain; upper design bound requires input. | Mutate named wheel joints; state whether both are coupled symmetrically or one-sided. |
| Medium density / wind | Nonnegative density and specified operating conditions; wind requires a separate nonzero-density/viscosity reference to have the intended effect. | Later module; leave disabled for the initial contact/motor/CoM study. |

A few examples make the calculations concrete; these are conditional bounds, not simulation results:

- Chassis full dimensions are currently 10.0 cm across the axle, 3.7 cm fore-aft and 17.1 cm vertically. Wheels have radius 3.4 cm, thickness 2.6 cm and center spacing 14.8 cm.
- For the present parallel assembly with fixed wheel spacing, keeping the chassis between the inner wheel faces gives `chassis_width + 2*clearance <= track_width - wheel_thickness`. Nominally that leaves 2.2 cm total lateral clearance. A full collision/placement check remains required.
- At upright ground contact, with unchanged local offsets, chassis-floor clearance is `wheel_radius - axle_height + chassis_bottom_height`. Here this is `radius - 2.0 cm`; require the configured positive clearance if chassis contact is excluded. Recompute placement when geometry changes.
- For a chassis CoM in box-body coordinates, obtain `C = trace(I)/(2m)*Id - I/m`. Necessary per-axis support bounds are `C_ii <= (upper_i-c_i)*(c_i-lower_i)`. For the current uniform box's fixed inertia, the coordinate exclusion bound is about 81.65% of each half-size, not the old capsule's 71.4%. This does not alone certify the full covariance or a realizable distribution. Whole-robot CoM requires mass-weighted assembly transforms and proper inertia aggregation; it cannot be bounded by the chassis box alone.

## 4. Test generation and simulation workflow

```text
configuration + world/robot + frozen PPO
    -> inspect and compile baseline
    -> calculate typed bounds and unresolved inputs
    -> generate boundary cases and incremental mutations
    -> validate model/feasibility before simulation
    -> run isolated neutral-policy tests
    -> refine observed transitions and report evidence
```

### Boundary value testing

For each supported finite bound L/U, generate the nominal value and cases immediately below, at and above each endpoint, using a property-specific epsilon. Respect open endpoints, coupled constraints and units. Invalid dimensions, missing fields and structurally invalid mass/inertia are expected input rejections; log those without attempting a meaningless rollout. Keep optional compileable but physically infeasible stress cases explicitly labeled and excluded from physically justified ranges.

### Data mutation testing

Generate at least 11 equally spaced values for each chosen finite exploration interval, with explicit steps and hashes; add boundary neighbors separately and deduplicate exact duplicates. In individual mode, vary one independent property or declared symmetric pair at a time, preserving/recomputing the specified derived quantities. Joint CoM mode deliberately varies the selected coordinate components as described above. Verify the requested change in the compiled model so compiler defaults cannot silently cancel it.

Where adjacent samples differ in numerical or balance outcomes, refine each bracket using a smaller step until the requested tolerance or resource budget is reached. Do not assume monotonic behavior or use one binary search to assert a universal threshold. Keep gaps and unresolved intervals visible. If no transition is found, report `not found within tested extent`; a search cap is not a discovered physical maximum. Exclusion proofs and empirical thresholds remain separate outputs.

### Baseline qualification before property sweeps

1. Freeze source XML/includes, environment code, checkpoint and versions by hash. Reproduce the original Env01-v3 behavior as a reference check.
2. Prove explicit-inertia conversion preserves the same compiled properties and trajectories under the same reference protocol.
3. Qualify the neutral adapter separately: zero targets, zero measurement offset, declared seeded pose/reset procedure, same observations/actions and timing. Require five full-duration baseline balance successes with no numerical flags before claiming a policy-based range. If it fails, still allow analytical calculation but stop policy sweeps and diagnose compatibility.
4. Repeat qualification for each new dimensions/mass configuration. A policy trained on one robot is not automatically competent on a resized one. Do not silently retrain or swap controllers to make a reference pass.

### Per-test protocol and recorded outcomes

Default to ten cumulative simulated seconds and seeds 0-4. With current timing that is **2,000 actions and 500,000 physics substeps per run**. Benchmark worker startup and reference wall time before choosing timeouts; do not assume the pendulum timeout/runtime is adequate.

Use headless isolated workers, deterministic PPO inference on CPU and a documented reset seed schedule. Reset immediately after ordinary termination/truncation; retain cumulative time and first-fall time. Full-duration balance means one uninterrupted episode. The reference demo waits before resetting in one path, so do not copy that loop into validation.

Record numerical validity, geometric/inertial feasibility, and balance/task performance independently. Log actual simulated time, resets, body orientations and velocities, actuator controls/forces, effort saturation, contact counts/forces and penetration, warning callbacks/counters, and engine time advances. Support both environment pitch termination and true body tilt diagnostics; distinguish yaw/spin/tipping that a one-angle criterion may miss. Thresholds must be explicit before the batch and tied to model scale or a labeled review heuristic.

Save state/action/target/contact traces around falls or flagged cases so later explanations can establish event order. Sample some diagnostics inside the 250-substep action interval or clearly report endpoint-only coverage; recording internal warnings alone does not observe every intermediate state. Added monitoring must not change the integration sequence or policy observation history.

Flush one JSONL result after every test and refresh CSV/report incrementally. Distinguish infrastructure errors, invalid models, numerical anomalies, review flags, known-infeasible distributions, unverified distributions, and ordinary balance failures. Keep expected rejected boundary inputs separate from unexpected failures of the test system.

## 5. Proposed implementation layout and reuse

One entry point: `validate_self_balancer.py`, with implemented subcommands `inspect`, `calculate`, `generate`, `run` and `report`. Supporting code is kept within this self-balancer validation folder:

| Module | Responsibility |
| --- | --- |
| `model_adapter.py` | Scene/include resolution, named bodies/joints/axes, compile audit, neutral environment and policy contract. |
| `bounds.py` | Unit-aware input validation, geometry/inertia formulas and bound provenance/status. |
| `variants.py` | Boundary neighbors, mutation grids, coupled derived updates and compiled difference checks. |
| `runner.py` | Frozen PPO inference, worker isolation, duration accounting, reference gates and diagnostics. |
| `reporting.py` | Separate numerical/physical/balance summaries, transition brackets and rounded readable reports. |
| `configs/`, `tests/` | Declared self-balancer profiles and meaningful regression fixtures. |

Reuse existing hashing, immutable snapshots, JSONL persistence, subprocess/timeout patterns, budget accounting and report conventions. Generalize the existing SB3 loader's metadata and space checks where useful. The current pendulum worker hardcodes two joints, one action, a stock Gymnasium ID, pendulum-specific metrics and motor manifest fields; it cannot run this robot unchanged. Preserve motor/CoM commands and historical evidence.

Each run saves its config, typed bounds, variant manifest, flattened XML or complete self-contained include bundle, checkpoint/environment fingerprints, per-test JSONL/worker logs, summary CSV, optional traces and report. Use a new locally ignored results directory. Baseline artifacts must carry a profile ID distinguishing legacy policy reproduction, neutral evaluation and later CAD-based physics.

## 6. Delivery order and acceptance checks

1. **Model inspection and neutral adapter.** Add the configurable scene path; audit conventions and reset behavior; prove policy spaces/action semantics, timing, reproducible reset and baseline qualification. No parameter sweep until this gate passes.
2. **Boundary calculator.** Implement the schema and supported self-balancer bounds, including expected invalid-input cases and honest unknown/unbounded results. Verify formulas against hand-derived simple box/cylinder examples and differently sized configurations.
3. **First end-to-end property.** Use motor torque-limit scaling to prove generation, policy trials and logging on the real wheel/contact robot. Confirm the compiled limit and actual torque response at a controlled nonzero tracking error; a changed limit does not necessarily change unsaturated motion.
4. **Geometry, mass and CoM modules.** Verify coupled placements, consistent inertia updates, frame transforms, collision clearance and separate realizability judgments. Start CoM with individual fore-aft/vertical sweeps, then support a joint grid and conditional range map. Do not reuse the pendulum range or interpret passing necessary inequalities as proof.
5. **Contact properties and refinement.** Add wheel-floor friction/damping, confirm effective contact/passive forces and finish refinement/reporting. Medium effects and disturbance worlds are later additions.

Regression checks must cover unchanged-baseline conversion, both RNG sources/neutral reset repeatability, quaternion ordering, zero targets throughout the run, derivative history at reset, policy I/O and wheel sign conventions, compile-applied mutations, expected invalid-input rejection, warning/nonfinite/time-rollback handling, full budget after ordinary falls, timeouts, immediate logging, and nonmonotonic range gaps. Baseline checks and exploratory runs must use the supplied frozen PPO; any untrained fixtures are labeled tests only.

Acceptance means the script can calculate and explain supported bounds for different configurations of this self-balancer, generate and run the corresponding tests, and report exclusions and empirical brackets without claiming universal or exact limits. No training, physical hardware actuation, unrelated model support, commits or publication is part of this plan.

## 7. Open engineering inputs and evidence

The first implementation can proceed with the compiled reference profile. Before making hardware claims, confirm whether the team's intended masses/CoM are the inertial-element values, geometry-derived values or new CAD values, and obtain component ratings/material or payload constraints for bounds that dimensions do not determine.

Points to discuss with Adam: was the geometry-derived inertia override intentional; which masses/CoM and checkpoint/model revision should be the reference; are floor friction and actuator limits measured values or training assumptions; and should CoM testing represent an isolated abstract edit or a specific movable component? Flat-ground results describe that contact world and frozen controller, not slopes, rough terrain or universal hardware limits. The velocity-servo model supplies torque/speed caps but does not explicitly model battery voltage, thermal limits or a measured motor torque-speed curve. Those effects require additional model data if they become part of the intended claim.

Inspection environment: Python 3.13.7, Gymnasium 1.3.0, MuJoCo 3.14.0, SB3 2.9.0 and PyTorch 2.14.0+cpu. The local environment has now been synced to Python 3.13. The PPO zip contains a six-input/two-output policy, records SB3 2.9.0 and 180,000 training timesteps, and has SHA256 `9bb72b08f45be7e043d000327cf79a67bb45914826685aa7380a5afd2c620f82`.

Compiled Env01 dimensions: nq=9, nv=8, nu=2. Env03 adds the block, giving nq=16, nv=14, nu=2. The starter also has nq=9/nv=8/nu=2, but identical counts do not imply compatible control semantics. Both inspected environment reset observations and PPO predictions were finite and within their declared spaces for seed 0. No balance duration has been measured in this planning step.

References: [MuJoCo explicit inertia](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-inertial), [contact pairs](https://mujoco.readthedocs.io/en/latest/XMLreference.html#contact-pair), and [SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm). Local source files and compiled-model inspection determine the model-specific findings above.
