"""Boundary neighbours, one-axis mutations and joint CoM grids."""
import copy
import itertools
import math
from pathlib import Path
import numpy as np
from bounds import calculate, feasibility
from model_adapter import AXES, compile_root, configured_baseline, digest, numbers, save_xml, write_json


def axis_values(bound, config):
    """Return sorted initial CoM offsets in meters, including endpoints and zero.

    bound is one axis entry from calculate(); config supplies points or step_mm
    (a scalar or per-axis dictionary). step_mm limits maximum initial spacing.
    For a centered nominal CoM, 11 points span -100%..100% of the half-size in
    20-percentage-point steps. This creates candidates; feasibility is checked
    later. Raise ValueError if the requested count exceeds max_cases.
    """
    low, high = bound["geometric"]
    step = config.get("step_mm")
    if isinstance(step, dict):
        step = step.get(bound["direction"])
    count = config["points"] if step is None else max(11, math.ceil((high-low)/(step/1000)) + 1)
    if count > config["max_cases"]:
        raise ValueError("Requested resolution exceeds max_cases")
    values = np.linspace(low, high, count).tolist()
    # Subtracting the nominal height can make the endpoints differ by one ULP.
    # Snap only machine-scale zero roundoff, before deduplication, so the nominal
    # case truly preserves inertia/trajectories and is not a tiny perturbation.
    zero_tolerance = 8 * np.finfo(float).eps * max(abs(low), abs(high))
    values = [0. if abs(value) <= zero_tolerance else value for value in values]
    return sorted(set([*values, 0.]))


def mutate(root, config, values):
    """Apply one property case to a copy of an explicit-inertia XML tree.

    Args:
        root: Baseline ElementTree root; it is never changed in place.
        config: Validated property request.
        values: Axis-to-offset dictionary for CoM (meters), or a dictionary
            mapping the scalar property name to its value in the stated units.

    Returns:
        (changed_root, physical_status), or (None, exclusion_status) when the
        requested case is inadmissible. Compiles admitted cases and checks
        effective model fields; this is not a policy rollout.
    """
    root = copy.deepcopy(root)
    base = compile_root(root)
    prop = config["property"]
    body = root.find(".//body[@name='robot_body']")
    inertial = body.find("inertial")
    physical = {"status": "model_parameter_unverified", "reasons": []}
    if prop == "center_of_mass":
        offsets = np.zeros(3)
        for direction, value in values.items():
            offsets[AXES[direction]] = value
        physical = feasibility(base, offsets)
        if physical["status"] == "known_infeasible":
            return None, physical
        expected = base.body_ipos.copy()
        expected[base.body("robot_body").id] += offsets
        inertial.set("pos", numbers(expected[base.body("robot_body").id]))
        model = compile_root(root)
        for name in ("body_mass", "body_inertia", "body_iquat", "geom_pos", "geom_size", "dof_damping", "actuator_forcerange"):
            if not np.allclose(getattr(base, name), getattr(model, name), atol=2e-13, rtol=2e-12):
                raise AssertionError(f"CoM mutation changed {name}")
        np.testing.assert_allclose(model.body_ipos, expected, atol=1e-14, rtol=1e-13)
    else:
        value = values[prop]
        strictly_positive = prop not in {"motor_strength", "friction", "damping"}
        if value < 0 or (strictly_positive and value == 0):
            return None, {"status": "invalid_input", "reasons": ["outside_parameter_domain"]}
        if prop == "mass_scale":
            inertial.set("mass", numbers(float(inertial.get("mass")) * value))
            inertial.set("diaginertia", numbers(np.fromstring(inertial.get("diaginertia"), sep=" ") * value))
            physical["status"] = "geometry_derived"
        elif prop == "motor_strength":
            for actuator in root.findall(".//actuator/velocity"):
                if value == 0:
                    actuator.set("kv", "0")  # zero servo force, including velocity feedback
                else:
                    actuator.set("forcerange", numbers(np.fromstring(actuator.get("forcerange"), sep=" ") * value))
        elif prop == "friction":
            for pair in root.findall(".//contact/pair"):
                friction = np.fromstring(pair.get("friction"), sep=" ")
                friction[:2] = value
                pair.set("friction", numbers(friction))
        elif prop == "damping":
            for joint in root.findall(".//joint[@type='hinge']"):
                joint.set("damping", numbers(value))
        else:
            if prop == "wheel_radius":
                settings = {"wheel_radius_m": value}
            else:
                full = 2 * np.array(base.geom_size[base.body_geomadr[base.body("robot_body").id]])
                full[{"chassis_width": 0, "chassis_depth": 1, "chassis_height": 2}[prop]] = value
                settings = {"chassis_size_m": full.tolist()}
            try:
                root = configured_baseline(root, settings)
            except ValueError as error:
                return None, {"status": "invalid_input", "reasons": [str(error)]}
            physical["status"] = "geometry_derived"
        model = compile_root(root)
        if prop == "mass_scale":
            i = model.body("robot_body").id
            np.testing.assert_allclose(model.body_mass[i], base.body_mass[i] * value)
            np.testing.assert_allclose(model.body_inertia[i], base.body_inertia[i] * value)
        elif prop == "motor_strength":
            if value == 0:
                assert np.all(model.actuator_gainprm == 0) and np.all(model.actuator_biasprm == 0)
            else:
                np.testing.assert_allclose(model.actuator_forcerange, base.actuator_forcerange * value)
        elif prop == "friction":
            np.testing.assert_allclose(model.pair_friction[:, :2], value)
        elif prop == "damping":
            for name in ("torso_l_wheel", "torso_r_wheel"):
                np.testing.assert_allclose(model.dof_damping[model.joint(name).dofadr[0]], value)
    return root, physical


def make_case(root, config, values, kind, directory, index):
    """Build one manifest entry and write its XML only if screening admits it.

    Takes a baseline XML root, validated config, property values, a kind label
    (mutation/boundary/refinement), an existing output directory and integer ID.
    Returns values/status plus an XML filename and hash for admitted cases.
    Excluded entries retain reasons but have no XML or simulation result.
    """
    changed, physical = mutate(root, config, values)
    case = {"id": f"case_{index:04d}", "values": {k: float(v) for k, v in values.items()}, "kind": kind, "physical": physical}
    if changed is not None:
        path = Path(directory) / (case["id"] + ".xml")
        save_xml(changed, path)
        case.update(xml=path.name, sha256=digest(path))
    return case


def generate(root, config, directory):
    """Write initial cases and a manifest into a new directory; return manifest.

    root is the configured baseline XML tree; config supplies the property,
    axes, resolution and case budget. Joint CoM uses all axis combinations.
    Add points just inside/on/outside relevant boundaries, then deduplicate.
    This generates inputs only; runner.py performs rollouts and refinement.
    Refuse an existing directory or a grid that exceeds max_cases.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    base = compile_root(root)
    calculation = calculate(base, config)
    candidates = []
    if config["property"] == "center_of_mass":
        bounds = calculation["bounds"]
        grids = [axis_values(b, config) for b in bounds]
        if math.prod(map(len, grids)) > config["max_cases"]:
            raise ValueError("Joint grid exceeds max_cases; increase budget or step size")
        # Joint tests combine axes; passing individual slices does not establish
        # that every combination balances or satisfies the moment constraints.
        for values in itertools.product(*grids):
            candidates.append(({b["direction"]: v for b, v in zip(bounds, values)}, "mutation"))
        for b in bounds:
            eps = config.get("boundary_epsilon", max(1e-6, b["normalization_length_m"] * .001))
            # Test both sides of a boundary. Inadmissible neighbours still get
            # manifest entries so an exclusion is distinguishable from a fall.
            for limits in (b["geometric"], b["necessary_moment"]):
                if limits is None:
                    continue
                for edge in limits:
                    for delta in (-eps, 0, eps):
                        values = {item["direction"]: 0. for item in bounds}
                        values[b["direction"]] = edge + delta
                        candidates.append((values, "boundary"))
    else:
        bound = calculation["bounds"][0]
        if bound["search"] is None:
            raise ValueError("No finite upper bound can be inferred; supply search_range")
        low, high = bound["search"]
        prop = config["property"]
        candidates.extend(({prop: x}, "mutation") for x in [*np.linspace(low, high, config["points"]), bound["nominal"]])
        eps = config.get("boundary_epsilon", (high-low) * .001)
        for edge in (low, high, bound["domain_lower"], bound["geometry_upper_exclusive"]):
            if edge is not None:
                candidates.extend(({prop: edge+d}, "boundary") for d in (-eps, 0., eps))
    cases, seen = [], set()
    for values, kind in candidates:
        key = tuple((k, round(float(v), 13)) for k, v in sorted(values.items()))
        if key in seen:
            continue
        seen.add(key)
        if len(cases) >= config["max_cases"]:
            raise ValueError("Boundary cases plus mutations exceed max_cases")
        cases.append(make_case(root, config, values, kind, directory, len(cases)))
    manifest = {"bounds": calculation, "cases": cases, "config": config}
    write_json(directory / "manifest.json", manifest)
    return manifest
