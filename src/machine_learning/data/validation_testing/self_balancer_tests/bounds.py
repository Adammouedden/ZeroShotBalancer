"""Conditional bounds; necessary moment checks never certify a distribution."""
import math
import numpy as np
from model_adapter import AXES, describe, inertia_tensor

PROPERTIES = {"center_of_mass", "motor_strength", "mass_scale", "friction", "damping", "chassis_width", "chassis_depth", "chassis_height", "wheel_radius"}
FIELDS = {"profile", "property", "body", "direction", "directions", "sampling", "inertia_mode", "search_extent", "step_mm", "refine_to_mm", "seconds_per_seed", "seeds", "points", "model", "search_range", "boundary_epsilon", "max_cases", "timeout_seconds", "max_refinement_cases", "refine_to", "checkpoint", "initial_pitch_rad", "clearance_m", "diagnostics"}


def validate_config(raw):
    """Merge a user JSON dictionary with defaults and validate the request.

    Args:
        raw: User settings, such as property, model dimensions, axes and budget.
            Distances use meters unless a setting explicitly ends in ``_mm``.

    Returns:
        A new configuration dictionary with defaults and a ``directions`` list,
        even for a one-axis request. The input dictionary is not modified.

    Raises:
        ValueError: A field is unknown, unsupported, ambiguous or out of range.
            Model dimension checks occur later in configured_baseline().
    """
    unknown = set(raw) - FIELDS
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    # c means configuration. These are defaults, not fixed physical limits:
    # **raw comes last so the user's JSON overrides configurable settings.
    c = {"profile": "env01_neutral_compiled", "property": "center_of_mass", "body": "robot_body",
         "inertia_mode": "fixed_central", "search_extent": "geometry", "points": 11,
         "seconds_per_seed": 10, "seeds": list(range(5)), "model": {}, "max_cases": 400,
         "timeout_seconds": 180, "max_refinement_cases": 0, "initial_pitch_rad": .02,
         "clearance_m": .002, **raw}
    # These checks describe implemented scope; changing a string in JSON cannot
    # add a new robot topology or a moving-payload inertia model.
    if c["profile"] != "env01_neutral_compiled" or c["body"] != "robot_body":
        raise ValueError("Only the named neutral profile and chassis body are supported")
    if c["property"] not in PROPERTIES:
        raise ValueError(f"Supported properties: {sorted(PROPERTIES)}")
    if c["inertia_mode"] != "fixed_central":
        raise ValueError("Only fixed_central CoM perturbations are implemented; payload distributions require a separate model")
    if "direction" in c and "directions" in c:
        raise ValueError("Specify direction OR directions")
    directions = c.get("directions", [c.get("direction", "fore_aft")])
    if not isinstance(directions, list) or not 1 <= len(directions) <= 2 or len(set(directions)) != len(directions) or any(d not in AXES for d in directions):
        raise ValueError("Specify one or two distinct directions: fore_aft, left_right, vertical")
    if len(directions) > 1 and c.get("sampling") != "grid":
        raise ValueError("Multiple directions require sampling=grid")
    if c.get("sampling", "grid") != "grid":
        raise ValueError("Only grid sampling is supported")
    c["directions"] = directions
    c.pop("direction", None)
    for key in ("points", "max_cases", "max_refinement_cases"):
        if type(c[key]) is not int or c[key] < (0 if key == "max_refinement_cases" else 1):
            raise ValueError(f"{key} must be a nonnegative/positive integer")
    if c["points"] < 11:
        raise ValueError("Use at least 11 points per axis")
    if not isinstance(c["seeds"], list) or not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]) or any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    for key in ("seconds_per_seed", "timeout_seconds", "step_mm", "refine_to_mm", "boundary_epsilon", "refine_to"):
        if key in c:
            values = c[key].values() if isinstance(c[key], dict) and key in {"step_mm", "refine_to_mm"} else [c[key]]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{key} must be finite and positive")
    for key in ("initial_pitch_rad", "clearance_m"):
        if not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"Invalid {key}")
    if c["initial_pitch_rad"] >= math.radians(50):
        raise ValueError("Initial pitch must be below the fall criterion")
    if "search_range" in c:
        r = c["search_range"]
        if not isinstance(r, list) or len(r) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in r) or r[0] >= r[1]:
            raise ValueError("search_range must be two increasing finite numbers")
    if c["property"] == "center_of_mass" and ("search_range" in c or c["search_extent"] != "geometry"):
        raise ValueError("CoM currently derives its search extent from geometry; no numeric search_range")
    return c


def calculate(model, config):
    """Calculate conditional property bounds without running the controller.

    Args:
        model: Compiled MuJoCo model of the configured reference robot.
        config: Validated request from validate_config().

    Returns:
        Dictionary containing model information, per-axis/property bounds and
        limitations. CoM intervals are meter offsets from the nominal chassis
        CoM, in chassis coordinates. Other properties carry their own units.
        Missing engineering caps are marked requires_input, not invented.

    A necessary moment interval excludes impossible CoM/inertia combinations;
    its interior is not a physical-realizability or balance guarantee.
    """
    info = describe(model)
    prop = config["property"]
    result = {"property": prop, "configuration": info, "bounds": [],
              "limitations": ["Bounds are conditional on this geometry and inertia; policy limits need simulation.",
                              "Necessary moment inequalities exclude cases but do not certify their interior."]}
    if prop == "center_of_mass":
        body = model.body("robot_body").id
        inertia = inertia_tensor(model, body)
        # Inertia I measures rotational mass spread. Convert it to positional
        # covariance C = trace(I)/(2m) * identity - I/m. Each diagonal C_ii is
        # mass-weighted variance along that body axis, in square meters.
        covariance = np.trace(inertia) / (2 * model.body_mass[body]) * np.eye(3) - inertia / model.body_mass[body]
        half = np.array(info["chassis_half_size_m"])
        center = np.array(info["chassis_geom_center_m"])
        nominal = np.array(info["chassis_com_m"])
        for direction in config["directions"]:
            i = AXES[direction]
            # For mass inside [center-half, center+half], a necessary condition
            # is C_ii + (com_i-center_i)^2 <= half_i^2. Here "radius" means
            # allowed distance from the box center, not the wheel radius.
            radius_squared = half[i]**2 - covariance[i, i]
            # Avoid sqrt of a negative number; report no necessary interval
            # below when the variance cannot fit inside this geometry.
            radius = math.sqrt(max(0, radius_squared))
            # Subtract nominal to express absolute positions as CoM changes.
            # Percentage = 100 * offset / half-size, not offset / nominal CoM.
            result["bounds"].append({"direction": direction, "axis": i, "units": "m offset from nominal CoM",
                "geometric": [float(center[i] - half[i] - nominal[i]), float(center[i] + half[i] - nominal[i])],
                "necessary_moment": None if radius_squared < 0 else [float(center[i] - radius - nominal[i]), float(center[i] + radius - nominal[i])],
                "status": "necessary_only", "normalization_length_m": float(half[i]),
                "inequality": "C_ii <= (upper_i - com_i) * (com_i - lower_i)", "frame": "robot_body"})
    else:
        nominal = {"motor_strength": 1., "mass_scale": 1., "friction": float(model.pair_friction[0, 0]),
                   "damping": float(model.dof_damping[model.joint('torso_l_wheel').dofadr[0]]),
                   "chassis_width": 2*info["chassis_half_size_m"][0], "chassis_depth": 2*info["chassis_half_size_m"][1],
                   "chassis_height": 2*info["chassis_half_size_m"][2], "wheel_radius": info["wheel_radius_m"]}[prop]
        # Chassis width is limited by wheel clearance. Most other finite maxima
        # need user-supplied search_range; motor 0..1 is an exploratory default.
        cap = info["track_width_m"] - info["wheel_thickness_m"] if prop == "chassis_width" else None
        interval = config.get("search_range", [0., 1.] if prop == "motor_strength" else ([0., cap] if cap else None))
        result["bounds"].append({"direction": prop, "nominal": nominal, "domain_lower": 0.,
            "lower_inclusive": prop in {"motor_strength", "friction", "damping"}, "geometry_upper_exclusive": cap,
            "search": interval, "status": "requires_input" if interval is None else "conditional_search",
            "units": "m" if prop.startswith("chassis_") or prop == "wheel_radius" else ("N m s/rad" if prop == "damping" else "dimensionless"),
            "source": "geometry clearance" if cap else "declared search cap; not a universal maximum"})
    return result


def feasibility(model, offsets):
    """Screen a proposed chassis CoM displacement against geometry and inertia.

    Args:
        model: Compiled reference model supplying fixed mass/central inertia.
        offsets: Three meter offsets [X, Y, Z] in the chassis body frame.

    Returns:
        Status, exclusion reasons and required/available variances in m^2.
        Outside-support or moment violations are known_infeasible. Admitted
        nonzero shifts are unverified; near-zero uses nominal_geometry_derived.
        This function does not mutate the model or simulate balance.
    """
    body = model.body("robot_body").id
    info = describe(model)
    center = np.array(info["chassis_geom_center_m"])
    half = np.array(info["chassis_half_size_m"])
    com = np.array(info["chassis_com_m"]) + offsets
    inertia = inertia_tensor(model, body)
    covariance = np.trace(inertia) / (2 * model.body_mass[body]) * np.eye(3) - inertia / model.body_mass[body]
    # Maximum coordinate variance allowed by the proposed mean inside the box:
    # (upper - mean) * (mean - lower). It shrinks toward zero at an edge.
    available = (center + half - com) * (com - center + half)
    reasons = []
    if np.any(com < center - half - 1e-12) or np.any(com > center + half + 1e-12):
        reasons.append("outside_chassis_support")
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
        reasons.append("negative_mass_covariance")
    if np.any(np.diag(covariance) > available + 1e-12):
        reasons.append("bounded_variance_violation")
    return {"status": "known_infeasible" if reasons else ("nominal_geometry_derived" if np.max(np.abs(offsets)) < 1e-12 else "unverified"),
            "reasons": reasons, "variance_m2": np.diag(covariance).tolist(), "available_variance_m2": available.tolist()}
