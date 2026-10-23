"""Explicit-inertia copies and a neutral adapter for Adam's two-wheel robot."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REFERENCE = PROJECT / "training/reference_baseline"
ROBOT_ENVIRONMENTS = PROJECT / "training/robot_environments"
WORLD = ROBOT_ENVIRONMENTS / "base_world_env.xml"
POLICY = REFERENCE / "models/Env01-v3_PPO/best_model.zip"
AXES = {"left_right": 0, "fore_aft": 1, "vertical": 2}

# Prefer this source tree over the editable installation. A saved worker must
# import its saved robot code, not whatever is currently installed in the repo.
sys.path.insert(0, str(PROJECT / "training"))
from robot_environments.base_world_env import BaseWorldEnv
from robot_environments.RobotBaseEnv import RobotBaseEnv

# Avoid the reference package's global Gym registration side effects.
package = types.ModuleType("_validation_reference_envs")
package.__path__ = [str(REFERENCE / "envs")]
sys.modules.setdefault(package.__name__, package)
reference_module = importlib.import_module(package.__name__ + ".reference_baseline_training_env")
ReferenceBaselineTrainingEnv = reference_module.ReferenceBaselineTrainingEnv
DEFAULT_REWARD_WEIGHTS = reference_module.DEFAULT_REWARD_WEIGHTS


def digest(path):
    """Read a file path and return its SHA-256 hex fingerprint; do not modify it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write JSON-serializable value as UTF-8 at path; reject NaN/infinity.

    The parent must exist. Overwrites the file and returns None.
    """
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def numbers(values):
    """Format a scalar or 1-D numeric sequence as space-separated MJCF text.

    Use 17 significant digits to preserve double precision during XML copies.
    """
    return " ".join(format(float(x), ".17g") for x in np.atleast_1d(values))


def save_xml(root, path):
    """Indent an ElementTree root in place and write UTF-8 XML to path.

    The parent must exist; the destination is overwritten. Returns None.
    """
    ET.indent(root)
    Path(path).write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def flatten(path, seen=()):
    """Read MJCF at path and return one XML root with includes expanded.

    Include paths are relative to the containing XML file. seen tracks the
    current include chain to reject cycles with ValueError. Source files stay
    untouched; the returned tree is independent and can be modified safely.
    """
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Cyclic XML include")
    root = ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "include":
                included = flatten(path.parent / child.attrib["file"], (*seen, path))
                index = list(parent).index(child)
                parent.remove(child)
                for offset, element in enumerate(list(included)):
                    parent.insert(index + offset, element)
    return root


def compile_root(root):
    """Compile an ElementTree root into an MjModel without writing a file.

    MuJoCo compilation errors propagate; no time steps or rollouts occur here.
    """
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def inertia_tensor(model, body):
    """Return a body's 3x3 central inertia in body coordinates, in kg m^2.

    model is compiled MuJoCo data and body is its integer body ID. Rotate the
    stored principal moments from the inertial frame into the body frame.
    """
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def describe(model):
    """Extract actual compiled robot properties into a JSON-ready dictionary.

    Takes an MjModel and checks the supported box-chassis/two-wheel topology.
    Returns dimensions in meters, masses in kg, inertia in kg m^2, and timing,
    gravity, contact and actuator settings. Chassis size is reported as half
    dimensions, while wheel thickness is full thickness. Unsupported chassis
    shape/orientation or state/action counts raise ValueError.
    """
    body = model.body("robot_body").id
    geom = int(model.body_geomadr[body])
    if (model.nq, model.nv, model.nu) != (9, 8, 2):
        raise ValueError("Only the selected free chassis/two-wheel topology is supported")
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("The chassis must be a box")
    if not np.allclose(model.geom_quat[geom], [1, 0, 0, 0]):
        raise ValueError("Rotated chassis geometry needs a separate support calculation")
    return {
        "chassis_half_size_m": model.geom_size[geom].tolist(),
        "chassis_geom_center_m": model.geom_pos[geom].tolist(),
        "chassis_com_m": model.body_ipos[body].tolist(),
        "chassis_mass_kg": float(model.body_mass[body]),
        "central_inertia_kg_m2": inertia_tensor(model, body).tolist(),
        "total_mass_kg": float(model.body_mass.sum()),
        "wheel_radius_m": float(model.geom_size[model.geom("l_wheel_geom").id, 0]),
        "wheel_thickness_m": float(2 * model.geom_size[model.geom("l_wheel_geom").id, 1]),
        "track_width_m": float(model.body_pos[model.body("r_wheel").id, 0] - model.body_pos[model.body("l_wheel").id, 0]),
        "physics_dt": float(model.opt.timestep), "control_dt": float(model.opt.timestep * 250),
        "gravity": model.opt.gravity.tolist(), "pair_friction": model.pair_friction.tolist(),
        "actuator_force_limits": model.actuator_forcerange.tolist(),
    }


def explicit_baseline(source=WORLD):
    """Create explicit inertials without changing the compiled reference physics.

    source is the world XML path, whose includes are expanded. Returns
    (converted_XML_root, original_MjModel); does not edit source files.
    Copy every body's compiled mass, CoM, principal moments and orientation
    before selecting inertiafromgeom=auto. Raise ValueError if the supported
    numeric model-array comparison detects a change after conversion.
    """
    root = flatten(source)
    original = compile_root(root)
    describe(original)
    for element in root.findall(".//body"):
        body = original.body(element.attrib["name"]).id
        for inertial in element.findall("inertial"):
            element.remove(inertial)
        ET.SubElement(element, "inertial", {
            "mass": numbers(original.body_mass[body]),
            "pos": numbers(original.body_ipos[body]),
            "diaginertia": numbers(original.body_inertia[body]),
            "quat": numbers(original.body_iquat[body]),
        })
    root.find("compiler").set("inertiafromgeom", "auto")
    converted = compile_root(root)
    differences = compare_models(original, converted)
    if differences:
        raise ValueError(f"Explicit-inertia conversion differs: {differences}")
    return root, original


def compare_models(a, b):
    """Compare exposed numeric arrays of two compiled MjModels with tolerances.

    Return {array_name: maximum_absolute_difference_or_shape_marker}; an empty
    dictionary means these arrays agree. This does not compare every scalar or
    nested option field, and is not a substitute for trajectory comparisons.
    """
    differences = {}
    for name in dir(a):
        if name.startswith("_"):
            continue
        left, right = getattr(a, name), getattr(b, name)
        if isinstance(left, np.ndarray) and np.issubdtype(left.dtype, np.number):
            if left.shape != right.shape or not np.allclose(left, right, atol=2e-13, rtol=2e-12):
                differences[name] = "shape" if left.shape != right.shape else float(np.max(np.abs(left - right)))
    return differences


def configured_baseline(root, settings):
    """Resize this topology; keep mass fixed unless explicitly supplied.

    A resized body is treated as uniform-density geometry and its central inertia
    is regenerated. Wheel attachment follows radius/track; chassis bottom remains
    at the original local height. No mutation of the training XMLs.

    Args:
        root: Explicit-inertia baseline ElementTree root, copied before editing.
        settings: Optional dimension/mass overrides. Chassis sizes and wheel
            thickness are full dimensions in meters; masses are kilograms.

    Returns:
        A new XML root, compiled and checked for basic chassis/wheel clearance.
        Raises ValueError for unsupported settings, nonpositive dimensions or
        failed clearance. This does not certify manufacturing feasibility.
    """
    allowed = {"chassis_size_m", "chassis_mass_kg", "wheel_radius_m", "wheel_thickness_m", "wheel_mass_kg", "track_width_m"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown model settings: {set(settings) - allowed}")
    root = copy.deepcopy(root)
    body = root.find(".//body[@name='robot_body']")
    geom = body.find("geom")
    size = np.fromstring(geom.get("size"), sep=" ")
    center = np.fromstring(geom.get("pos"), sep=" ")
    if "chassis_size_m" in settings:
        full = np.asarray(settings["chassis_size_m"], dtype=float)
        if full.shape != (3,) or not np.isfinite(full).all() or np.any(full <= 0):
            raise ValueError("chassis_size_m must contain three finite positive full dimensions")
        center[2] += full[2] / 2 - size[2]
        size = full / 2
        geom.set("size", numbers(size))
        geom.set("pos", numbers(center))
    for key, value in settings.items():
        if key != "chassis_size_m" and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be finite and positive")
    inertial = body.find("inertial")
    if "chassis_mass_kg" in settings:
        inertial.set("mass", numbers(settings["chassis_mass_kg"]))
    if {"chassis_size_m", "chassis_mass_kg"} & settings.keys():
        mass = float(inertial.get("mass"))
        inertial.set("pos", numbers(center))
        inertial.set("quat", "1 0 0 0")
        inertial.set("diaginertia", numbers(mass / 3 * np.array([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2, size[0]**2 + size[1]**2])))
    for name, sign in (("l_wheel", -1), ("r_wheel", 1)):
        wheel = body.find(f"body[@name='{name}']")
        wg, wi = wheel.find("geom"), wheel.find("inertial")
        radius, half = np.fromstring(wg.get("size"), sep=" ")
        radius = settings.get("wheel_radius_m", radius)
        half = settings.get("wheel_thickness_m", 2 * half) / 2
        pos = np.fromstring(wheel.get("pos"), sep=" ")
        pos[2] = radius
        if "track_width_m" in settings:
            pos[0] = sign * settings["track_width_m"] / 2
        wheel.set("pos", numbers(pos))
        wg.set("size", numbers([radius, half]))
        if "wheel_mass_kg" in settings:
            wi.set("mass", numbers(settings["wheel_mass_kg"]))
        if {"wheel_radius_m", "wheel_thickness_m", "wheel_mass_kg"} & settings.keys():
            mass = float(wi.get("mass"))
            wi.set("quat", wg.get("quat"))
            wi.set("diaginertia", numbers([mass * (3 * radius**2 + 4 * half**2) / 12] * 2 + [mass * radius**2 / 2]))
    model = compile_root(root)
    info = describe(model)
    if 2 * size[0] >= info["track_width_m"] - info["wheel_thickness_m"]:
        raise ValueError("Chassis does not clear the inner wheel faces")
    if center[2] - size[2] <= 0:
        raise ValueError("Chassis bottom must clear the wheel contact plane")
    return root


class NeutralEnv(BaseWorldEnv):
    """Versioned neutral reset; inherited policy observations and action semantics."""
    def __init__(self, scene, initial_pitch_rad=0.02, clearance_m=0.002):
        """Load an XML path for headless evaluation with the original policy spaces.

        initial_pitch_rad is the symmetric reset pitch bound; clearance_m is
        the initial gap above floor contact. Store both for seeded resets.
        One control action advances 250 physics steps (5 ms for this world).
        Construction loads a model; it does not train a policy.
        """
        utils.EzPickle.__init__(self, str(scene), initial_pitch_rad, clearance_m)
        self.initial_pitch_rad = initial_pitch_rad
        self.clearance_m = clearance_m
        observation_space = Box(np.array([-2 * math.pi] * 2 + [-1] * 4, dtype=np.float32),
                                np.array([2 * math.pi] * 2 + [1] * 4, dtype=np.float32))
        MujocoEnv.__init__(self, str(Path(scene).resolve()), 250, observation_space, render_mode=None)
        self.loop_count = 0
        self.last_time = self.last_pitch = None
        self.target_wheel_speed = self.target_yaw = 0.0

    def _set_action_space(self):
        """Set and return two normalized wheel-speed-increment actions in [-1, 1]."""
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        return self.action_space

    def get_pitch(self):
        """Return chassis X Euler angle in radians using the correct quaternion order."""
        q = self.data.body("robot_body").xquat
        return float(Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[0])

    def reset_model(self):
        """Place the robot above the floor and return the initial policy observation.

        Gymnasium reset(seed=...) supplies self.np_random. Sample pitch/yaw,
        clear velocities and derivative history, and set speed/yaw targets to
        zero. Geometry support determines placement even for a resized robot.
        Mutates simulator state; it does not change the model's physical values.
        """
        qpos = self.init_qpos.copy()
        qpos[:3] = 0
        pitch = self.np_random.uniform(-self.initial_pitch_rad, self.initial_pitch_rad)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        xyzw = Rotation.from_euler("xyz", [pitch, 0, yaw]).as_quat()
        qpos[3:7] = xyzw[[3, 0, 1, 2]]
        self.set_state(qpos, np.zeros(self.model.nv))
        # Exact support of the box/cylinders at the chosen orientation.
        minimum = math.inf
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] == 0:
                continue
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            size = self.model.geom_size[geom]
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rotation[2]) @ size
            elif self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z = rotation[2, 2]
                extent = size[0] * math.sqrt(max(0.0, 1 - z*z)) + size[1] * abs(z)
            else:
                raise ValueError("Unsupported geometry in placement calculation")
            minimum = min(minimum, self.data.geom_xpos[geom, 2] - extent)
        floor = self.model.geom_pos[self.model.geom("floor").id, 2]
        qpos[2] += floor + self.clearance_m - minimum
        self.set_state(qpos, np.zeros(self.model.nv))
        self.target_wheel_speed = self.target_yaw = 0.0
        self.last_time = self.last_pitch = None
        return self._get_obs()
