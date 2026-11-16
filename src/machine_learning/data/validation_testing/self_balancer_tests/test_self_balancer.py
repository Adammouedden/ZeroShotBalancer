"""Regression tests; zero-input fixtures below are never used for study sweeps."""
import copy
import json
import math
from pathlib import Path
import shutil
import uuid
import unittest
import numpy as np
import mujoco

from bounds import calculate, feasibility, validate_config
from model_adapter import (HERE, NeutralEnv, compare_models, compile_root, configured_baseline,
                           describe, digest, explicit_baseline, save_xml)
from variants import axis_values, generate, mutate
from reporting import refinement_candidates
from runner import LegacyCopy, execute_job, metrics_valid, trial, read_rows


class ZeroFixture:
    """Test-only controller to exercise falls/resets; not a trained policy."""
    def predict(self, obs):
        """Ignore the observation and return two zero actions for failure-path tests."""
        return np.zeros(2, dtype=np.float32)


class ValidationTests(unittest.TestCase):
    """Checks of bounds, mutations, execution and reporting using local fixtures.

    unittest supplies self/cls; test methods return None and signal failures
    through assertions. Temporary artifacts live under generated/ and are
    removed after each test. These fixtures are not the PPO study evaluations.
    """
    @classmethod
    def setUpClass(cls):
        """Prepare the shared original/explicit model pair once for this test class."""
        (HERE / "generated").mkdir(exist_ok=True)
        cls.root, cls.original = explicit_baseline()

    def setUp(self):
        """Create an isolated per-test folder and baseline XML from the shared tree."""
        # Normal inherited Windows ACLs; tempfile's private 0700 ACL can exclude
        # the sandbox token even inside the writable workspace.
        self.directory = (HERE / "generated" / ("test_" + uuid.uuid4().hex)).resolve()
        assert self.directory.is_relative_to((HERE / "generated").resolve())
        self.directory.mkdir()
        self.path = self.directory / "baseline.xml"
        save_xml(copy.deepcopy(self.root), self.path)

    def tearDown(self):
        """Remove only this test's folder after checking it remains under generated/."""
        assert self.directory.is_relative_to((HERE / "generated").resolve())
        shutil.rmtree(self.directory)

    def test_explicit_conversion_all_compiled_numeric_arrays(self):
        """Assert explicit inertials preserve the inspected numeric model arrays."""
        self.assertEqual(compare_models(self.original, compile_root(self.root)), {})

    def test_renamed_reference_initializes_modular_rewards(self):
        """Exercise one reference step and check Adam's modular reward bookkeeping."""
        env = LegacyCopy(self.path)
        try:
            env.reset(seed=0)
            observation, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertTrue(np.isfinite(observation).all())
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
            self.assertEqual(set(info["reward_terms"]), {"alive", "pitch", "speed_error", "lean", "yaw"})
        finally:
            env.close()

    def test_com_changes_only_requested_inertial_position(self):
        """Apply a 1 mm Y shift and check the effective CoM and unverified label."""
        config = validate_config({"direction": "fore_aft"})
        root, physical = mutate(self.root, config, {"fore_aft": .001})
        model = compile_root(root)
        np.testing.assert_allclose(model.body_ipos[1] - self.original.body_ipos[1], [0, .001, 0], atol=1e-15)
        self.assertEqual(physical["status"], "unverified")

    def test_variance_bound_matches_hand_box_formula_and_scales(self):
        """Compare resized-box bounds with half-size * sqrt(2/3) on both axes."""
        config = validate_config({"directions": ["fore_aft", "vertical"], "sampling": "grid"})
        for size in ([.1, .037, .171], [.11, .06, .3]):
            root = configured_baseline(self.root, {"chassis_size_m": size, "chassis_mass_kg": .8})
            result = calculate(compile_root(root), config)
            for b in result["bounds"]:
                self.assertAlmostEqual(b["necessary_moment"][1], size[b["axis"]] / 2 * math.sqrt(2/3), places=12)

    def test_vertical_grid_has_exact_zero_and_unchanged_compiled_reference(self):
        """Require an exact nominal sample despite floating-point endpoint roundoff."""
        config = validate_config({"direction": "vertical"})
        bound = calculate(self.original, config)["bounds"][0]
        values = axis_values(bound, config)
        self.assertEqual(len(values), 11)
        self.assertEqual(values[5], 0.)
        root, _ = mutate(self.root, config, {"vertical": values[5]})
        np.testing.assert_array_equal(compile_root(root).body_ipos, self.original.body_ipos)

    def test_support_and_extreme_moment_rejections(self):
        """Reject impossible offsets while recognizing the original nominal case."""
        for offset in ([0, .03, 0], [0, .0185, 0], [0, 0, .0855]):
            self.assertEqual(feasibility(self.original, offset)["status"], "known_infeasible")
        self.assertEqual(feasibility(self.original, [0, 0, 0])["status"], "nominal_geometry_derived")

    def test_unknown_caps_are_not_fabricated(self):
        """Require user input when geometry cannot supply a finite mass-scale cap."""
        result = calculate(self.original, validate_config({"property": "mass_scale"}))
        self.assertEqual(result["bounds"][0]["status"], "requires_input")

    def test_motor_cap_changes_measured_saturated_force(self):
        """Check applied force at zero, half and full torque capacity, not just XML."""
        for scale in (0, .5, 1):
            root, _ = mutate(self.root, validate_config({"property": "motor_strength"}), {"motor_strength": scale})
            model = compile_root(root)
            data = mujoco.MjData(model)
            data.ctrl[:] = 70
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.actuator_force, .65 * scale, atol=1e-12)

    def test_friction_and_damping_reach_effective_fields(self):
        """Check contact-pair friction and actual passive damping-force effects."""
        root, _ = mutate(self.root, validate_config({"property": "friction"}), {"friction": .2})
        model = compile_root(root)
        np.testing.assert_allclose(model.pair_friction[:, :2], .2)
        np.testing.assert_allclose(model.geom_friction, self.original.geom_friction)
        root, _ = mutate(self.root, validate_config({"property": "damping"}), {"damping": .02})
        model = compile_root(root)
        data = mujoco.MjData(model)
        data.qvel[-2:] = [2, -3]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qfrc_passive[-2:], [-.04, .06])

    def test_mass_scale_preserves_shape_and_scales_inertia(self):
        """Double mass and inertia together while checking geometry stays unchanged."""
        root, _ = mutate(self.root, validate_config({"property": "mass_scale"}), {"mass_scale": 2})
        model = compile_root(root)
        np.testing.assert_allclose(model.body_mass[1], self.original.body_mass[1]*2)
        np.testing.assert_allclose(model.body_inertia[1], self.original.body_inertia[1]*2)
        np.testing.assert_array_equal(model.geom_size, self.original.geom_size)

    def test_invalid_dimensions_and_coupled_clearance_rejected(self):
        """Reject nonpositive dimensions/mass and a chassis that overlaps the wheels."""
        for settings in ({"chassis_mass_kg": -1}, {"chassis_size_m": [.2, .03, .17]}, {"wheel_radius_m": 0}):
            with self.assertRaises(ValueError):
                configured_baseline(self.root, settings)

    def test_config_typos_and_ambiguous_axes_rejected(self):
        """Check invalid field values, unsupported modes and duplicate/ambiguous input."""
        for config in ({"property": "com"}, {"direction": "horizontal"}, {"step_mm": 0},
                       {"direction": "fore_aft", "directions": ["vertical"]}, {"seconds_per_seed": float('nan')},
                       {"inertia_mode": "payload"}, {"seeds": [0, 0]}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_joint_grid_contains_combinations_and_exclusions(self):
        """Inspect the 11x11 initial grid, exclusions and written XML fingerprints."""
        config = validate_config({"directions": ["fore_aft", "vertical"], "sampling": "grid"})
        manifest = generate(self.root, config, self.directory / "grid")
        grid = [c for c in manifest["cases"] if c["kind"] == "mutation"]
        self.assertEqual(len(grid), 121)
        self.assertTrue(any(c["values"]["fore_aft"] != 0 and c["values"]["vertical"] != 0 and "xml" in c for c in grid))
        self.assertTrue(any("xml" not in c for c in grid))
        for case in manifest["cases"]:
            if "xml" in case:
                self.assertEqual(digest(self.directory / "grid" / case["xml"]), case["sha256"])

    def test_neutral_reset_repeatable_correct_quaternion_and_history(self):
        """Repeat a seeded reset after stepping and check orientation/history/contact."""
        env = NeutralEnv(self.path)
        try:
            first, _ = env.reset(seed=3)
            qpos = env.data.qpos.copy()
            self.assertLessEqual(abs(env.get_pitch()), .02 + 1e-12)
            self.assertGreater(env.data.body("robot_body").xmat.reshape(3, 3)[2, 2], .99)
            env.step(np.array([.1, -.1], dtype=np.float32))
            again, _ = env.reset(seed=3)
            np.testing.assert_array_equal(first, again)
            np.testing.assert_array_equal(qpos, env.data.qpos)
            self.assertEqual(again[1], 0.)
            self.assertEqual(env.data.ncon, 0)
        finally:
            env.close()

    def test_targets_remain_zero_and_wheel_sign_contract(self):
        """Check inherited wheel-speed/action signs and zero neutral command targets."""
        env = NeutralEnv(self.path)
        try:
            env.reset(seed=0)
            env.data.qvel[-2:] = [2, -2]
            self.assertEqual(env.get_wheel_speed(), 2)
            self.assertEqual(env.get_wheel_yaw(), 0)
            env.step(np.array([.1, -.2], dtype=np.float32))
            np.testing.assert_allclose(env.data.ctrl, [2.4, -2.8], atol=1e-7)
            self.assertEqual((env.target_wheel_speed, env.target_yaw), (0, 0))
        finally:
            env.close()

    def test_falls_do_not_shorten_cumulative_budget(self):
        """Use zero actions to provoke resets but still complete the two-second budget."""
        env = NeutralEnv(self.path, initial_pitch_rad=.2)
        try:
            row = trial(env, ZeroFixture(), validate_config({"seconds_per_seed": 2}), 4, self.directory / "trace.jsonl")
            self.assertEqual(row["simulated_seconds"], 2)
            self.assertGreater(row["resets"], 0)
            self.assertFalse(row["balance"])
            self.assertEqual(row["numerical"], "valid_observed")
        finally:
            env.close()

    def test_nonfinite_and_time_rollback_are_numerical_anomalies(self):
        """Inject NaN velocity and inconsistent time to check numerical-anomaly labels."""
        env = NeutralEnv(self.path)
        try:
            obs, _ = env.reset(seed=1)
            env.data.qvel[0] = np.nan
            flags = metrics_valid(env, obs, np.zeros(2), 1.)
            self.assertIn("nonfinite_qvel", flags)
            self.assertIn("unexpected_time_advance", flags)
        finally:
            env.close()

    def test_nonmonotonic_refinement_preserves_both_transitions(self):
        """Use fail/pass/fail synthetic rows to require both midpoint proposals."""
        config = validate_config({"property": "motor_strength", "seeds": [0]})
        cases = [{"id": str(i), "values": {"motor_strength": v}} for i, v in enumerate([0., .5, 1.])]
        rows = [{"case_id": str(i), "seed": 0, "balance": i == 1, "numerical": "valid_observed", "review_flags": []} for i in range(3)]
        proposals = refinement_candidates(rows, {"cases": cases}, config)
        self.assertEqual(proposals, [{"motor_strength": .25}, {"motor_strength": .75}])

    def test_worker_timeout_is_classified(self):
        """Give a child an intentionally tiny wall-time budget and require timeout."""
        status = execute_job({"type": "reference"}, self.directory / "timeout", .001)
        self.assertEqual(status["status"], "timeout")

    def test_retry_selection_preserves_original_results(self):
        """Read a selected retry without replacing old rows; reject escaped paths."""
        original = self.directory / "results.jsonl"
        original.write_text('{"seed": 0}\n', encoding="utf-8")
        attempt = self.directory / "attempt_2"
        attempt.mkdir()
        (attempt / "results.jsonl").write_text('{"seed": 0}\n{"seed": 1}\n', encoding="utf-8")
        (self.directory / "selected_attempt.json").write_text('{"directory": "attempt_2"}', encoding="utf-8")
        self.assertEqual([r["seed"] for r in read_rows(self.directory)], [0, 1])
        self.assertEqual(original.read_text(encoding="utf-8"), '{"seed": 0}\n')
        (self.directory / "selected_attempt.json").write_text('{"directory": "../outside"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_rows(self.directory)


if __name__ == "__main__":
    unittest.main()
