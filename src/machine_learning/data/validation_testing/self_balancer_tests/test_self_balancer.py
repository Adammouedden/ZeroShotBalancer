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
        return np.zeros(2, dtype=np.float32)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (HERE / "generated").mkdir(exist_ok=True)
        cls.root, cls.original = explicit_baseline()

    def setUp(self):
        # Normal inherited Windows ACLs; tempfile's private 0700 ACL can exclude
        # the sandbox token even inside the writable workspace.
        self.directory = (HERE / "generated" / ("test_" + uuid.uuid4().hex)).resolve()
        assert self.directory.is_relative_to((HERE / "generated").resolve())
        self.directory.mkdir()
        self.path = self.directory / "baseline.xml"
        save_xml(copy.deepcopy(self.root), self.path)

    def tearDown(self):
        assert self.directory.is_relative_to((HERE / "generated").resolve())
        shutil.rmtree(self.directory)

    def test_explicit_conversion_all_compiled_numeric_arrays(self):
        self.assertEqual(compare_models(self.original, compile_root(self.root)), {})

    def test_renamed_reference_initializes_modular_rewards(self):
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
        config = validate_config({"direction": "fore_aft"})
        root, physical = mutate(self.root, config, {"fore_aft": .001})
        model = compile_root(root)
        np.testing.assert_allclose(model.body_ipos[1] - self.original.body_ipos[1], [0, .001, 0], atol=1e-15)
        self.assertEqual(physical["status"], "unverified")

    def test_variance_bound_matches_hand_box_formula_and_scales(self):
        config = validate_config({"directions": ["fore_aft", "vertical"], "sampling": "grid"})
        for size in ([.1, .037, .171], [.11, .06, .3]):
            root = configured_baseline(self.root, {"chassis_size_m": size, "chassis_mass_kg": .8})
            result = calculate(compile_root(root), config)
            for b in result["bounds"]:
                self.assertAlmostEqual(b["necessary_moment"][1], size[b["axis"]] / 2 * math.sqrt(2/3), places=12)

    def test_vertical_grid_has_exact_zero_and_unchanged_compiled_reference(self):
        config = validate_config({"direction": "vertical"})
        bound = calculate(self.original, config)["bounds"][0]
        values = axis_values(bound, config)
        self.assertEqual(len(values), 11)
        self.assertEqual(values[5], 0.)
        root, _ = mutate(self.root, config, {"vertical": values[5]})
        np.testing.assert_array_equal(compile_root(root).body_ipos, self.original.body_ipos)

    def test_support_and_extreme_moment_rejections(self):
        for offset in ([0, .03, 0], [0, .0185, 0], [0, 0, .0855]):
            self.assertEqual(feasibility(self.original, offset)["status"], "known_infeasible")
        self.assertEqual(feasibility(self.original, [0, 0, 0])["status"], "nominal_geometry_derived")

    def test_unknown_caps_are_not_fabricated(self):
        result = calculate(self.original, validate_config({"property": "mass_scale"}))
        self.assertEqual(result["bounds"][0]["status"], "requires_input")

    def test_motor_cap_changes_measured_saturated_force(self):
        for scale in (0, .5, 1):
            root, _ = mutate(self.root, validate_config({"property": "motor_strength"}), {"motor_strength": scale})
            model = compile_root(root)
            data = mujoco.MjData(model)
            data.ctrl[:] = 70
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.actuator_force, .65 * scale, atol=1e-12)

    def test_friction_and_damping_reach_effective_fields(self):
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
        root, _ = mutate(self.root, validate_config({"property": "mass_scale"}), {"mass_scale": 2})
        model = compile_root(root)
        np.testing.assert_allclose(model.body_mass[1], self.original.body_mass[1]*2)
        np.testing.assert_allclose(model.body_inertia[1], self.original.body_inertia[1]*2)
        np.testing.assert_array_equal(model.geom_size, self.original.geom_size)

    def test_invalid_dimensions_and_coupled_clearance_rejected(self):
        for settings in ({"chassis_mass_kg": -1}, {"chassis_size_m": [.2, .03, .17]}, {"wheel_radius_m": 0}):
            with self.assertRaises(ValueError):
                configured_baseline(self.root, settings)

    def test_config_typos_and_ambiguous_axes_rejected(self):
        for config in ({"property": "com"}, {"direction": "horizontal"}, {"step_mm": 0},
                       {"direction": "fore_aft", "directions": ["vertical"]}, {"seconds_per_seed": float('nan')},
                       {"inertia_mode": "payload"}, {"seeds": [0, 0]}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_joint_grid_contains_combinations_and_exclusions(self):
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
        config = validate_config({"property": "motor_strength", "seeds": [0]})
        cases = [{"id": str(i), "values": {"motor_strength": v}} for i, v in enumerate([0., .5, 1.])]
        rows = [{"case_id": str(i), "seed": 0, "balance": i == 1, "numerical": "valid_observed", "review_flags": []} for i in range(3)]
        proposals = refinement_candidates(rows, {"cases": cases}, config)
        self.assertEqual(proposals, [{"motor_strength": .25}, {"motor_strength": .75}])

    def test_worker_timeout_is_classified(self):
        status = execute_job({"type": "reference"}, self.directory / "timeout", .001)
        self.assertEqual(status["status"], "timeout")

    def test_retry_selection_preserves_original_results(self):
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
