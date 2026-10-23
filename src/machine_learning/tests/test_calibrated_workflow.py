"""Regression checks for checkpoint preprocessing and shared-ticket evaluations."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from click.testing import CliRunner
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from visibility import experiment_tracking
from visibility.observation_normalization import (
    NormalizedCheckpointCallback, NormalizedEvalCallback,
    normalization_state, store_normalization, wrap_for_policy,
)


def normalized_cartpole():
    return gym.wrappers.NormalizeObservation(Monitor(
        gym.make("CartPole-v1", max_episode_steps=5)
    ))


def test_saved_checkpoints_keep_matching_frozen_statistics(tmp_path):
    train_env, eval_env = normalized_cartpole(), normalized_cartpole()
    eval_env.update_running_mean = False
    try:
        model = PPO("MlpPolicy", train_env, n_steps=16, batch_size=8,
                    n_epochs=1, device="cpu", seed=7)
        evaluator = NormalizedEvalCallback(
            eval_env, train_env, eval_freq=16, n_eval_episodes=1,
            best_model_save_path=str(tmp_path), verbose=0,
        )
        checkpoint = NormalizedCheckpointCallback(
            train_env, save_freq=16, save_path=str(tmp_path), name_prefix="periodic",
        )
        model.learn(32, callback=CallbackList([checkpoint, evaluator]))
        files = [tmp_path / "best_model.zip", tmp_path / "periodic_16_steps.zip",
                 tmp_path / "periodic_32_steps.zip"]
        for path in files:
            saved = PPO.load(path, device="cpu")
            assert saved.normalization_required
            raw = gym.make("CartPole-v1")
            restored = wrap_for_policy(raw, saved)
            try:
                snapshot = normalization_state(restored)
                sample = np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float32)
                expected = np.float32((sample - np.array(snapshot["mean"])) /
                                      np.sqrt(np.array(snapshot["var"]) + snapshot["epsilon"]))
                np.testing.assert_array_equal(restored.observation(sample), expected)
                restored.reset(seed=3)
                restored.step(0)
                assert normalization_state(restored) == snapshot
            finally:
                restored.close()
        assert eval_env.obs_rms.count == train_env.obs_rms.count
        assert not eval_env.update_running_mean
        # Statistics are snapshots, not aliases subsequently changed by training.
        early = PPO.load(files[1]).observation_normalization
        late = PPO.load(files[2]).observation_normalization
        assert early["count"] < late["count"]
        store_normalization(model, train_env)
        sample = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        train_env.update_running_mean = False
        expected_action, _ = model.predict(train_env.observation(sample), deterministic=True)
        model.save(tmp_path / "final_model.zip")
        reloaded = PPO.load(tmp_path / "final_model.zip")
        restored = wrap_for_policy(gym.make("CartPole-v1"), reloaded)
        try:
            actual_action, _ = reloaded.predict(restored.observation(sample), deterministic=True)
            np.testing.assert_array_equal(actual_action, expected_action)
        finally:
            restored.close()
    finally:
        train_env.close()
        eval_env.close()


def test_required_missing_statistics_fail_and_legacy_remains_raw():
    from types import SimpleNamespace
    env = gym.make("CartPole-v1")
    try:
        assert wrap_for_policy(env, SimpleNamespace()) is env
        with pytest.raises(ValueError, match="requires observation statistics"):
            wrap_for_policy(env, SimpleNamespace(normalization_required=True))
    finally:
        env.close()


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(experiment_tracking, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.syspath_prepend(str(root / "evaluation"))
    import evaluate
    return evaluate


def test_video_cli_passes_world_and_separates_same_ticket_outputs(evaluation, monkeypatch):
    calls = []

    def record(model, output, schedule, env_id, duration, seed, reset_grace, xml_file):
        calls.append((env_id, xml_file, output))
        Path(output).write_bytes(b"fixture")
        return {"duration_s": 0.1, "falls": 0, "xml_file": xml_file}

    monkeypatch.setattr(evaluation, "record_video", record)
    runner = CliRunner()
    base = ["video", "-out", "STR-118", "-m", "fixture.zip", "-x", "base_world_calibrated.xml"]
    for env in evaluation.VIDEO_ENVS:
        result = runner.invoke(evaluation.cli, base + ["-e", env])
        assert result.exit_code == 0, result.output
    assert calls[0][2] != calls[1][2]
    assert all(c[1] == "base_world_calibrated.xml" for c in calls)
    assert all(Path(c[2]).with_suffix('.json').is_file() for c in calls)
    result = runner.invoke(evaluation.cli, base + ["-e", evaluation.VIDEO_ENVS[0]])
    assert result.exit_code != 0
    assert len(calls) == 2


def test_compare_cli_accepts_both_world_options(evaluation, monkeypatch, tmp_path):
    import compare
    from concurrent.futures import Future
    calls = []

    class InlinePool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args, **kwargs):
            calls.append(kwargs["xml_file"])
            future = Future()
            future.set_result({"duration_s": 0.1, "falls": 0, "mean_speed_error": 0,
                               "mean_yaw_error": 0, "mean_abs_pitch_deg": 0,
                               "max_reward": 0, "total_reward": 0,
                               "xml_file": kwargs["xml_file"]})
            return future

    monkeypatch.setattr(compare, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(compare, "side_by_side", lambda *args: None)
    model = tmp_path / "fixture.zip"
    model.touch()
    result = CliRunner().invoke(compare.compare, [
        "-out", "STR-118", "-a", str(model), "-b", str(model),
        "--env-a", "ReferenceBaselineTrainingEnv", "--env-b", "ControlAgentTrainingEnv",
        "--xml-a", "base_world_calibrated.xml", "--xml-b", "base_world_calibrated.xml",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["base_world_calibrated.xml"] * 2


@pytest.fixture
def reference_training(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "training/reference_baseline"))
    import reference_sb_rl
    for setting, folder in (("MODEL_DIR", "models"), ("LOG_DIR", "logs"),
                            ("RECORDING_DIR", "movies")):
        monkeypatch.setattr(reference_sb_rl, setting, str(tmp_path / folder))
    return reference_sb_rl


def test_reference_evaluation_preserves_live_training_state(reference_training, monkeypatch, tmp_path):
    """Exercise real training and evaluation callbacks on two calibrated MuJoCo worlds."""
    from unittest.mock import Mock

    trainer = reference_training
    original_make = gym.make
    created = []

    def short_environment(*args, **kwargs):
        env = original_make(*args, max_episode_steps=4, **kwargs)
        env.close = Mock(wraps=env.close)
        created.append(env)
        return env

    monkeypatch.setattr(trainer.gym, "make", short_environment)
    monkeypatch.setattr(trainer, "algorithm_factory", lambda algorithm_name, env: PPO(
        "MlpPolicy", env, n_steps=16, batch_size=8, n_epochs=1,
        seed=7, device="cpu", tensorboard_log=trainer.LOG_DIR,
    ))
    checks = []

    def physical_state(env):
        data = env.unwrapped.data
        return np.concatenate(([data.time], data.qpos, data.qvel, data.ctrl))

    class CheckingEvaluation(trainer.EvalCallback):
        def __init__(self, env, **kwargs):
            kwargs.update(eval_freq=16, n_eval_episodes=1)
            super().__init__(env, **kwargs)

        def _on_step(self):
            if self.n_calls % self.eval_freq:
                return super()._on_step()
            training_env = self.training_env.envs[0]
            evaluation_env = self.eval_env.envs[0]
            assert training_env.unwrapped.data is not evaluation_env.unwrapped.data
            assert training_env.unwrapped.model is not evaluation_env.unwrapped.model
            before = physical_state(training_env)
            keep_training = super()._on_step()
            np.testing.assert_array_equal(physical_state(training_env), before)
            checks.append(self.num_timesteps)
            return keep_training

    monkeypatch.setattr(trainer, "EvalCallback", CheckingEvaluation)
    result = CliRunner().invoke(trainer.cli, [
        "-a", "PPO", "train", "-e", "ReferenceBaselineTrainingEnv", "-t", "32",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert checks == [16, 32]
    assert len(created) == 2
    for env in created:
        assert Path(env.unwrapped.fullpath).name == "base_world_calibrated.xml"
        env.close.assert_called_once()
    model_dir = Path(trainer.MODEL_DIR) / "ReferenceBaselineTrainingEnv_PPO"
    for filename in ("best_model.zip", "final_model.zip"):
        saved = PPO.load(model_dir / filename, device="cpu")
        assert saved.training_xml_file == trainer.TRAINING_XML
        assert saved.training_evaluation_separate_env
        assert not getattr(saved, "normalization_required", False)
        if filename == "final_model.zip":
            assert saved.num_timesteps == 32
    assert list(Path(trainer.LOG_DIR).rglob("events.out.tfevents.*"))
    assert list(Path(trainer.RECORDING_DIR).rglob("*.mp4"))


def test_reference_closes_both_simulators_on_training_error(reference_training, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    trainer = reference_training
    original_make = gym.make
    created = []

    def tracked_environment(*args, **kwargs):
        env = original_make(*args, **kwargs)
        env.close = Mock(wraps=env.close)
        created.append(env)
        return env

    monkeypatch.setattr(trainer.gym, "make", tracked_environment)
    monkeypatch.setattr(trainer, "algorithm_factory", lambda **kwargs: SimpleNamespace(
        learn=Mock(side_effect=RuntimeError("training interrupted")),
    ))
    result = CliRunner().invoke(trainer.cli, [
        "-a", "PPO", "train", "-e", "ReferenceBaselineTrainingEnv", "-t", "32",
    ])
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "training interrupted"
    assert len(created) == 2
    for env in created:
        env.close.assert_called_once()
    assert not list(Path(trainer.MODEL_DIR).rglob("*.zip"))


def test_reference_rejects_existing_run_before_creating_simulators(reference_training, monkeypatch):
    trainer = reference_training
    model_dir = Path(trainer.MODEL_DIR) / "ReferenceBaselineTrainingEnv_PPO"
    model_dir.mkdir(parents=True)
    previous = model_dir / "best_model.zip"
    previous.write_bytes(b"existing model")

    def unexpected_creation(*args, **kwargs):
        pytest.fail("A populated model directory must be rejected before environment creation")

    monkeypatch.setattr(trainer.gym, "make", unexpected_creation)
    result = CliRunner().invoke(trainer.cli, [
        "-a", "PPO", "train", "-e", "ReferenceBaselineTrainingEnv", "-t", "32",
    ])
    assert result.exit_code != 0
    assert "already contains a run" in result.output
    assert previous.read_bytes() == b"existing model"
