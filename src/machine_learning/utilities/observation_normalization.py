"""Keep Gymnasium observation statistics inside each SB3 policy checkpoint."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def normalization_state(env: gym.wrappers.NormalizeObservation) -> dict:
    """Return a JSON-serializable snapshot, independent of later training updates."""
    return {
        "version": 1,
        "mean": env.obs_rms.mean.tolist(),
        "var": env.obs_rms.var.tolist(),
        "count": float(env.obs_rms.count),
        "epsilon": float(env.epsilon),
    }


def restore_normalization(env, state: dict, *, training: bool = False):
    """Validate and restore a snapshot; evaluation never updates its statistics."""
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Missing or unsupported checkpoint observation normalization.")
    mean = np.asarray(state["mean"], dtype=np.float64)
    var = np.asarray(state["var"], dtype=np.float64)
    count, epsilon = float(state["count"]), float(state["epsilon"])
    shape = env.observation_space.shape
    if (mean.shape != shape or var.shape != shape
            or not np.isfinite(mean).all() or not np.isfinite(var).all()
            or (var < 0).any() or not np.isfinite(count) or count <= 0
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("Checkpoint normalization is invalid or has the wrong observation shape.")
    env.obs_rms.mean = mean.copy()
    env.obs_rms.var = var.copy()
    env.obs_rms.count = count
    env.epsilon = epsilon
    env.update_running_mean = training
    return env


def store_normalization(model, env):
    """SB3 saves these attributes in the same ZIP as the matching policy weights."""
    model.normalization_required = True
    model.observation_normalization = normalization_state(env)


def wrap_for_policy(env, model):
    """Restore new policies' preprocessing; legacy unnormalized policies stay raw."""
    state = getattr(model, "observation_normalization", None)
    if state is None:
        if getattr(model, "normalization_required", False):
            raise ValueError("This policy requires observation statistics, but its checkpoint has none.")
        return env
    return restore_normalization(gym.wrappers.NormalizeObservation(env), state)


class NormalizedCheckpointCallback(CheckpointCallback):
    """Snapshot normalization immediately before each periodic model save."""

    def __init__(self, normalization_env, **kwargs):
        super().__init__(**kwargs)
        self.normalization_env = normalization_env

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            store_normalization(self.model, self.normalization_env)
        return super()._on_step()


class NormalizedEvalCallback(EvalCallback):
    """Evaluate in a separate env with frozen statistics and save a matched best model."""

    def __init__(self, eval_env, normalization_env, **kwargs):
        self.normalization_env = normalization_env
        self.normalized_eval_env = eval_env
        super().__init__(eval_env, **kwargs)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            store_normalization(self.model, self.normalization_env)
            restore_normalization(
                self.normalized_eval_env, self.model.observation_normalization,
            )
        return super()._on_step()
