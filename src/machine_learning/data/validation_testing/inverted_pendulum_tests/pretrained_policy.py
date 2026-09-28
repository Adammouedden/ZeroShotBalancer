"""Download, identify and evaluate a fixed SB3 SAC or PPO checkpoint (no training)."""

import hashlib
from pathlib import Path
import shutil
import warnings

DEFAULT_REPO = "farama-minari/InvertedPendulum-v5-SAC-expert"
DEFAULT_FILENAME = "invertedpendulum-v5-sac-expert.zip"


def digest(path):
    """Return the SHA-256 hex fingerprint of the file at path without changing it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_policy(directory, checkpoint=None, repo=DEFAULT_REPO,
                   filename=DEFAULT_FILENAME, revision="main", vecnormalize=None, algorithm="SAC"):
    """Resolve and copy a frozen pendulum policy into a new run's policy folder.

    Args:
        directory: Existing run folder; its policy/ subfolder must not exist.
        checkpoint: Local weights path, or None to download from Hugging Face.
        repo, filename, revision: Hub source identifiers when downloading;
            revision is resolved to an immutable commit before obtaining bytes.
        vecnormalize: Optional saved observation-normalization file to copy.
        algorithm: SAC or PPO, matching the supplied checkpoint.

    Returns:
        Paths, hashes and provenance metadata for SB3Controller. Downloading
        needs network access; local input only copies files. No training occurs.

    The environment labels here are pendulum-specific. The self-balancer uses
    runner.policy_metadata() for its own PPO and shares SB3Controller below.
    """
    if algorithm not in {"SAC", "PPO"}:
        raise ValueError(f"Unsupported policy algorithm: {algorithm}")
    metadata = {"algorithm": algorithm, "deterministic": True, "device": "cpu",
                "evaluation_environment": "InvertedPendulum-v5",
                "observation_order": ["cart_position", "pole_angle", "cart_velocity", "pole_velocity"],
                "normalization": "VecNormalize" if vecnormalize else "none (raw observations)"}
    if checkpoint is None:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(repo, revision=revision)
        # Keep Hub's nested snapshot paths below the Windows path-length limit.
        cache = Path(__file__).resolve().parents[3] / ".venv" / "hf-cache"
        checkpoint = hf_hub_download(repo_id=repo, filename=filename, revision=info.sha,
                                     cache_dir=str(cache))
        metadata.update(source="huggingface", repo_id=repo, filename=filename,
                        requested_revision=revision, resolved_revision=info.sha)
    else:
        checkpoint = Path(checkpoint).resolve()
        metadata.update(source="local", original_checkpoint=str(checkpoint))
    target = Path(directory) / "policy"
    target.mkdir()
    saved = target / "checkpoint.zip"
    shutil.copyfile(checkpoint, saved)
    metadata.update(checkpoint_path=str(saved.resolve()), checkpoint_sha256=digest(saved))
    if vecnormalize:
        normalization = target / "vecnormalize.pkl"
        shutil.copyfile(vecnormalize, normalization)
        metadata.update(vecnormalize_path=str(normalization.resolve()),
                        vecnormalize_sha256=digest(normalization))
    return metadata


class SB3Controller:
    """Frozen SAC/PPO inference shared by pendulum and self-balancer evaluations."""

    def __init__(self, metadata, env):
        """Load a hashed checkpoint on CPU and validate its spaces against env.

        metadata supplies algorithm, checkpoint_path/checkpoint_sha256 and
        optional VecNormalize path/hash. env supplies action/observation spaces
        and, if needed, the normalization wrapper's environment. Reject changed
        artifacts or mismatched shapes/bounds with ValueError. Normalization
        statistics are frozen; compatibility notes retain a known loader warning.
        """
        import numpy as np
        import stable_baselines3
        from stable_baselines3 import PPO, SAC
        import torch

        if digest(metadata["checkpoint_path"]) != metadata["checkpoint_sha256"]:
            raise ValueError("Policy checkpoint changed since run preparation.")
        torch.set_num_threads(1)
        loader = {"SAC": SAC, "PPO": PPO}[metadata["algorithm"]]
        self.compatibility_notes = []
        with warnings.catch_warnings(record=True) as load_warnings:
            warnings.simplefilter("always")
            self.model = loader.load(metadata["checkpoint_path"], device="cpu")
        for warning in load_warnings:
            message = str(warning.message)
            # The published checkpoint was serialized with NumPy 1.x. NumPy 2
            # still loads this alias; its deprecation is not a dynamics warning.
            # Keep the notice, and propagate every other warning to the runner.
            if (issubclass(warning.category, DeprecationWarning)
                    and message.startswith("numpy.core.numeric is deprecated and has been renamed to numpy._core.numeric")):
                if message not in self.compatibility_notes:
                    self.compatibility_notes.append(message)
            else:
                warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
        # Shapes alone are insufficient: verify action bounds as well.
        for name in ("observation_space", "action_space"):
            expected, actual = getattr(env, name), getattr(self.model, name)
            if (actual.shape != expected.shape
                    or not np.array_equal(actual.low, expected.low)
                    or not np.array_equal(actual.high, expected.high)):
                raise ValueError(f"{metadata['algorithm']} {name} does not match InvertedPendulum-v5.")
        self.normalizer = None
        if "vecnormalize_path" in metadata:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            if digest(metadata["vecnormalize_path"]) != metadata["vecnormalize_sha256"]:
                raise ValueError("Observation normalization file changed.")
            self.normalizer = VecNormalize.load(metadata["vecnormalize_path"], DummyVecEnv([lambda: env]))
            self.normalizer.training = False
            self.normalizer.norm_reward = False
        self.versions = {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__}

    def predict(self, observation):
        """Return a deterministic action for one observation with optional scaling.

        Apply saved observation normalization when present; neither policy
        weights nor normalization statistics are updated by this call.
        """
        if self.normalizer is not None:
            observation = self.normalizer.normalize_obs(observation)
        action, _ = self.model.predict(observation, deterministic=True)
        return action
