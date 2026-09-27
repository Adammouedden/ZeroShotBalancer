import json
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import os
'''
Reward Terms Callback is a class that can be passed into the training script so that reward weights are
stored as hyper parameters. Developers can then see them listed in tensorboard during training.
Having reward terms and reward weights stored will help with reproducibility and future experiments.

The class first inherits from the BaseCallback stable baselines class

It has a function that is called once training starts, this will use the sb3 logger to record "hparams",
where it stores keys and values from the reward weights

It should also store a plain-file copy next to the event file

The next funciton is called on step, which will be able to read all reward terms so that devs can
see individual components of a reward function before they are summed into one scalar

'''


class RewardTermsCallback(BaseCallback):
    def __init__(self, reward_weights: dict):
        super().__init__()
        self.reward_weights = reward_weights

    def _on_training_start(self):
        #Hparam accepts any dictionary in the format of {string : float/int/bool etc}
        # This means that self.reward_weights will work with our intended format
        # metric_dict is required and must be non-empty: TensorBoard's HParams tab pairs each run's
        # hparams with these metrics (looked up by tag from the logged scalars)
        # exclude: only the tensorboard writer can handle HParam, the others raise FormatUnsupportedError
        self.logger.record(
            "hparams",
            HParam(self.reward_weights, {"eval/mean_reward": 0, "rollout/ep_rew_mean": 0}),
            exclude=("stdout", "log", "json", "csv"),
        )

        with open(os.path.join(self.logger.get_dir(), "reward_weights.json"), "w") as f:
            json.dump(self.reward_weights, f, indent=2)

        # running per-episode sum of each reward term, one dict per parallel env
        self._ep_sums = [defaultdict(float) for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            for k, v in info.get("reward_terms", {}).items():
                self._ep_sums[i][k] += v
            if done:
                for k, v in self._ep_sums[i].items():
                    self.logger.record_mean(f"reward_terms/{k}", v)
                self._ep_sums[i].clear()

        return True
