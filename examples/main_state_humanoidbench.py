import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from spectralrl.algo.state import *
from spectralrl.buffer.state import ReplayBuffer
from spectralrl.env.humanoidbench_env import HumanoidBenchEnv
from spectralrl.utils.logger import TensorboardLogger
from spectralrl.utils.utils import set_device, set_seed_everywhere


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # setup logger, seed and device
        self.logger = TensorboardLogger(
            "/".join([cfg.log_dir, cfg.algo.cls, cfg.name, cfg.task]),
            "_".join(["seed"+str(cfg.seed), cfg.name]),
            activate=not cfg.debug
        )
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        self.seed = set_seed_everywhere(cfg.seed)
        self.device = set_device(cfg.device)

        # setup envs
        self.train_env = HumanoidBenchEnv(cfg.task, self.seed, frame_skip=cfg.frame_skip)
        self.eval_env = HumanoidBenchEnv(cfg.task, self.seed + 100, frame_skip=cfg.frame_skip)
        self.frame_skip = cfg.frame_skip

        # setup buffer
        self.buffer = ReplayBuffer(
            obs_dim=self.train_env.observation_space.shape[0],
            action_dim=self.train_env.action_space.shape[0],
            max_size=cfg.buffer_size,
            norm_obs=cfg.norm_obs,
            norm_reward=cfg.norm_reward,
        )

        # setup agent
        algo_cls = {
            "sac": SAC,
            "td3": TD3,
            "ctrlsr_td3": CtrlSR_TD3,
            "diffsr_td3": DiffSR_TD3,
        }.get(cfg.algo.cls)
        self.agent = algo_cls(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            cfg.algo,
            self.device
        )

        self.global_step = 0
        self.global_episode = 0

    @property
    def global_frame(self):
        return self.global_step * self.frame_skip

    def train(self):
        cfg = self.cfg

        ep_length, ep_return = 0, 0
        obs, _ = self.train_env.reset()
        for t in trange(int(cfg.train_frames // self.frame_skip + 1), desc="main"):
            action = self.agent.select_action(self.buffer.normalize_obs(obs), self.global_step, deterministic=False)
            if self.global_frame < cfg.random_frames:
                action = self.train_env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            ep_length += 1
            ep_return += reward

            self.buffer.add(obs, action, next_obs, reward, terminated and not truncated)
            obs = next_obs

            if terminated or truncated:
                obs, _ = self.train_env.reset()
                self.global_episode += 1
                self.logger.log_scalars("rollout", {
                    "episode_return": ep_return,
                    "episode_length": ep_length
                }, step=self.global_frame)
                ep_length = ep_return = 0

            if self.global_frame < cfg.warmup_frames:
                train_metrics = {}
            else:
                train_metrics = self.agent.train_step(self.buffer, cfg.batch_size)

            if self.global_frame % cfg.log_frames == 0:
                self.logger.log_scalars("", train_metrics, step=self.global_frame)

            if self.global_frame % cfg.eval_frames == 0:
                eval_metrics = self.evaluate()
                self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)

            self.global_step += 1

    def evaluate(self):
        self.agent.train(False)
        all_lengths = []
        all_returns = []
        for i_episode in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_length = ep_return = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.agent.select_action(self.buffer.normalize_obs(obs), None, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_return += reward
                ep_length += 1
            all_lengths.append(ep_length)
            all_returns.append(ep_return)
        all_lengths = np.asarray(all_lengths)
        all_returns = np.asarray(all_returns)
        self.agent.train(True)
        metrics = {
            "return_mean": all_returns.mean(),
            "return_std": all_returns.std(),
            "length_mean": all_lengths.mean()
        }
        return metrics


@hydra.main(version_base=None, config_path="./config/state_humanoidbench", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
