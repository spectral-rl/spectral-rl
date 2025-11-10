import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from spectralrl.algo.visual import *
from spectralrl.buffer.visual import VisualReplayBuffer
from spectralrl.utils.logger import TensorboardLogger
from spectralrl.utils.utils import set_device, set_seed_everywhere
from spectralrl.utils.video import VideoRecorder

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


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
        self.recorder = VideoRecorder(
            self.logger.output_dir if cfg.save_video else None
        )
        self.seed = set_seed_everywhere(cfg.seed)
        self.device = set_device(cfg.device)

        self.suite, self.task = cfg.task.split("-", 1)
        self.frame_skip = cfg.frame_skip
        if self.suite == "metaworld":
            from spectralrl.env.metaworld_env import MetaWorldEnv
            self.train_env = MetaWorldEnv(self.task, self.seed, cfg.frame_skip, cfg.frame_stack)
            self.eval_env = MetaWorldEnv(self.task, self.seed + 100, cfg.frame_skip, cfg.frame_stack)
        elif self.suite == "dmc":
            from spectralrl.env.dmc_env import DMControlEnv
            self.train_env = DMControlEnv(self.task, self.seed, True, cfg.frame_skip, cfg.frame_stack)
            self.eval_env = DMControlEnv(self.task, self.seed + 100, True, cfg.frame_skip, cfg.frame_stack)
        else:
            raise NotImplementedError(f"Unrecognized domain: {self.suite}.")

        # create buffer
        data_specs = {
            "obs": self.train_env.observation_space.shape,
            "action": self.train_env.action_space.shape,
            "reward": (1, ),
            "discount": (1, ),
        }
        self.buffer = VisualReplayBuffer(
            buffer_size=cfg.buffer_size,
            nstep=cfg.nstep,
            discount=cfg.discount,
            frame_stack=cfg.frame_stack,
            data_specs=data_specs
        )

        # setup agent
        algo_cls = {
            "drqv2": DrQv2,
            "ctrlsr_td3": CtrlSR_TD3,
            "diffsr_td3": DiffSR_TD3,
        }.get(cfg.algo.cls)
        self.agent = algo_cls(
            self.train_env.observation_space.shape,
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

        ep_length, ep_return, ep_success = 0, 0, 0
        obs, _ = self.train_env.reset()
        for t in trange(cfg.train_frames // cfg.frame_skip + 1, desc="main"):
            action = self.agent.select_action(obs, self.global_step, deterministic=False)
            if self.global_frame < cfg.random_frames:
                action = self.train_env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            ep_length += 1
            ep_return += reward
            ep_success += info.get("success", 0.0)

            self.buffer.add(obs, action, next_obs, reward, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, _ = self.train_env.reset()
                self.global_episode += 1
                self.logger.log_scalars("rollout", {
                    "return": ep_return,
                    "success": float(ep_success>=1.0),
                    "episode_frame": ep_length*cfg.frame_skip
                }, step=self.global_frame)
                ep_length = ep_return = ep_success = 0

            if self.global_frame < cfg.warmup_frames:
                train_metrics = {}
            else:
                # pretrain
                if cfg.pretrain_steps > 0 and self.global_frame == cfg.warmup_frames:
                    for i_pretrain in trange(cfg.pretrain_steps, desc="pretrain"):
                        pretrain_metrics = self.agent.pretrain_step(self.buffer, cfg.batch_size, step=i_pretrain)
                        if i_pretrain % 1000 == 0:
                            self.logger.log_scalars("pretrain", pretrain_metrics, step=i_pretrain)
                # train
                for i_update in range(cfg.utd):
                    train_metrics = self.agent.train_step(self.buffer, cfg.batch_size, self.global_step)

            if self.global_frame % cfg.log_frames == 0:
                self.logger.log_scalars("", train_metrics, step=self.global_frame)

            if self.global_frame % cfg.eval_frames == 0:
                eval_metrics = self.evaluate()
                agent_eval_metrics, agent_grid = self.agent.evaluate(self.buffer)
                eval_metrics.update(agent_eval_metrics)
                if agent_grid is not None:
                    self.logger.log_image("eval", agent_grid, step=self.global_frame)
                self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
                self.logger.info(eval_metrics)

            self.global_step += 1

    def evaluate(self):
        self.agent.train(False)
        self.recorder.init()
        all_lengths = []
        all_returns = []
        all_success = []
        for i_episode in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset()
            self.recorder.record(self.eval_env)
            ep_length = ep_return = ep_success = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.agent.select_action(obs, self.global_step, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                self.recorder.record(self.eval_env)
                ep_return += reward
                ep_length += 1
                ep_success += info.get("success", 0.0)
            all_lengths.append(ep_length)
            all_returns.append(ep_return)
            all_success.append(float(ep_success>=1.0))
        self.recorder.save(f"eval_{self.global_frame}.mp4")
        all_lengths = np.asarray(all_lengths)
        all_returns = np.asarray(all_returns)
        all_success = np.asarray(all_success, dtype=np.float32)
        self.agent.train(True)
        metrics = {
            "return_mean": all_returns.mean(),
            "return_std": all_returns.std(),
            "length_mean": all_lengths.mean(),
            "success_mean": all_success.mean()
        }
        return metrics


@hydra.main(version_base=None, config_path="./config/visual", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
