import os
from collections import deque

import gym
import numpy as np


class MetaWorldEnv(object):
    def __init__(
        self,
        task: str,
        seed: int,
        frame_skip: int,
        frame_stack: int,
        horizon: int = 500,
        image_size: int = 84,
        camera: str = "corner2"
    ) -> None:
        super().__init__()

        self.task = task
        self.seed = seed
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.horizon = horizon
        self.image_size = image_size
        self.camera = camera

        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        )
        os.environ["MUJOCO_GL"] = "egl"
        task = f"{task}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]

        self.env = env_cls(seed=seed)
        self.env._freeze_rand_vec = False

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.frame_stack*3, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.env.action_space.shape),
            high=np.ones(self.env.action_space.shape),
            dtype=self.env.action_space.dtype
        )
        self.max_ep_timesteps = (horizon + frame_skip - 1) // frame_skip

        self.queue = deque(maxlen=self.frame_stack)

    def get_obs(self):
        obs = self.render(self.image_size)
        return np.transpose(obs, (2, 0, 1))

    def reset(self):
        self.t = 0
        if self.camera == "corner2":
            self.env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        self.env.reset()
        obs = self.get_obs()
        for _ in range(self.frame_stack):
            self.queue.append(obs)

        return np.concatenate(self.queue), {}

    def step(self, action):
        self.t += 1
        action = action.astype(np.float32)

        success = 0.0
        reward = 0.0
        for _ in range(self.frame_skip):
            _, rew, done, info = self.env.step(action)
            success += float(info["success"])
            reward += rew or 0.0
        success = min(success, 1.0)
        obs = self.get_obs()
        self.queue.append(obs)
        return np.concatenate(self.queue), reward, False, self.t == self.max_ep_timesteps, {"success": success}

    def render(self, size: int):
        return self.env.sim.render(size, size, mode="offscreen", camera_name=self.camera)
