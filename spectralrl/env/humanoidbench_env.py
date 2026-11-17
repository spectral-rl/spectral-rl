"""HumanoidBench (Sferrazza et al., 2024) environments, e.g. ``h1-walk-v0`` or ``h1hand-run-v0``.

Installation. HumanoidBench pins gymnasium==0.29.1 and mujoco==3.1.6, which conflict with this
repo's pins (gymnasium>=1.0, mujoco==3.2.7). It runs fine on the newer versions, so install it
without its dependencies (an editable install, as upstream recommends, since its sub-packages
lack ``__init__.py`` files):

    git clone https://github.com/carlosferrazza/humanoid-bench && cd humanoid-bench
    git checkout cb1189039151c8aadaaa987b442da54383c87fab
    pip install --no-deps -e .
    pip install "jax[cpu]"    # imported unconditionally by humanoid_bench.wrappers
"""
import sys

import gymnasium as gym
import numpy as np

# tasks that need extra constructor arguments (as in the HumanoidBench and SimbaV2 training scripts)
_EXTRA_KWARGS = {"h1hand-package-v0": {"policy_path": None}}


def _import_humanoid_bench():
    """Import humanoid_bench, which registers its tasks with gymnasium.

    HumanoidBench vendors dm_control's table of MuJoCo struct sizes for mujoco==3.1.6; under
    mujoco==3.2.7 a few of those fields were renamed and the import fails. The installed
    dm_control ships the same table (same format) for the installed mujoco, so we use that one.
    """
    if "humanoid_bench" in sys.modules:
        return
    from dm_control.mujoco.wrapper.mjbindings import sizes
    sys.modules.setdefault("humanoid_bench.dmc_deps.dmc_sizes", sizes)
    import humanoid_bench  # noqa: F401


class HumanoidBenchEnv(object):
    """A HumanoidBench task with action repeat.

    Follows the protocol of public single-environment off-policy implementations on HumanoidBench
    (e.g. SimbaV2): actions in [-1, 1] (mapped to the actuator ranges by HumanoidBench), each
    action repeated ``frame_skip`` times with rewards summed, episodes truncated after the task's
    registered limit (1000 simulator steps for the locomotion tasks) and terminated when the robot
    falls. The default (privileged) state observation is used.
    """

    def __init__(
        self,
        task: str,
        seed: int,
        frame_skip: int = 2,
    ) -> None:
        super().__init__()
        _import_humanoid_bench()

        self.task = task
        self.seed = seed
        self.frame_skip = frame_skip

        self.env = gym.make(task, render_mode=None, **_EXTRA_KWARGS.get(task, {}))
        self.env.reset(seed=seed)

        obs_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = self.env.action_space
        self.action_space.seed(seed)
        assert np.all(self.action_space.low == -1.0) and np.all(self.action_space.high == 1.0)
        self.max_ep_timesteps = (self.env.spec.max_episode_steps + frame_skip - 1) // frame_skip

    def reset(self):
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        reward = 0.0
        for _ in range(self.frame_skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break
        return obs.astype(np.float32), reward, terminated, truncated, info
