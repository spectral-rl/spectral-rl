import gymnasium as gym


class GymEnv(object):
    def __init__(
        self,
        task: str,
        seed: int,
    ) -> None:
        super().__init__()

        self.task = task
        self.seed = seed

        self.env = gym.make(self.task)
        self.env.reset(seed=self.seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
