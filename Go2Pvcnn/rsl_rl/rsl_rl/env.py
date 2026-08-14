"""Minimal VecEnv protocol used by the local RSL-RL runner."""


class VecEnv:
    """Base class for vectorized environments consumed by `OnPolicyRunner`."""

    num_envs: int
    num_actions: int
    max_episode_length: int
    device: str

    def get_observations(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError
