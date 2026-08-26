"""Grouped 103-observation actor critic for the eight-channel residual policy."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .residual_observation import RESIDUAL_OBSERVATION_DIM


RESIDUAL_ACTION_DIM = 8
M1_SLICE = slice(0, 59)
ARM_SLICE = slice(59, 79)
WRENCH_SLICE = slice(79, 85)
CONTEXT_SLICE = slice(85, 103)


def _encoder(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ELU())


class ResidualActorCritic(nn.Module):
    """Independent grouped actor/critic encoders with the RSL-RL policy API."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        init_noise_std: float = 0.01,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"unsupported ResidualActorCritic arguments: {unknown}")
        if num_actor_obs != RESIDUAL_OBSERVATION_DIM:
            raise ValueError("num_actor_obs must be 103")
        if num_critic_obs != RESIDUAL_OBSERVATION_DIM:
            raise ValueError("num_critic_obs must be 103")
        if num_actions != RESIDUAL_ACTION_DIM:
            raise ValueError("num_actions must be 8")
        if (
            isinstance(init_noise_std, bool)
            or not isinstance(init_noise_std, (int, float))
            or not math.isfinite(float(init_noise_std))
            or float(init_noise_std) <= 0.0
        ):
            raise ValueError("init_noise_std must be finite and positive")
        if noise_std_type not in ("scalar", "log"):
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.actor_m1 = _encoder(59, 128)
        self.actor_arm = _encoder(20, 64)
        self.actor_wrench = _encoder(6, 32)
        self.actor_context = _encoder(18, 32)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 8), nn.Tanh()
        )
        self.critic_m1 = _encoder(59, 128)
        self.critic_arm = _encoder(20, 64)
        self.critic_wrench = _encoder(6, 32)
        self.critic_context = _encoder(18, 32)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        )
        self.noise_std_type = noise_std_type
        initial = float(init_noise_std) * torch.ones(RESIDUAL_ACTION_DIM)
        if noise_std_type == "scalar":
            self.std = nn.Parameter(initial)
        else:
            self.log_std = nn.Parameter(torch.log(initial))
        self.register_buffer(
            "active_action_mask", torch.ones(RESIDUAL_ACTION_DIM, dtype=torch.bool), persistent=False
        )
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _validate_observations(name: str, observations: torch.Tensor) -> None:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if observations.ndim != 2 or observations.shape[1] != RESIDUAL_OBSERVATION_DIM:
            raise ValueError(f"{name} must have width 103")
        if not observations.is_floating_point():
            raise TypeError(f"{name} must have floating dtype")
        if not torch.isfinite(observations).all().item():
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _grouped(
        observations: torch.Tensor,
        m1: nn.Module,
        arm: nn.Module,
        wrench: nn.Module,
        context: nn.Module,
    ) -> torch.Tensor:
        return torch.cat(
            (
                m1(observations[:, M1_SLICE]),
                arm(observations[:, ARM_SLICE]),
                wrench(observations[:, WRENCH_SLICE]),
                context(observations[:, CONTEXT_SLICE]),
            ),
            dim=-1,
        )

    def _actor(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_observations("observations", observations)
        encoded = self._grouped(
            observations,
            self.actor_m1,
            self.actor_arm,
            self.actor_wrench,
            self.actor_context,
        )
        return self.actor_head(encoded)

    def _critic(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_observations("critic_observations", observations)
        encoded = self._grouped(
            observations,
            self.critic_m1,
            self.critic_arm,
            self.critic_wrench,
            self.critic_context,
        )
        return self.critic_head(encoded)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been updated")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been updated")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been updated")
        return self.distribution.entropy().sum(dim=-1)

    @property
    def noise_parameter(self) -> torch.Tensor:
        return self.std if self.noise_std_type == "scalar" else self.log_std

    @property
    def effective_action_std(self) -> torch.Tensor:
        return self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._actor(observations)
        self.distribution = Normal(mean, self.effective_action_std.expand_as(mean))

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._actor(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been updated")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._critic(critic_observations)

    def reset(self, dones=None) -> None:
        return None

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def clip_std(self, min=None, max=None) -> None:
        if self.noise_std_type == "scalar":
            self.std.copy_(self.std.clip(min=min, max=max))
            return
        log_min = None if min is None else math.log(float(min))
        log_max = None if max is None else math.log(float(max))
        self.log_std.copy_(self.log_std.clip(min=log_min, max=log_max))


__all__ = ["ResidualActorCritic", "RESIDUAL_ACTION_DIM"]
