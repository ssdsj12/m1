"""Residual mixture-density model for the offline fingertip expert."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .contracts import MIXTURE_COMPONENTS, MODEL_INPUT_DIM, PRIOR_HORIZON, MixtureDistribution


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class FingertipMixtureNet(nn.Module):
    """A geometry-only residual MLP returning four diagonal Gaussian components."""

    def __init__(self, *, hidden: Sequence[int] = (512, 512, 512)) -> None:
        super().__init__()
        widths = tuple(hidden)
        if not widths or any(type(width) is not int or width <= 0 for width in widths):
            raise ValueError("hidden must contain positive integer widths")
        self.hidden = widths
        self.input = nn.Sequential(nn.Linear(MODEL_INPUT_DIM, widths[0]), nn.SiLU())
        self.transitions = nn.ModuleList(
            nn.Sequential(nn.Linear(left, right), nn.SiLU())
            for left, right in zip(widths, widths[1:])
        )
        self.blocks = nn.ModuleList(_ResidualBlock(width) for width in widths)
        dimensions = MIXTURE_COMPONENTS * PRIOR_HORIZON * 5 * 3
        self.head = nn.Linear(widths[-1], MIXTURE_COMPONENTS + dimensions * 2)

    def forward(self, network_input: torch.Tensor) -> MixtureDistribution:
        if not isinstance(network_input, torch.Tensor):
            raise TypeError("network_input must be a tensor")
        if network_input.dtype != torch.float32 or network_input.ndim != 2:
            raise TypeError("network_input must be a float32 rank-two tensor")
        if network_input.shape[-1] != MODEL_INPUT_DIM:
            raise ValueError(f"network_input must have trailing dimension {MODEL_INPUT_DIM}")
        if not torch.isfinite(network_input).all().item():
            raise ValueError("network_input must be finite")
        value = self.input(network_input)
        for index, block in enumerate(self.blocks):
            value = block(value)
            if index < len(self.transitions):
                value = self.transitions[index](value)
        output = self.head(value)
        logits = output[:, :MIXTURE_COMPONENTS]
        geometry = output[:, MIXTURE_COMPONENTS:]
        count = MIXTURE_COMPONENTS * PRIOR_HORIZON * 5 * 3
        mean = geometry[:, :count].reshape(-1, MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3)
        log_std = geometry[:, count:].reshape(-1, MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3)
        return MixtureDistribution(logits=logits, mean=mean, log_std=log_std.clamp(-7.0, 3.0))


def mixture_log_prob(distribution: MixtureDistribution, target: torch.Tensor) -> torch.Tensor:
    """Return one exact log density for every batch item in a mixture distribution."""

    if not isinstance(distribution, MixtureDistribution):
        raise TypeError("distribution must be a MixtureDistribution")
    if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
        raise TypeError("target must be a float32 tensor")
    expected = (*distribution.logits.shape[:-1], PRIOR_HORIZON, 5, 3)
    if target.shape != expected:
        raise ValueError(f"target must have shape {expected}")
    if target.device != distribution.mean.device or not torch.isfinite(target).all().item():
        raise ValueError("target must be finite and share the distribution device")
    target_components = target.unsqueeze(-4)
    inv_var = torch.exp(-2.0 * distribution.log_std)
    log_component = -0.5 * (
        (target_components - distribution.mean).square() * inv_var
        + 2.0 * distribution.log_std
        + math.log(2.0 * math.pi)
    )
    log_component = log_component.flatten(start_dim=-3).sum(dim=-1)
    return torch.logsumexp(log_component + distribution.logits.log_softmax(dim=-1), dim=-1)


def mixture_nll(distribution: MixtureDistribution, target: torch.Tensor) -> torch.Tensor:
    """Mean negative log likelihood over batch items, without per-dimension scaling."""

    return -mixture_log_prob(distribution, target).mean()


def temporal_regularizer(distribution: MixtureDistribution, *, dt: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite acceleration and jerk penalties of all mixture component means."""

    if type(dt) is not float or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite float")
    velocity = distribution.mean
    acceleration = (velocity[:, :, 1:] - velocity[:, :, :-1]) / dt
    jerk = (acceleration[:, :, 1:] - acceleration[:, :, :-1]) / dt
    acceleration_penalty = acceleration.square().mean()
    jerk_penalty = jerk.square().mean()
    if not bool(torch.isfinite(acceleration_penalty + jerk_penalty).item()):
        raise FloatingPointError("temporal regularizer is non-finite")
    return acceleration_penalty, jerk_penalty


__all__ = [
    "FingertipMixtureNet",
    "mixture_log_prob",
    "mixture_nll",
    "temporal_regularizer",
]
