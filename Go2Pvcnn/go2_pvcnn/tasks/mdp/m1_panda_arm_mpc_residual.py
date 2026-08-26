"""Stability-first reward and small stationary EE curriculum primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualRewardSignals:
    roll: torch.Tensor
    pitch: torch.Tensor
    base_height_error: torch.Tensor
    support_margin: torch.Tensor
    wheel_contact_count: torch.Tensor
    joint_margin: torch.Tensor
    hard_failure: torch.Tensor
    ee_position_error: torch.Tensor
    ee_orientation_error: torch.Tensor
    wrench_error: torch.Tensor
    wheel_slip: torch.Tensor
    residual: torch.Tensor
    previous_residual: torch.Tensor
    intervention: torch.Tensor


@dataclass(frozen=True)
class ResidualReward:
    total: torch.Tensor
    stability: torch.Tensor
    task: torch.Tensor
    tracking_penalty: torch.Tensor
    regularization: torch.Tensor
    gate: torch.Tensor


def _validated(signals: ResidualRewardSignals) -> tuple[int, torch.device, torch.dtype]:
    if not isinstance(signals, ResidualRewardSignals):
        raise TypeError("signals must be ResidualRewardSignals")
    reference = signals.roll
    if not isinstance(reference, torch.Tensor) or reference.ndim != 1 or not reference.is_floating_point():
        raise TypeError("roll must be a floating vector")
    count = reference.shape[0]
    for name in (
        "pitch", "base_height_error", "support_margin", "wheel_contact_count",
        "joint_margin", "hard_failure", "ee_position_error", "ee_orientation_error",
        "wrench_error", "wheel_slip", "intervention",
    ):
        value = getattr(signals, name)
        if not isinstance(value, torch.Tensor) or value.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},)")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} dtype and device must match roll")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    for name in ("residual", "previous_residual"):
        value = getattr(signals, name)
        if not isinstance(value, torch.Tensor) or value.shape != (count, 8):
            raise ValueError(f"{name} must have shape ({count}, 8)")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} dtype and device must match roll")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if not torch.isfinite(reference).all().item():
        raise ValueError("roll must contain only finite values")
    return count, reference.device, reference.dtype


def stability_gate(signals: ResidualRewardSignals) -> torch.Tensor:
    """Return a multiplicative task gate that closes before hard instability."""

    _validated(signals)
    tilt = torch.sqrt(signals.roll.square() + signals.pitch.square())
    tilt_gate = torch.exp(-torch.square(tilt / math.radians(10.0)))
    height_gate = torch.exp(-torch.square(signals.base_height_error / 0.08))
    support_gate = torch.clamp(signals.support_margin / 0.05, 0.0, 1.0)
    joint_gate = torch.clamp(signals.joint_margin / 0.10, 0.0, 1.0)
    contact_gate = torch.clamp(signals.wheel_contact_count / 4.0, 0.0, 1.0)
    return torch.clamp(
        tilt_gate * height_gate * support_gate * joint_gate * contact_gate,
        0.0,
        1.0,
    )


def compute_residual_reward(signals: ResidualRewardSignals) -> ResidualReward:
    _validated(signals)
    gate = stability_gate(signals)
    stability = 4.0 * gate - 20.0 * signals.hard_failure
    task_score = torch.exp(
        -torch.square(signals.ee_position_error / 0.03)
        -torch.square(signals.ee_orientation_error / 0.12)
    )
    task = 2.0 * gate * task_score
    tracking_penalty = -0.2 * signals.wrench_error - 0.5 * signals.wheel_slip
    magnitude = signals.residual.square().mean(dim=-1)
    rate = (signals.residual - signals.previous_residual).square().mean(dim=-1)
    regularization = -0.02 * magnitude - 0.01 * rate - 0.2 * signals.intervention
    total = stability + task + tracking_penalty + regularization
    return ResidualReward(
        total=total,
        stability=stability,
        task=task,
        tracking_penalty=tracking_penalty,
        regularization=regularization,
        gate=gate,
    )


class SmallEeTrajectory:
    """Deterministic stationary-base six-axis target with approved small bounds."""

    def __init__(self, *, seed: int, scale: float) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or not 0.0 <= float(scale) <= 1.0:
            raise ValueError("scale must be finite and in [0,1]")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._phase = 2.0 * torch.pi * torch.rand(6, generator=generator)
        self._frequency = 0.05 + 0.15 * torch.rand(6, generator=generator)
        self._scale = float(scale)
        self.base_command = torch.zeros(3)

    def sample(self, center: torch.Tensor, time_s: float) -> torch.Tensor:
        if not isinstance(center, torch.Tensor) or center.shape != (6,) or not center.is_floating_point():
            raise ValueError("center must be a floating tensor with shape (6,)")
        if not torch.isfinite(center).all().item():
            raise ValueError("center must contain only finite values")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(float(time_s)) or float(time_s) < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        amplitude = center.new_tensor((0.03, 0.03, 0.03, 0.08, 0.08, 0.08))
        phase = self._phase.to(device=center.device, dtype=center.dtype)
        frequency = self._frequency.to(device=center.device, dtype=center.dtype)
        return center + self._scale * amplitude * torch.sin(
            2.0 * torch.pi * frequency * float(time_s) + phase
        )


__all__ = [
    "ResidualReward", "ResidualRewardSignals", "SmallEeTrajectory",
    "compute_residual_reward", "stability_gate",
]
