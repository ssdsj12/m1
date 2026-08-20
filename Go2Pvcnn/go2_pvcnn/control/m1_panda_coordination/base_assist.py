"""Bounded planar M1 assistance for Panda null-space recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BaseAssistCfg:
    max_speed_xy: float = 0.05
    max_yaw_rate: float = 0.1
    max_accel_xy: float = 0.2
    max_yaw_accel: float = 0.3
    max_displacement_xy: float = 0.25
    enable_margin: float = 0.1
    disable_margin: float = 0.2
    minimum_sigma: float = 0.05
    radius_gain: float = 1.0


@dataclass(frozen=True)
class BaseAssistDecision:
    base_velocity: torch.Tensor
    active: bool
    reason: str
    arm_margin_before: torch.Tensor
    arm_margin_after: torch.Tensor


def _vector(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def compute_base_assist(
    *,
    base_pose: torch.Tensor,
    arrived_base_pose: torch.Tensor,
    target_base_pose: torch.Tensor,
    arm_margin_before: torch.Tensor,
    arm_margin_after: torch.Tensor,
    sigma_min: torch.Tensor,
    previous_velocity: torch.Tensor,
    dt: float,
    cfg: BaseAssistCfg | None = None,
) -> BaseAssistDecision:
    """Choose a bounded planar correction only when it improves arm margin."""
    cfg = cfg or BaseAssistCfg()
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    for name, value in (
        ("base_pose", base_pose),
        ("arrived_base_pose", arrived_base_pose),
        ("target_base_pose", target_base_pose),
        ("previous_velocity", previous_velocity),
    ):
        _vector(name, value)
    for name, value in (("arm_margin_before", arm_margin_before), ("arm_margin_after", arm_margin_after), ("sigma_min", sigma_min)):
        if not isinstance(value, torch.Tensor) or value.shape != ():
            raise ValueError(f"{name} must be scalar")
    zero = torch.zeros_like(previous_velocity)
    finite = all(bool(torch.isfinite(value).all()) for value in (base_pose, arrived_base_pose, target_base_pose, previous_velocity, arm_margin_before, arm_margin_after, sigma_min))
    if not finite:
        return BaseAssistDecision(zero, False, "non_finite", arm_margin_before.clone(), arm_margin_after.clone())
    if float(arm_margin_before) >= cfg.disable_margin and float(sigma_min) >= cfg.minimum_sigma:
        return BaseAssistDecision(zero, False, "margin_sufficient", arm_margin_before.clone(), arm_margin_after.clone())
    if float(arm_margin_after) <= float(arm_margin_before):
        return BaseAssistDecision(zero, False, "no_improvement", arm_margin_before.clone(), arm_margin_after.clone())

    displacement = base_pose[:2] - arrived_base_pose[:2]
    remaining = cfg.max_displacement_xy - float(torch.linalg.vector_norm(displacement))
    if remaining <= 0.0:
        return BaseAssistDecision(zero, False, "displacement_limit", arm_margin_before.clone(), arm_margin_after.clone())
    direction = target_base_pose - base_pose
    direction[2] = torch.remainder(direction[2] + torch.pi, 2 * torch.pi) - torch.pi
    xy_norm = torch.linalg.vector_norm(direction[:2])
    if float(xy_norm) > 1.0e-9:
        direction[:2] = direction[:2] / xy_norm
    speed = min(cfg.max_speed_xy, remaining / dt)
    velocity = direction * 0.0
    velocity[:2] = direction[:2] * speed
    velocity[2] = torch.clamp(direction[2] * cfg.radius_gain, -cfg.max_yaw_rate, cfg.max_yaw_rate)
    delta = velocity - previous_velocity
    velocity[:2] = previous_velocity[:2] + torch.clamp(delta[:2], -cfg.max_accel_xy * dt, cfg.max_accel_xy * dt)
    velocity[2] = previous_velocity[2] + torch.clamp(delta[2], -cfg.max_yaw_accel * dt, cfg.max_yaw_accel * dt)
    return BaseAssistDecision(velocity, True, "null_space_margin", arm_margin_before.clone(), arm_margin_after.clone())


__all__ = ["BaseAssistCfg", "BaseAssistDecision", "compute_base_assist"]
