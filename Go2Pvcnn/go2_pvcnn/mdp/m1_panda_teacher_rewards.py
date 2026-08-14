"""Lightweight reward helpers for M1 + Panda Teacher balance training."""

from __future__ import annotations

import torch


def base_xy_drift_l2(env, asset_cfg) -> torch.Tensor:
    """Penalize squared horizontal displacement from each environment origin."""
    asset = env.scene[asset_cfg.name]
    delta_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.sum(delta_xy.square(), dim=1)


def selected_joint_velocity_l2(env, asset_cfg) -> torch.Tensor:
    """Penalize velocity only for the joints resolved by ``asset_cfg``."""
    asset = env.scene[asset_cfg.name]
    selected = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(selected.square(), dim=1)


def selected_joint_torques_l2(env, asset_cfg) -> torch.Tensor:
    """Penalize applied torque only for the joints resolved by ``asset_cfg``."""
    asset = env.scene[asset_cfg.name]
    selected = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(selected.square(), dim=1)


def _require_residual_state(env, attribute: str, label: str) -> torch.Tensor:
    value = getattr(env, attribute, None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"missing {label} tensor published by Teacher wrapper")
    if value.ndim != 2 or value.shape[0] != env.num_envs:
        raise RuntimeError(
            f"{label} shape must be ({env.num_envs}, channels), got {tuple(value.shape)}"
        )
    if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{label} must contain finite floating values")
    return value


def teacher_residual_l2(env) -> torch.Tensor:
    """Penalize the current trainable normalized residual amplitude."""
    current = _require_residual_state(
        env, "m1_teacher_trainable_residual", "trainable residual"
    )
    return torch.sum(current.square(), dim=1)


def teacher_residual_rate_l2(env) -> torch.Tensor:
    """Penalize changes in the trainable normalized residual."""
    current = _require_residual_state(
        env, "m1_teacher_trainable_residual", "trainable residual"
    )
    previous = _require_residual_state(
        env,
        "m1_teacher_previous_trainable_residual",
        "previous trainable residual",
    )
    if previous.shape != current.shape:
        raise RuntimeError(
            "current and previous trainable residual must have the same shape"
        )
    return torch.sum((current - previous).square(), dim=1)


__all__ = [
    "base_xy_drift_l2",
    "selected_joint_torques_l2",
    "selected_joint_velocity_l2",
    "teacher_residual_l2",
    "teacher_residual_rate_l2",
]
