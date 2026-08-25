"""Reward and observation primitives for folded-load locomotion."""

from __future__ import annotations

import torch


ACTIVE_ACTION_DIM = 16
CANONICAL_ACTION_DIM = 23


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating torch.Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_actions(actions: torch.Tensor) -> torch.Tensor:
    actions = _require_finite("actions", actions)
    if actions.ndim != 2 or actions.shape[1] != CANONICAL_ACTION_DIM:
        raise ValueError("actions must be a [N, 23] tensor")
    return actions


def track_vx_error(error: torch.Tensor) -> torch.Tensor:
    """Exponential body-X tracking score with the approved 0.05 m/s scale."""

    error = _require_finite("vx_error", error)
    return torch.exp(-error.square() / 0.05**2)


def track_wz_error(error: torch.Tensor) -> torch.Tensor:
    """Exponential yaw-rate tracking score with the approved 0.15 rad/s scale."""

    error = _require_finite("wz_error", error)
    return torch.exp(-error.square() / 0.15**2)


def active_action_l2_tensor(actions: torch.Tensor) -> torch.Tensor:
    actions = _require_actions(actions)
    return actions[:, :ACTIVE_ACTION_DIM].square().sum(dim=-1)


def active_action_rate_l2_tensor(
    actions: torch.Tensor, previous_actions: torch.Tensor
) -> torch.Tensor:
    actions = _require_actions(actions)
    previous_actions = _require_actions(previous_actions)
    if actions.shape != previous_actions.shape:
        raise ValueError("actions and previous_actions must have matching shapes")
    return (actions[:, :ACTIVE_ACTION_DIM] - previous_actions[:, :ACTIVE_ACTION_DIM]).square().sum(dim=-1)


def folded_load_compat_base_error_b(env) -> torch.Tensor:
    """Finite compatibility padding for the legacy three-value target slot."""

    reference = env.scene["robot"].data.root_lin_vel_b
    return _require_finite("root_lin_vel_b", reference).new_zeros((env.num_envs, 3))


def folded_load_compat_ee_error_b(env) -> torch.Tensor:
    """Finite compatibility padding for the legacy six-value EE slot."""

    reference = env.scene["robot"].data.root_lin_vel_b
    return _require_finite("root_lin_vel_b", reference).new_zeros((env.num_envs, 6))


def folded_load_desired_twist_b(env) -> torch.Tensor:
    """Map episode command ``[vx, 0, wz]`` into the existing six-value slot."""

    commands = getattr(env, "folded_load_commands", None)
    if commands is None:
        reference = _require_finite(
            "root_lin_vel_b", env.scene["robot"].data.root_lin_vel_b
        )
        commands = reference.new_zeros((env.num_envs, 3))
        env.folded_load_commands = commands
    commands = _require_finite("folded_load_commands", commands)
    if commands.shape != (env.num_envs, 3):
        raise ValueError("folded_load_commands must have shape [num_envs, 3]")
    desired = commands.new_zeros((env.num_envs, 6))
    desired[:, 0] = commands[:, 0]
    desired[:, 1] = commands[:, 1]
    desired[:, 5] = commands[:, 2]
    return desired


def folded_load_track_vx(env) -> torch.Tensor:
    velocity = _require_finite(
        "root_lin_vel_b", env.scene["robot"].data.root_lin_vel_b
    )
    commands = _require_finite("folded_load_commands", env.folded_load_commands)
    return track_vx_error(velocity[:, 0] - commands[:, 0])


def folded_load_track_wz(env) -> torch.Tensor:
    velocity = _require_finite(
        "root_ang_vel_b", env.scene["robot"].data.root_ang_vel_b
    )
    commands = _require_finite("folded_load_commands", env.folded_load_commands)
    return track_wz_error(velocity[:, 2] - commands[:, 2])


def folded_load_lateral_velocity_l2(env) -> torch.Tensor:
    velocity = _require_finite(
        "root_lin_vel_b", env.scene["robot"].data.root_lin_vel_b
    )
    return velocity[:, 1].square()


def folded_load_active_action_l2(env) -> torch.Tensor:
    return active_action_l2_tensor(env.action_manager.action)


def folded_load_active_action_rate_l2(env) -> torch.Tensor:
    return active_action_rate_l2_tensor(
        env.action_manager.action, env.action_manager.prev_action
    )


__all__ = [
    "ACTIVE_ACTION_DIM",
    "CANONICAL_ACTION_DIM",
    "active_action_l2_tensor",
    "active_action_rate_l2_tensor",
    "folded_load_active_action_l2",
    "folded_load_active_action_rate_l2",
    "folded_load_compat_base_error_b",
    "folded_load_compat_ee_error_b",
    "folded_load_desired_twist_b",
    "folded_load_lateral_velocity_l2",
    "folded_load_track_vx",
    "folded_load_track_wz",
    "track_vx_error",
    "track_wz_error",
]
