"""Frozen deployable observation and residual-action contracts for Student S1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


STUDENT_OBSERVATION_DIM = 100
STUDENT_HISTORY_LENGTH = 10
STUDENT_ACTION_DIM = 23
LEG_SLICE = slice(0, 12)
WHEEL_SLICE = slice(12, 16)
ARM_SLICE = slice(16, 23)


def _require_positive_finite(name: str, value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StudentActionScaleCfg:
    leg_position_rad: float = 0.25
    wheel_velocity_radps: float = 8.0
    arm_position_rad: float = 0.20
    leg_slew_per_step: float = 0.02
    wheel_slew_per_step: float = 0.50
    arm_slew_per_step: float = 0.01

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _require_positive_finite(name, value)


@dataclass(frozen=True)
class StudentNominalCommand:
    position: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True)
class StudentActionCommand:
    normalized_action: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    saturated: torch.Tensor


def _validate_tensor(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, int] | None = None,
    reference: torch.Tensor | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != STUDENT_ACTION_DIM:
        raise ValueError(
            f"{name} must have shape [E,{STUDENT_ACTION_DIM}], got {tuple(value.shape)}"
        )
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if reference is not None:
        if value.device != reference.device:
            raise ValueError(
                f"{name} must be on device {reference.device}, got {value.device}"
            )
        if value.dtype != reference.dtype:
            raise TypeError(
                f"{name} must have dtype {reference.dtype}, got {value.dtype}"
            )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_nominal(nominal: StudentNominalCommand) -> tuple[int, int]:
    if not isinstance(nominal, StudentNominalCommand):
        raise TypeError("nominal must be a StudentNominalCommand")
    _validate_tensor("nominal.position", nominal.position)
    shape = tuple(nominal.position.shape)
    _validate_tensor(
        "nominal.velocity",
        nominal.velocity,
        shape=shape,
        reference=nominal.position,
    )
    return shape


def _group_vector(
    cfg: StudentActionScaleCfg,
    reference: torch.Tensor,
    *,
    slew: bool,
) -> torch.Tensor:
    values = (
        [cfg.leg_slew_per_step if slew else cfg.leg_position_rad] * 12
        + [cfg.wheel_slew_per_step if slew else cfg.wheel_velocity_radps] * 4
        + [cfg.arm_slew_per_step if slew else cfg.arm_position_rad] * 7
    )
    return reference.new_tensor(values).unsqueeze(0)


def teacher_residual_label(
    q_des: torch.Tensor,
    qd_des: torch.Tensor,
    nominal: StudentNominalCommand,
    cfg: StudentActionScaleCfg,
) -> torch.Tensor:
    """Convert safe Teacher position/velocity targets into normalized residual labels."""
    shape = _validate_nominal(nominal)
    _validate_tensor(
        "q_des", q_des, shape=shape, reference=nominal.position
    )
    _validate_tensor(
        "qd_des", qd_des, shape=shape, reference=nominal.position
    )
    scale = _group_vector(cfg, nominal.position, slew=False)
    label = torch.zeros_like(nominal.position)
    label[:, LEG_SLICE] = (
        q_des[:, LEG_SLICE] - nominal.position[:, LEG_SLICE]
    ) / scale[:, LEG_SLICE]
    label[:, WHEEL_SLICE] = (
        qd_des[:, WHEEL_SLICE] - nominal.velocity[:, WHEEL_SLICE]
    ) / scale[:, WHEEL_SLICE]
    label[:, ARM_SLICE] = (
        q_des[:, ARM_SLICE] - nominal.position[:, ARM_SLICE]
    ) / scale[:, ARM_SLICE]
    return label


def apply_student_residual(
    normalized_action: torch.Tensor,
    nominal: StudentNominalCommand,
    cfg: StudentActionScaleCfg,
    *,
    previous_action: torch.Tensor,
) -> StudentActionCommand:
    """Apply amplitude and physical slew limits before reconstructing safe targets."""
    shape = _validate_nominal(nominal)
    _validate_tensor(
        "normalized_action",
        normalized_action,
        shape=shape,
        reference=nominal.position,
    )
    _validate_tensor(
        "previous_action",
        previous_action,
        shape=shape,
        reference=nominal.position,
    )
    scale = _group_vector(cfg, nominal.position, slew=False)
    slew_limit = _group_vector(cfg, nominal.position, slew=True)
    amplitude_clipped = normalized_action.abs() > 1.0
    target_physical = normalized_action.clamp(-1.0, 1.0) * scale
    previous_physical = previous_action.clamp(-1.0, 1.0) * scale
    requested_delta = target_physical - previous_physical
    limited_delta = torch.maximum(
        torch.minimum(requested_delta, slew_limit), -slew_limit
    )
    applied_physical = previous_physical + limited_delta
    applied_normalized = applied_physical / scale
    saturated = amplitude_clipped | (requested_delta.abs() > slew_limit)

    position = nominal.position.clone()
    velocity = nominal.velocity.clone()
    position[:, LEG_SLICE] += applied_physical[:, LEG_SLICE]
    velocity[:, WHEEL_SLICE] += applied_physical[:, WHEEL_SLICE]
    position[:, ARM_SLICE] += applied_physical[:, ARM_SLICE]
    return StudentActionCommand(
        normalized_action=applied_normalized,
        position=position,
        velocity=velocity,
        saturated=saturated,
    )


__all__ = [
    "STUDENT_ACTION_DIM",
    "STUDENT_HISTORY_LENGTH",
    "STUDENT_OBSERVATION_DIM",
    "StudentActionCommand",
    "StudentActionScaleCfg",
    "StudentNominalCommand",
    "apply_student_residual",
    "teacher_residual_label",
]
