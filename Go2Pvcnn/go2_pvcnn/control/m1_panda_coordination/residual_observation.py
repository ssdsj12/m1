"""Named, deterministic observation groups for a future 8D residual policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


M1_OBSERVATION_DIM = 59
ARM_OBSERVATION_DIM = 20
MOUNT_WRENCH_DIM = 6
TASK_OBSERVATION_DIM = 6
STABILITY_OBSERVATION_DIM = 4
PREVIOUS_RESIDUAL_DIM = 8
RESIDUAL_OBSERVATION_DIM = (
    M1_OBSERVATION_DIM
    + ARM_OBSERVATION_DIM
    + MOUNT_WRENCH_DIM
    + TASK_OBSERVATION_DIM
    + STABILITY_OBSERVATION_DIM
    + PREVIOUS_RESIDUAL_DIM
)


@dataclass(frozen=True)
class ResidualObservationParts:
    m1_state: torch.Tensor
    arm_state: torch.Tensor
    filtered_mount_wrench: torch.Tensor
    task_state: torch.Tensor
    sigma_min: torch.Tensor
    joint_limit_margin_min: torch.Tensor
    joint_limit_margin_mean: torch.Tensor
    support_margin: torch.Tensor
    previous_residual: torch.Tensor


@dataclass(frozen=True)
class ResidualObservation:
    groups: tuple[str, ...]
    m1_state: torch.Tensor
    arm_state: torch.Tensor
    filtered_mount_wrench: torch.Tensor
    task_state: torch.Tensor
    stability: torch.Tensor
    previous_residual: torch.Tensor
    flat: torch.Tensor


_FIELD_WIDTHS = (
    ("m1_state", M1_OBSERVATION_DIM),
    ("arm_state", ARM_OBSERVATION_DIM),
    ("filtered_mount_wrench", MOUNT_WRENCH_DIM),
    ("task_state", TASK_OBSERVATION_DIM),
    ("sigma_min", 1),
    ("joint_limit_margin_min", 1),
    ("joint_limit_margin_mean", 1),
    ("support_margin", 1),
    ("previous_residual", PREVIOUS_RESIDUAL_DIM),
)


def build_residual_observation(parts: ResidualObservationParts) -> ResidualObservation:
    """Validate named groups and flatten them without changing caller tensors."""

    if not isinstance(parts, ResidualObservationParts):
        raise TypeError("parts must be a ResidualObservationParts")
    reference: torch.Tensor | None = None
    batch_shape: tuple[int, ...] | None = None
    cloned: dict[str, torch.Tensor] = {}
    for name, width in _FIELD_WIDTHS:
        value = getattr(parts, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f"{name} must end with width {width}")
        if not value.is_floating_point():
            raise TypeError(f"{name} dtype must be floating")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
        if reference is None:
            reference = value
            batch_shape = value.shape[:-1]
        else:
            if value.shape[:-1] != batch_shape:
                raise ValueError(f"{name} batch dimensions must match m1_state")
            if value.dtype != reference.dtype:
                raise TypeError(f"{name} dtype must match m1_state")
            if value.device != reference.device:
                raise ValueError(f"{name} device must match m1_state")
        cloned[name] = value.clone()
    stability = torch.cat(
        (
            cloned["sigma_min"],
            cloned["joint_limit_margin_min"],
            cloned["joint_limit_margin_mean"],
            cloned["support_margin"],
        ),
        dim=-1,
    )
    groups = (
        "m1_state",
        "arm_state",
        "filtered_mount_wrench",
        "task_state",
        "stability",
        "previous_residual",
    )
    flat = torch.cat(
        (
            cloned["m1_state"],
            cloned["arm_state"],
            cloned["filtered_mount_wrench"],
            cloned["task_state"],
            stability,
            cloned["previous_residual"],
        ),
        dim=-1,
    )
    return ResidualObservation(
        groups=groups,
        m1_state=cloned["m1_state"],
        arm_state=cloned["arm_state"],
        filtered_mount_wrench=cloned["filtered_mount_wrench"],
        task_state=cloned["task_state"],
        stability=stability.clone(),
        previous_residual=cloned["previous_residual"],
        flat=flat.clone(),
    )


__all__ = [
    "ARM_OBSERVATION_DIM",
    "M1_OBSERVATION_DIM",
    "RESIDUAL_OBSERVATION_DIM",
    "STABILITY_OBSERVATION_DIM",
    "TASK_OBSERVATION_DIM",
    "ResidualObservation",
    "ResidualObservationParts",
    "build_residual_observation",
]
