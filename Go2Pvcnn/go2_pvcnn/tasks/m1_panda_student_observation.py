"""Frozen 100-value deployable observation assembly for Student S1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    STUDENT_OBSERVATION_DIM,
)


@dataclass(frozen=True)
class StudentObservationParts:
    root_linear_velocity_b: torch.Tensor
    root_angular_velocity_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    m1_joint_position: torch.Tensor
    m1_joint_velocity: torch.Tensor
    panda_arm_position: torch.Tensor
    panda_arm_velocity: torch.Tensor
    ee_pose_error_b: torch.Tensor
    desired_ee_twist_b: torch.Tensor
    wheel_contact: torch.Tensor
    mount_wrench_b: torch.Tensor
    previous_action: torch.Tensor


_LAYOUT = (
    ("root_linear_velocity_b", 3),
    ("root_angular_velocity_b", 3),
    ("projected_gravity_b", 3),
    ("m1_joint_position", 16),
    ("m1_joint_velocity", 16),
    ("panda_arm_position", 7),
    ("panda_arm_velocity", 7),
    ("ee_pose_error_b", 6),
    ("desired_ee_twist_b", 6),
    ("wheel_contact", 4),
    ("mount_wrench_b", 6),
    ("previous_action", 23),
)


def build_student_observation(parts: StudentObservationParts) -> torch.Tensor:
    if not isinstance(parts, StudentObservationParts):
        raise TypeError("parts must be StudentObservationParts")
    tensors: list[torch.Tensor] = []
    batch: int | None = None
    device: torch.device | None = None
    for name, width in _LAYOUT:
        value = getattr(parts, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [E,{width}]")
        if not value.dtype.is_floating_point and name != "wheel_contact":
            raise TypeError(f"{name} must use a floating dtype")
        if batch is None:
            batch, device = value.shape[0], value.device
        if value.shape[0] != batch or value.device != device:
            raise ValueError("all observation parts must share batch and device")
        value = value.to(dtype=torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        tensors.append(value)
    observation = torch.cat(tensors, dim=-1)
    if observation.shape[-1] != STUDENT_OBSERVATION_DIM:
        raise RuntimeError("Student observation layout drifted from width 100")
    return observation


__all__ = ["StudentObservationParts", "build_student_observation"]
