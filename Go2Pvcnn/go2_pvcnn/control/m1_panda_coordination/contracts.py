"""Shared dimensions, joint ordering, and tensor validation for M1 + Panda WBC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


COORD_DOF = 10
GENERALIZED_DOF = 31
CONTROLLED_DOF = 23

M1_LEG_JOINT_NAMES = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
)
M1_WHEEL_JOINT_NAMES = (
    "FAR_FOOT_JOINT",
    "FBL_FOOT_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_FOOT_JOINT",
)
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
PANDA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")


@dataclass(frozen=True)
class PandaLinkDynamicsState:
    """Atomic CPU-float64 snapshot of Panda subtree rigid-body dynamics."""

    link_names: tuple[str, ...]
    mass: torch.Tensor
    link_pos_w: torch.Tensor
    link_quat_w: torch.Tensor
    com_pos_w: torch.Tensor
    com_quat_w: torch.Tensor
    inertia_com_local: torch.Tensor
    linear_vel_w: torch.Tensor
    angular_vel_w: torch.Tensor
    linear_acc_w: torch.Tensor
    angular_acc_w: torch.Tensor

    def __post_init__(self) -> None:
        if not self.link_names or any(
            not isinstance(name, str) or not name for name in self.link_names
        ):
            raise ValueError("link_names must contain at least one non-empty name")
        if len(set(self.link_names)) != len(self.link_names):
            raise ValueError("link_names must be unique")
        links = len(self.link_names)
        require_tensor(
            "mass", self.mass, trailing_shape=(links,), dtype=torch.float64, device="cpu"
        )
        for name, value, trailing_shape in (
            ("link_pos_w", self.link_pos_w, (links, 3)),
            ("link_quat_w", self.link_quat_w, (links, 4)),
            ("com_pos_w", self.com_pos_w, (links, 3)),
            ("com_quat_w", self.com_quat_w, (links, 4)),
            ("inertia_com_local", self.inertia_com_local, (links, 3, 3)),
            ("linear_vel_w", self.linear_vel_w, (links, 3)),
            ("angular_vel_w", self.angular_vel_w, (links, 3)),
            ("linear_acc_w", self.linear_acc_w, (links, 3)),
            ("angular_acc_w", self.angular_acc_w, (links, 3)),
        ):
            require_tensor(
                name,
                value,
                trailing_shape=trailing_shape,
                dtype=torch.float64,
                device="cpu",
            )
        if not torch.all(self.mass > 0.0).item():
            raise ValueError("mass must be strictly positive")
        for name, quaternion in (
            ("link_quat_w", self.link_quat_w),
            ("com_quat_w", self.com_quat_w),
        ):
            quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
            if not torch.allclose(
                quaternion_norm,
                torch.ones_like(quaternion_norm),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError(f"{name} must contain unit quaternions")

    @property
    def link_count(self) -> int:
        return len(self.link_names)


def _indices_for(names: tuple[str, ...], name_to_index: dict[str, int]) -> torch.Tensor:
    return torch.tensor([name_to_index[name] for name in names], dtype=torch.long)


@dataclass(frozen=True)
class WbcJointMap:
    """Runtime articulation indices expressed in the canonical WBC order."""

    legs: torch.Tensor
    wheels: torch.Tensor
    panda_arm: torch.Tensor
    fingers: torch.Tensor
    controlled: torch.Tensor

    @classmethod
    def resolve(cls, actual_joint_names: Sequence[str]) -> "WbcJointMap":
        if isinstance(actual_joint_names, (str, bytes)) or not isinstance(
            actual_joint_names, Sequence
        ):
            raise TypeError("actual_joint_names must be a sequence of strings")
        if any(not isinstance(name, str) for name in actual_joint_names):
            raise TypeError("actual_joint_names must be a sequence of strings")

        name_to_index: dict[str, int] = {}
        for index, name in enumerate(actual_joint_names):
            if name in name_to_index:
                raise ValueError(f"actual_joint_names contains duplicate name: {name}")
            name_to_index[name] = index

        required = (
            M1_LEG_JOINT_NAMES
            + M1_WHEEL_JOINT_NAMES
            + PANDA_ARM_JOINT_NAMES
            + PANDA_FINGER_JOINT_NAMES
        )
        missing = tuple(name for name in required if name not in name_to_index)
        if missing:
            raise ValueError(f"missing required joints: {', '.join(missing)}")

        legs = _indices_for(M1_LEG_JOINT_NAMES, name_to_index)
        wheels = _indices_for(M1_WHEEL_JOINT_NAMES, name_to_index)
        panda_arm = _indices_for(PANDA_ARM_JOINT_NAMES, name_to_index)
        fingers = _indices_for(PANDA_FINGER_JOINT_NAMES, name_to_index)
        controlled = torch.cat((legs, wheels, panda_arm))
        return cls(
            legs=legs,
            wheels=wheels,
            panda_arm=panda_arm,
            fingers=fingers,
            controlled=controlled,
        )


def require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Validate a finite tensor without copying or changing its device."""

    if not isinstance(trailing_shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in trailing_shape
    ):
        raise TypeError("trailing_shape must be a tuple of non-negative integers")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if len(trailing_shape) > value.ndim or (
        trailing_shape and tuple(value.shape[-len(trailing_shape) :]) != trailing_shape
    ):
        raise ValueError(
            f"{name} must end with shape {trailing_shape}; got {tuple(value.shape)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}; got {value.dtype}")
    if device is not None:
        expected_device = torch.device(device)
        if value.device != expected_device:
            raise ValueError(
                f"{name} must be on device {expected_device}; got {value.device}"
            )
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value
