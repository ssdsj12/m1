"""Immutable CPU-float64 contracts shared by the bimanual control stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from go2_pvcnn.control.m1_panda_coordination.contracts import require_tensor


ACTIVE_CONTROL_DOF = 43
GENERALIZED_DOF = 59
WHEEL_CONSTRAINT_DOF = 12


def _timestamp(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("timestamp_ns must be a positive integer")
    return value


def _float64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    require_tensor(name, value, trailing_shape=shape, dtype=torch.float64, device="cpu")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have exact shape {shape}; got {tuple(value.shape)}")
    return value.clone()


def _bool(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on device cpu; got {value.device}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {tuple(value.shape)}")
    return value.clone()


@dataclass(frozen=True)
class SideArmState:
    q: torch.Tensor
    qd: torch.Tensor
    palm_pose_b: torch.Tensor
    palm_twist_b: torch.Tensor
    jacobian_b: torch.Tensor
    mass_matrix: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        for name, shape in (
            ("q", (7,)),
            ("qd", (7,)),
            ("palm_pose_b", (6,)),
            ("palm_twist_b", (6,)),
            ("jacobian_b", (6, 7)),
            ("mass_matrix", (7, 7)),
            ("bias", (7,)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))


@dataclass(frozen=True)
class SideHandState:
    q: torch.Tensor
    qd: torch.Tensor
    fingertip_forces_b: torch.Tensor
    fingertip_positions_b: torch.Tensor
    contact_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name, shape in (
            ("q", (6,)),
            ("qd", (6,)),
            ("fingertip_forces_b", (5, 3)),
            ("fingertip_positions_b", (5, 3)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))
        object.__setattr__(
            self, "contact_mask", _bool("contact_mask", self.contact_mask, (5,))
        )


@dataclass(frozen=True)
class BoxState:
    pose_b: torch.Tensor
    twist_b: torch.Tensor
    mass: torch.Tensor
    inertia_b: torch.Tensor
    supported: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_b", _float64("pose_b", self.pose_b, (6,)))
        object.__setattr__(self, "twist_b", _float64("twist_b", self.twist_b, (6,)))
        object.__setattr__(self, "mass", _float64("mass", self.mass, ()))
        object.__setattr__(self, "inertia_b", _float64("inertia_b", self.inertia_b, (3, 3)))
        if self.mass.item() <= 0.0:
            raise ValueError("mass must be strictly positive")
        if not torch.allclose(self.inertia_b, self.inertia_b.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("inertia_b must be symmetric")
        if not torch.all(torch.linalg.eigvalsh(self.inertia_b) > 0.0).item():
            raise ValueError("inertia_b must be positive definite")
        if not isinstance(self.supported, bool):
            raise TypeError("supported must be bool")


@dataclass(frozen=True)
class BimanualSnapshot:
    timestamp_ns: int
    base_state: torch.Tensor
    m1_q: torch.Tensor
    m1_qd: torch.Tensor
    platform_q_qd: torch.Tensor
    left_arm: SideArmState
    right_arm: SideArmState
    left_hand: SideHandState
    right_hand: SideHandState
    box: BoxState

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns)
        for name, shape in (
            ("base_state", (13,)),
            ("m1_q", (16,)),
            ("m1_qd", (16,)),
            ("platform_q_qd", (2,)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))
        for name, expected_type in (
            ("left_arm", SideArmState),
            ("right_arm", SideArmState),
            ("left_hand", SideHandState),
            ("right_hand", SideHandState),
            ("box", BoxState),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")


@dataclass(frozen=True)
class FullDynamicsState:
    """Full floating-base dynamics with a 43-column active selection map."""

    mass_matrix: torch.Tensor
    bias: torch.Tensor
    actuation_matrix: torch.Tensor
    wheel_contact_jacobian: torch.Tensor
    wheel_contact_bias: torch.Tensor

    def __post_init__(self) -> None:
        for name, shape in (
            ("mass_matrix", (GENERALIZED_DOF, GENERALIZED_DOF)),
            ("bias", (GENERALIZED_DOF,)),
            ("actuation_matrix", (GENERALIZED_DOF, ACTIVE_CONTROL_DOF)),
            (
                "wheel_contact_jacobian",
                (WHEEL_CONSTRAINT_DOF, GENERALIZED_DOF),
            ),
            ("wheel_contact_bias", (WHEEL_CONSTRAINT_DOF,)),
        ):
            object.__setattr__(
                self, name, _float64(name, getattr(self, name), shape)
            )
        if not torch.allclose(
            self.mass_matrix,
            self.mass_matrix.T,
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise ValueError("mass_matrix must be symmetric")
        try:
            torch.linalg.cholesky(self.mass_matrix)
        except torch.linalg.LinAlgError as error:
            raise ValueError("mass_matrix must be positive definite") from error
        if torch.linalg.matrix_rank(self.actuation_matrix).item() != ACTIVE_CONTROL_DOF:
            raise ValueError("actuation_matrix must have full active-column rank")


@dataclass(frozen=True)
class BimanualCommand:
    timestamp_ns: int
    effort: torch.Tensor
    feasible: bool
    fallback_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns)
        object.__setattr__(
            self, "effort", _float64("effort", self.effort, (ACTIVE_CONTROL_DOF,))
        )
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be bool")
        if not isinstance(self.fallback_reasons, tuple) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in self.fallback_reasons
        ):
            raise ValueError("fallback_reasons must be a tuple of non-empty strings")


class BimanualPhase(Enum):
    APPROACH = auto()
    PRELOAD = auto()
    GRASP = auto()
    LIFT = auto()
    HOLD = auto()
    LOWER = auto()
    RELEASE = auto()
    DONE = auto()
    HOLD_SAFE = auto()
    LOWER_SAFE = auto()
    SAFE_RELEASE = auto()
    TERMINATED = auto()


def validate_monotonic_snapshot(
    previous: BimanualSnapshot, current: BimanualSnapshot
) -> None:
    """Require strictly increasing timestamps for atomic control snapshots."""

    if not isinstance(previous, BimanualSnapshot) or not isinstance(
        current, BimanualSnapshot
    ):
        raise TypeError("previous and current must be BimanualSnapshot instances")
    if current.timestamp_ns <= previous.timestamp_ns:
        raise ValueError("snapshot timestamp_ns must be strictly monotonic")
