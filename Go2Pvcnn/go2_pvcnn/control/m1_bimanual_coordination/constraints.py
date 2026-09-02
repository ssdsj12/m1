"""Explicit 43-channel effort, collision, and grasp constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import ACTIVE_CONTROL_DOF, BimanualSnapshot
from .dual_arm_mpc import DualArmMpcSolution
from .hand_mpc import HandMpcSolution
from .object_mpc import ObjectMpcSolution


@dataclass(frozen=True)
class BimanualConstraintCfg:
    minimum_collision_distance_m: float = 0.02
    control_dt: float = 0.005

    def __post_init__(self) -> None:
        for name in ("minimum_collision_distance_m", "control_dt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class BimanualWbcRequest:
    snapshot: BimanualSnapshot
    object_solution: ObjectMpcSolution
    arm_solution: DualArmMpcSolution
    left_hand_solution: HandMpcSolution
    right_hand_solution: HandMpcSolution
    collision_distances: torch.Tensor
    collision_jacobian: torch.Tensor

    def __post_init__(self) -> None:
        for name, expected in (
            ("snapshot", BimanualSnapshot),
            ("object_solution", ObjectMpcSolution),
            ("arm_solution", DualArmMpcSolution),
            ("left_hand_solution", HandMpcSolution),
            ("right_hand_solution", HandMpcSolution),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if not isinstance(self.collision_distances, torch.Tensor):
            raise TypeError("collision_distances must be a torch.Tensor")
        if not isinstance(self.collision_jacobian, torch.Tensor):
            raise TypeError("collision_jacobian must be a torch.Tensor")
        for name, value in (
            ("collision_distances", self.collision_distances),
            ("collision_jacobian", self.collision_jacobian),
        ):
            if value.dtype != torch.float64:
                raise TypeError(f"{name} must have dtype torch.float64")
            if value.device.type != "cpu":
                raise ValueError(f"{name} must be on device cpu")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if self.collision_distances.ndim != 1 or self.collision_distances.numel() == 0:
            raise ValueError("collision_distances must have shape (pairs,) with pairs > 0")
        expected_shape = (self.collision_distances.numel(), ACTIVE_CONTROL_DOF)
        if tuple(self.collision_jacobian.shape) != expected_shape:
            raise ValueError(
                f"collision_jacobian must have shape {expected_shape}; got {tuple(self.collision_jacobian.shape)}"
            )
        object.__setattr__(
            self, "collision_distances", self.collision_distances.clone()
        )
        object.__setattr__(
            self, "collision_jacobian", self.collision_jacobian.clone()
        )


@dataclass(frozen=True)
class BimanualConstraintSet:
    lower_effort: torch.Tensor
    upper_effort: torch.Tensor
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    min_collision_distance: float
    force_closure_margin: float

    def __post_init__(self) -> None:
        if self.lower_effort.shape != (ACTIVE_CONTROL_DOF,) or self.upper_effort.shape != (
            ACTIVE_CONTROL_DOF,
        ):
            raise ValueError("effort limits must have shape (43,)")
        if self.inequality_matrix.ndim != 2 or self.inequality_matrix.shape[1] != ACTIVE_CONTROL_DOF:
            raise ValueError("inequality_matrix must have shape (rows, 43)")
        if self.inequality_upper.shape != (self.inequality_matrix.shape[0],):
            raise ValueError("inequality_upper must match inequality rows")


def effort_limits() -> torch.Tensor:
    """Return positive limits in the frozen 16+1+14+12 channel order."""

    panda = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    return torch.tensor(
        (
            *(150.0 for _ in range(12)),
            *(200.0 for _ in range(4)),
            50.0,
            *panda,
            *panda,
            *(10.0 for _ in range(12)),
        ),
        dtype=torch.float64,
    )


def subsolution_failure_reason(request: BimanualWbcRequest) -> str | None:
    if not request.object_solution.diagnostics.feasible:
        return "object_mpc_infeasible"
    if not request.arm_solution.both_feasible:
        return "dual_arm_mpc_infeasible"
    if not request.arm_solution.left.diagnostics.feasible:
        return "left_arm_mpc_infeasible"
    if not request.arm_solution.right.diagnostics.feasible:
        return "right_arm_mpc_infeasible"
    if not request.left_hand_solution.diagnostics.feasible:
        return "left_hand_mpc_infeasible"
    if not request.right_hand_solution.diagnostics.feasible:
        return "right_hand_mpc_infeasible"
    if request.object_solution.diagnostics.force_closure_margin <= 0.0:
        return "force_closure_lost"
    return None


def build_bimanual_constraints(
    request: BimanualWbcRequest,
    cfg: BimanualConstraintCfg | None = None,
) -> BimanualConstraintSet:
    """Linearize pair distances against one 200 Hz effort command."""

    if not isinstance(request, BimanualWbcRequest):
        raise TypeError("request must be BimanualWbcRequest")
    cfg = BimanualConstraintCfg() if cfg is None else cfg
    if not isinstance(cfg, BimanualConstraintCfg):
        raise TypeError("cfg must be BimanualConstraintCfg")
    limits = effort_limits()
    # d_next ~= d + dt^2 J tau; enforce d_next >= d_min.
    collision_matrix = -(cfg.control_dt**2) * request.collision_jacobian
    collision_upper = (
        request.collision_distances - cfg.minimum_collision_distance_m
    )
    force_closure_margin = min(
        request.object_solution.diagnostics.force_closure_margin,
        request.left_hand_solution.diagnostics.slip_margin,
        request.right_hand_solution.diagnostics.slip_margin,
    )
    closure_row = torch.zeros((1, ACTIVE_CONTROL_DOF), dtype=torch.float64)
    inequality_matrix = torch.cat((collision_matrix, closure_row), dim=0)
    inequality_upper = torch.cat(
        (collision_upper, torch.tensor([force_closure_margin], dtype=torch.float64))
    )
    return BimanualConstraintSet(
        lower_effort=-limits,
        upper_effort=limits,
        inequality_matrix=inequality_matrix,
        inequality_upper=inequality_upper,
        min_collision_distance=float(request.collision_distances.min().item()),
        force_closure_margin=float(force_closure_margin),
    )
