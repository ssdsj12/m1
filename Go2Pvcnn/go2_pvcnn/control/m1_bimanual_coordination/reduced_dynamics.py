"""Condense constrained 59-DOF dynamics into an affine 43-effort map."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import (
    ACTIVE_CONTROL_DOF,
    GENERALIZED_DOF,
    WHEEL_CONSTRAINT_DOF,
    FullDynamicsState,
    _float64,
)


ACTIVE_DOF = ACTIVE_CONTROL_DOF


def build_actuation_matrix(active_joint_ids: tuple[int, ...]) -> torch.Tensor:
    """Map 43 active joint efforts into floating-base generalized forces."""

    if len(active_joint_ids) != ACTIVE_DOF:
        raise ValueError(f"active_joint_ids must contain {ACTIVE_DOF} entries")
    if len(set(active_joint_ids)) != ACTIVE_DOF:
        raise ValueError("active_joint_ids must be unique")
    if any(joint_id < 0 or joint_id >= GENERALIZED_DOF - 6 for joint_id in active_joint_ids):
        raise ValueError("active_joint_ids contain an out-of-range physical joint")
    result = torch.zeros((GENERALIZED_DOF, ACTIVE_DOF), dtype=torch.float64)
    for active_column, joint_id in enumerate(active_joint_ids):
        result[joint_id + 6, active_column] = 1.0
    return result


def stack_stationary_wheel_jacobians(
    spatial_jacobians: torch.Tensor,
) -> torch.Tensor:
    """Stack four wheel-center linear Jacobians into twelve contact rows."""

    if not isinstance(spatial_jacobians, torch.Tensor):
        raise TypeError("spatial_jacobians must be a torch.Tensor")
    if spatial_jacobians.dtype != torch.float64 or spatial_jacobians.device.type != "cpu":
        raise TypeError("spatial_jacobians must be a CPU float64 tensor")
    if tuple(spatial_jacobians.shape) != (4, 6, GENERALIZED_DOF):
        raise ValueError("spatial_jacobians must have exact shape (4, 6, 59)")
    if not torch.isfinite(spatial_jacobians).all().item():
        raise ValueError("spatial_jacobians must contain only finite values")
    return spatial_jacobians[:, :3].reshape(WHEEL_CONSTRAINT_DOF, GENERALIZED_DOF).clone()


@dataclass(frozen=True)
class ReducedDynamicsMap:
    qdd_offset: torch.Tensor
    qdd_from_effort: torch.Tensor
    contact_offset: torch.Tensor
    contact_from_effort: torch.Tensor
    condition_number: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("qdd_offset", (GENERALIZED_DOF,)),
            ("qdd_from_effort", (GENERALIZED_DOF, ACTIVE_DOF)),
            ("contact_offset", (WHEEL_CONSTRAINT_DOF,)),
            (
                "contact_from_effort",
                (WHEEL_CONSTRAINT_DOF, ACTIVE_DOF),
            ),
        ):
            object.__setattr__(
                self, name, _float64(name, getattr(self, name), shape)
            )
        if not math.isfinite(self.condition_number) or self.condition_number <= 0.0:
            raise ValueError("condition_number must be finite and positive")


def condense_constrained_dynamics(
    state: FullDynamicsState,
) -> ReducedDynamicsMap:
    """Solve one KKT factorization for affine acceleration/contact-force maps."""

    if not isinstance(state, FullDynamicsState):
        raise TypeError("state must be FullDynamicsState")
    zero_contact = torch.zeros(
        (WHEEL_CONSTRAINT_DOF, WHEEL_CONSTRAINT_DOF), dtype=torch.float64
    )
    kkt = torch.cat(
        (
            torch.cat(
                (state.mass_matrix, -state.wheel_contact_jacobian.T), dim=1
            ),
            torch.cat((state.wheel_contact_jacobian, zero_contact), dim=1),
        ),
        dim=0,
    )
    affine_rhs = torch.cat((-state.bias, -state.wheel_contact_bias))
    effort_rhs = torch.cat(
        (
            state.actuation_matrix,
            torch.zeros(
                (WHEEL_CONSTRAINT_DOF, ACTIVE_DOF), dtype=torch.float64
            ),
        ),
        dim=0,
    )
    condition_number = float(torch.linalg.cond(kkt).item())
    if not math.isfinite(condition_number):
        raise ValueError("constrained dynamics KKT matrix is singular")
    solution = torch.linalg.solve(
        kkt, torch.cat((affine_rhs.unsqueeze(1), effort_rhs), dim=1)
    )
    if not torch.isfinite(solution).all().item():
        raise ValueError("constrained dynamics solution is non-finite")
    return ReducedDynamicsMap(
        qdd_offset=solution[:GENERALIZED_DOF, 0],
        qdd_from_effort=solution[:GENERALIZED_DOF, 1:],
        contact_offset=solution[GENERALIZED_DOF:, 0],
        contact_from_effort=solution[GENERALIZED_DOF:, 1:],
        condition_number=condition_number,
    )


__all__ = [
    "ACTIVE_DOF",
    "GENERALIZED_DOF",
    "WHEEL_CONSTRAINT_DOF",
    "ReducedDynamicsMap",
    "build_actuation_matrix",
    "condense_constrained_dynamics",
    "stack_stationary_wheel_jacobians",
]
