"""Atomic 43-channel whole-body effort QP for bimanual manipulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    solve_reference_qp,
)

from .constraints import (
    BimanualConstraintCfg,
    BimanualConstraintSet,
    BimanualWbcRequest,
    build_bimanual_constraints,
    subsolution_failure_reason,
)
from .contracts import ACTIVE_CONTROL_DOF


@dataclass(frozen=True)
class BimanualWbcCfg:
    regularization: float = 1.0e-8
    platform_kp: float = 40.0
    platform_kd: float = 8.0
    arm_kp: float = 20.0
    arm_kd: float = 4.0
    hand_kp: float = 20.0
    hand_kd: float = 1.0
    qp_tolerance: float = 1.0e-7
    qp_max_iterations: int = 256

    def __post_init__(self) -> None:
        for name in (
            "regularization",
            "platform_kp",
            "platform_kd",
            "arm_kp",
            "arm_kd",
            "hand_kp",
            "hand_kd",
            "qp_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.qp_max_iterations, bool) or not isinstance(
            self.qp_max_iterations, int
        ) or self.qp_max_iterations <= 0:
            raise ValueError("qp_max_iterations must be a positive integer")


@dataclass(frozen=True)
class BimanualWbcDiagnostics:
    qp_iterations: int
    min_collision_distance: float
    force_closure_margin: float
    support_margin: float
    fallback_reason: str | None


@dataclass(frozen=True)
class BimanualWbcSolution:
    effort: torch.Tensor
    feasible: bool
    fallback_used: bool
    diagnostics: BimanualWbcDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.effort, torch.Tensor):
            raise TypeError("effort must be a torch.Tensor")
        if self.effort.dtype != torch.float64:
            raise TypeError("effort must have dtype torch.float64")
        if self.effort.device.type != "cpu":
            raise ValueError("effort must be on device cpu")
        if self.effort.shape != (ACTIVE_CONTROL_DOF,):
            raise ValueError("effort must have shape (43,)")
        if not torch.isfinite(self.effort).all().item():
            raise ValueError("effort must contain only finite values")
        object.__setattr__(self, "effort", self.effort.clone())
        if not isinstance(self.feasible, bool) or not isinstance(self.fallback_used, bool):
            raise TypeError("feasible and fallback_used must be bool")
        if not isinstance(self.diagnostics, BimanualWbcDiagnostics):
            raise TypeError("diagnostics must be BimanualWbcDiagnostics")


def _nominal_effort(request: BimanualWbcRequest, cfg: BimanualWbcCfg) -> torch.Tensor:
    effort = torch.zeros(ACTIVE_CONTROL_DOF, dtype=torch.float64)
    # M1 stays at its accepted implicit-actuator hold; wheel channels 12:16 are
    # explicitly zero for the fixed-condition first task.
    platform = request.snapshot.platform_q_qd
    effort[16] = (
        cfg.platform_kp * (request.object_solution.platform_yaw[0] - platform[0])
        - cfg.platform_kd * platform[1]
    )
    for offset, state, solution in (
        (17, request.snapshot.left_arm, request.arm_solution.left),
        (24, request.snapshot.right_arm, request.arm_solution.right),
    ):
        effort[offset : offset + 7] = (
            state.mass_matrix @ solution.qdd[0]
            + state.bias
            + cfg.arm_kp * (solution.q_ref - state.q)
            + cfg.arm_kd * (solution.qd_ref - state.qd)
        )
    for offset, state, solution in (
        (31, request.snapshot.left_hand, request.left_hand_solution),
        (37, request.snapshot.right_hand, request.right_hand_solution),
    ):
        effort[offset : offset + 6] = (
            cfg.hand_kp * (solution.q_ref - state.q)
            + cfg.hand_kd * (solution.qd_ref - state.qd)
        )
    effort[12:16] = 0.0
    return effort


def build_wbc_problem(
    request: BimanualWbcRequest,
    constraints: BimanualConstraintSet,
    cfg: BimanualWbcCfg | None = None,
) -> DenseQpProblem:
    cfg = BimanualWbcCfg() if cfg is None else cfg
    nominal = _nominal_effort(request, cfg)
    identity = torch.eye(ACTIVE_CONTROL_DOF, dtype=torch.float64)
    return DenseQpProblem(
        hessian=2.0 * (1.0 + cfg.regularization) * identity,
        gradient=-2.0 * nominal,
        equality_matrix=torch.empty((0, ACTIVE_CONTROL_DOF), dtype=torch.float64),
        equality_rhs=torch.empty(0, dtype=torch.float64),
        inequality_matrix=constraints.inequality_matrix,
        inequality_upper=constraints.inequality_upper,
        lower_bound=constraints.lower_effort,
        upper_bound=constraints.upper_effort,
    )


def _support_margin(request: BimanualWbcRequest) -> float:
    planar_offset = float(
        torch.linalg.vector_norm(request.snapshot.base_state[:2]).item()
    )
    return 0.25 - planar_offset


class BimanualWholeBodyQp:
    """Return a complete effort vector only when every layer is accepted."""

    def __init__(
        self,
        cfg: BimanualWbcCfg | None = None,
        constraint_cfg: BimanualConstraintCfg | None = None,
    ) -> None:
        self.cfg = BimanualWbcCfg() if cfg is None else cfg
        self.constraint_cfg = (
            BimanualConstraintCfg() if constraint_cfg is None else constraint_cfg
        )
        if not isinstance(self.cfg, BimanualWbcCfg):
            raise TypeError("cfg must be BimanualWbcCfg")
        if not isinstance(self.constraint_cfg, BimanualConstraintCfg):
            raise TypeError("constraint_cfg must be BimanualConstraintCfg")
        self._last_safe: BimanualWbcSolution | None = None

    def _fallback(
        self,
        request: BimanualWbcRequest,
        reason: str,
        constraints: BimanualConstraintSet | None = None,
    ) -> BimanualWbcSolution:
        effort = (
            torch.zeros(ACTIVE_CONTROL_DOF, dtype=torch.float64)
            if self._last_safe is None
            else self._last_safe.effort
        )
        return BimanualWbcSolution(
            effort=effort,
            feasible=False,
            fallback_used=True,
            diagnostics=BimanualWbcDiagnostics(
                qp_iterations=0,
                min_collision_distance=(
                    float(request.collision_distances.min().item())
                    if constraints is None
                    else constraints.min_collision_distance
                ),
                force_closure_margin=(
                    request.object_solution.diagnostics.force_closure_margin
                    if constraints is None
                    else constraints.force_closure_margin
                ),
                support_margin=_support_margin(request),
                fallback_reason=reason,
            ),
        )

    def solve(self, request: BimanualWbcRequest) -> BimanualWbcSolution:
        if not isinstance(request, BimanualWbcRequest):
            raise TypeError("request must be BimanualWbcRequest")
        failure = subsolution_failure_reason(request)
        if failure is not None:
            return self._fallback(request, failure)
        constraints = build_bimanual_constraints(request, self.constraint_cfg)
        result = solve_reference_qp(
            build_wbc_problem(request, constraints, self.cfg),
            tolerance=self.cfg.qp_tolerance,
            max_iterations=self.cfg.qp_max_iterations,
        )
        if not result.success or not torch.isfinite(result.solution).all().item():
            return self._fallback(request, "qp_infeasible", constraints)
        solution = BimanualWbcSolution(
            effort=result.solution,
            feasible=True,
            fallback_used=False,
            diagnostics=BimanualWbcDiagnostics(
                qp_iterations=result.iterations,
                min_collision_distance=constraints.min_collision_distance,
                force_closure_margin=constraints.force_closure_margin,
                support_margin=_support_margin(request),
                fallback_reason=None,
            ),
        )
        self._last_safe = BimanualWbcSolution(
            effort=solution.effort,
            feasible=True,
            fallback_used=False,
            diagnostics=solution.diagnostics,
        )
        return solution


__all__ = [
    "BimanualConstraintSet",
    "BimanualWbcCfg",
    "BimanualWbcDiagnostics",
    "BimanualWbcRequest",
    "BimanualWbcSolution",
    "BimanualWholeBodyQp",
    "build_bimanual_constraints",
    "build_wbc_problem",
]
