"""Standing whole-body QP formulation for the C0 M1 + Panda Teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import CONTROLLED_DOF, GENERALIZED_DOF, require_tensor
from .qp_backend import DenseQpProblem, DenseQpResult, solve_reference_qp


CONTACT_COUNT = 4
CONTACT_FORCE_DOF = 3
CONTACT_DOF = CONTACT_COUNT * CONTACT_FORCE_DOF
DECISION_DOF = GENERALIZED_DOF + CONTACT_DOF


@dataclass(frozen=True)
class StandingWbcCfg:
    balance_weight: float = 1.0e6
    base_pose_weight: float = 1.0e5
    leg_posture_weight: float = 1.0e4
    arm_tracking_weight: float = 1.0e3
    wheel_stop_weight: float = 1.0e3
    force_equalization_weight: float = 10.0
    regularization: float = 1.0e-6


@dataclass(frozen=True)
class StandingWbcInput:
    mass_matrix: torch.Tensor
    bias_force: torch.Tensor
    contact_jacobian: torch.Tensor
    contact_jacobian_dot_qd: torch.Tensor
    mount_wrench_jacobian: torch.Tensor
    external_wrench: torch.Tensor
    balance_jacobian: torch.Tensor
    balance_acceleration: torch.Tensor
    base_jacobian: torch.Tensor
    base_acceleration: torch.Tensor
    leg_generalized_indices: torch.Tensor
    wheel_generalized_indices: torch.Tensor
    arm_generalized_indices: torch.Tensor
    leg_acceleration: torch.Tensor
    wheel_acceleration: torch.Tensor
    arm_acceleration: torch.Tensor
    qdd_lower: torch.Tensor
    qdd_upper: torch.Tensor
    effort_limit: torch.Tensor
    friction_coefficient: float


@dataclass(frozen=True)
class StandingWbcProblem:
    qp: DenseQpProblem
    external_generalized_force: torch.Tensor
    torque_matrix: torch.Tensor
    torque_offset: torch.Tensor
    task_matrices: dict[str, torch.Tensor]
    task_targets: dict[str, torch.Tensor]


@dataclass(frozen=True)
class StandingWbcResult:
    qdd: torch.Tensor
    contact_force: torch.Tensor
    effort: torch.Tensor | None
    qp_result: DenseQpResult
    task_residuals: dict[str, float]


def _validate_index_group(
    name: str,
    value: torch.Tensor,
    expected_size: int,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.int64")
    if value.device != device:
        raise ValueError(f"{name} device must match mass_matrix")
    if ((value < 6) | (value >= GENERALIZED_DOF)).any().item():
        raise ValueError(f"{name} must contain actuated generalized indices")


def _validate_input(state: StandingWbcInput) -> torch.Tensor:
    if not isinstance(state, StandingWbcInput):
        raise TypeError("state must be a StandingWbcInput")
    if isinstance(state.mass_matrix, torch.Tensor) and state.mass_matrix.ndim == 3:
        raise ValueError("C0 standing WBC supports one environment")
    require_tensor(
        "mass_matrix",
        state.mass_matrix,
        trailing_shape=(GENERALIZED_DOF, GENERALIZED_DOF),
    )
    if not state.mass_matrix.is_floating_point():
        raise TypeError("mass_matrix must have a floating dtype")
    dtype = state.mass_matrix.dtype
    device = state.mass_matrix.device

    tensors = (
        ("bias_force", state.bias_force, (GENERALIZED_DOF,)),
        ("contact_jacobian", state.contact_jacobian, (CONTACT_DOF, GENERALIZED_DOF)),
        ("contact_jacobian_dot_qd", state.contact_jacobian_dot_qd, (CONTACT_DOF,)),
        ("mount_wrench_jacobian", state.mount_wrench_jacobian, (6, GENERALIZED_DOF)),
        ("external_wrench", state.external_wrench, (6,)),
        ("balance_jacobian", state.balance_jacobian, (3, GENERALIZED_DOF)),
        ("balance_acceleration", state.balance_acceleration, (3,)),
        ("base_jacobian", state.base_jacobian, (6, GENERALIZED_DOF)),
        ("base_acceleration", state.base_acceleration, (6,)),
        ("leg_acceleration", state.leg_acceleration, (12,)),
        ("wheel_acceleration", state.wheel_acceleration, (4,)),
        ("arm_acceleration", state.arm_acceleration, (7,)),
        ("qdd_lower", state.qdd_lower, (GENERALIZED_DOF,)),
        ("qdd_upper", state.qdd_upper, (GENERALIZED_DOF,)),
        ("effort_limit", state.effort_limit, (CONTROLLED_DOF,)),
    )
    for name, value, trailing_shape in tensors:
        require_tensor(name, value, trailing_shape=trailing_shape)
        if value.ndim != len(trailing_shape):
            raise ValueError("C0 standing WBC supports one environment")
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype must match mass_matrix")
        if value.device != device:
            raise ValueError(f"{name} device must match mass_matrix")

    _validate_index_group(
        "leg_generalized_indices", state.leg_generalized_indices, 12, device
    )
    _validate_index_group(
        "wheel_generalized_indices", state.wheel_generalized_indices, 4, device
    )
    _validate_index_group(
        "arm_generalized_indices", state.arm_generalized_indices, 7, device
    )
    controlled = torch.cat(
        (
            state.leg_generalized_indices,
            state.wheel_generalized_indices,
            state.arm_generalized_indices,
        )
    )
    if torch.unique(controlled).numel() != CONTROLLED_DOF:
        raise ValueError("controlled generalized indices must be unique")
    if torch.any(state.effort_limit <= 0.0).item():
        raise ValueError("effort_limit must be positive")
    coefficient = state.friction_coefficient
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
        raise TypeError("friction_coefficient must be a real number")
    if not math.isfinite(float(coefficient)) or float(coefficient) <= 0.0:
        raise ValueError("friction_coefficient must be finite and positive")
    return controlled


def _decision_task(jacobian: torch.Tensor) -> torch.Tensor:
    result = jacobian.new_zeros((jacobian.shape[0], DECISION_DOF))
    result[:, :GENERALIZED_DOF] = jacobian
    return result


def _selector(indices: torch.Tensor, dtype: torch.dtype, device: torch.device):
    result = torch.zeros(
        (indices.numel(), GENERALIZED_DOF), dtype=dtype, device=device
    )
    result[torch.arange(indices.numel(), device=device), indices] = 1.0
    return result


def _add_objective(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    matrix: torch.Tensor,
    target: torch.Tensor,
    weight: float,
) -> None:
    hessian.add_(2.0 * weight * matrix.transpose(0, 1) @ matrix)
    gradient.add_(-2.0 * weight * matrix.transpose(0, 1) @ target)


def _friction_inequalities(
    dtype: torch.dtype,
    device: torch.device,
    friction_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.zeros((CONTACT_COUNT * 5, DECISION_DOF), dtype=dtype, device=device)
    for contact in range(CONTACT_COUNT):
        row = 5 * contact
        force = GENERALIZED_DOF + CONTACT_FORCE_DOF * contact
        matrix[row, force + 2] = -1.0
        matrix[row + 1, force] = 1.0
        matrix[row + 1, force + 2] = -friction_coefficient
        matrix[row + 2, force] = -1.0
        matrix[row + 2, force + 2] = -friction_coefficient
        matrix[row + 3, force + 1] = 1.0
        matrix[row + 3, force + 2] = -friction_coefficient
        matrix[row + 4, force + 1] = -1.0
        matrix[row + 4, force + 2] = -friction_coefficient
    return matrix, torch.zeros(matrix.shape[0], dtype=dtype, device=device)


def build_standing_wbc_problem(
    state: StandingWbcInput,
    cfg: StandingWbcCfg | None = None,
) -> StandingWbcProblem:
    """Build the 31-acceleration plus 12-contact-force standing QP."""

    cfg = cfg or StandingWbcCfg()
    controlled = _validate_input(state)
    dtype = state.mass_matrix.dtype
    device = state.mass_matrix.device
    external = state.mount_wrench_jacobian.transpose(0, 1) @ state.external_wrench
    contact_transpose = state.contact_jacobian.transpose(0, 1)

    dynamics = torch.cat(
        (state.mass_matrix[:6], -contact_transpose[:6]), dim=1
    )
    dynamics_rhs = -state.bias_force[:6] + external[:6]
    contact = torch.zeros((CONTACT_DOF, DECISION_DOF), dtype=dtype, device=device)
    contact[:, :GENERALIZED_DOF] = state.contact_jacobian
    equality_matrix = torch.cat((dynamics, contact), dim=0)
    equality_rhs = torch.cat(
        (dynamics_rhs, -state.contact_jacobian_dot_qd), dim=0
    )

    torque_matrix = torch.cat(
        (
            state.mass_matrix.index_select(0, controlled),
            -contact_transpose.index_select(0, controlled),
        ),
        dim=1,
    )
    torque_offset = (state.bias_force - external).index_select(0, controlled)

    friction_matrix, friction_upper = _friction_inequalities(
        dtype, device, float(state.friction_coefficient)
    )
    torque_inequality = torch.cat((torque_matrix, -torque_matrix), dim=0)
    torque_upper = torch.cat(
        (
            state.effort_limit - torque_offset,
            state.effort_limit + torque_offset,
        ),
        dim=0,
    )
    inequality_matrix = torch.cat((friction_matrix, torque_inequality), dim=0)
    inequality_upper = torch.cat((friction_upper, torque_upper), dim=0)

    leg_matrix = _decision_task(
        _selector(state.leg_generalized_indices, dtype, device)
    )
    wheel_matrix = _decision_task(
        _selector(state.wheel_generalized_indices, dtype, device)
    )
    arm_matrix = _decision_task(
        _selector(state.arm_generalized_indices, dtype, device)
    )
    task_matrices = {
        "balance": _decision_task(state.balance_jacobian),
        "base": _decision_task(state.base_jacobian),
        "legs": leg_matrix,
        "wheels": wheel_matrix,
        "arm": arm_matrix,
    }
    task_targets = {
        "balance": state.balance_acceleration,
        "base": state.base_acceleration,
        "legs": state.leg_acceleration,
        "wheels": state.wheel_acceleration,
        "arm": state.arm_acceleration,
    }
    weights = {
        "balance": cfg.balance_weight,
        "base": cfg.base_pose_weight,
        "legs": cfg.leg_posture_weight,
        "wheels": cfg.wheel_stop_weight,
        "arm": cfg.arm_tracking_weight,
    }

    hessian = 2.0 * cfg.regularization * torch.eye(
        DECISION_DOF, dtype=dtype, device=device
    )
    gradient = torch.zeros(DECISION_DOF, dtype=dtype, device=device)
    for name, matrix in task_matrices.items():
        _add_objective(
            hessian, gradient, matrix, task_targets[name], weights[name]
        )

    force_equalization = torch.zeros(
        (CONTACT_COUNT, DECISION_DOF), dtype=dtype, device=device
    )
    normal_indices = torch.tensor(
        [GENERALIZED_DOF + 3 * contact + 2 for contact in range(CONTACT_COUNT)],
        dtype=torch.long,
        device=device,
    )
    force_equalization[
        torch.arange(CONTACT_COUNT, device=device), normal_indices
    ] = 1.0
    force_equalization[:, normal_indices] -= 1.0 / CONTACT_COUNT
    _add_objective(
        hessian,
        gradient,
        force_equalization,
        torch.zeros(CONTACT_COUNT, dtype=dtype, device=device),
        cfg.force_equalization_weight,
    )

    lower_bound = torch.cat(
        (
            state.qdd_lower,
            torch.full((CONTACT_DOF,), -torch.inf, dtype=dtype, device=device),
        )
    )
    upper_bound = torch.cat(
        (
            state.qdd_upper,
            torch.full((CONTACT_DOF,), torch.inf, dtype=dtype, device=device),
        )
    )
    qp = DenseQpProblem(
        hessian=hessian,
        gradient=gradient,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_upper=inequality_upper,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return StandingWbcProblem(
        qp=qp,
        external_generalized_force=external,
        torque_matrix=torque_matrix,
        torque_offset=torque_offset,
        task_matrices=task_matrices,
        task_targets=task_targets,
    )


def solve_standing_wbc(
    state: StandingWbcInput,
    cfg: StandingWbcCfg | None = None,
) -> StandingWbcResult:
    """Build and solve one C0 standing WBC problem."""

    assembled = build_standing_wbc_problem(state, cfg)
    qp_result = solve_reference_qp(assembled.qp)
    solution = qp_result.solution
    qdd = solution[:GENERALIZED_DOF]
    contact_force = solution[GENERALIZED_DOF:].reshape(CONTACT_COUNT, CONTACT_FORCE_DOF)

    torque_matrix = assembled.torque_matrix.detach().to(device="cpu", dtype=torch.float64)
    torque_offset = assembled.torque_offset.detach().to(device="cpu", dtype=torch.float64)
    effort = torque_matrix @ solution + torque_offset if qp_result.success else None

    task_residuals = {}
    for name, matrix in assembled.task_matrices.items():
        matrix_cpu = matrix.detach().to(device="cpu", dtype=torch.float64)
        target_cpu = assembled.task_targets[name].detach().to(
            device="cpu", dtype=torch.float64
        )
        task_residuals[name] = float(
            torch.linalg.vector_norm(matrix_cpu @ solution - target_cpu).item()
        )
    contact_matrix = assembled.qp.equality_matrix[6:].detach().to(
        device="cpu", dtype=torch.float64
    )
    contact_rhs = assembled.qp.equality_rhs[6:].detach().to(
        device="cpu", dtype=torch.float64
    )
    task_residuals["contact"] = float(
        torch.linalg.vector_norm(contact_matrix @ solution - contact_rhs).item()
    )
    return StandingWbcResult(
        qdd=qdd,
        contact_force=contact_force,
        effort=effort,
        qp_result=qp_result,
        task_residuals=task_residuals,
    )
