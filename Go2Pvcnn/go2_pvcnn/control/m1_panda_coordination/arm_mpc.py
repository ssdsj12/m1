"""Pure float64 contracts and linearized dynamics for the Panda arm MPC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import require_tensor
from .qp_backend import DenseQpProblem, solve_reference_qp


ARM_DOF = 7
ARM_TASK_DOF = 6
ARM_MPC_HORIZON_STEPS = 20
ARM_MPC_DT = 0.02


def _require_exact_cpu64(
    name: str, value: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    require_tensor(
        name,
        value,
        trailing_shape=(),
        dtype=torch.float64,
        device="cpu",
    )
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    return value


def _require_positive_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    value = _require_exact_cpu64(name, value, (ARM_DOF,))
    if not torch.all(value > 0.0).item():
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True)
class ArmMpcCfg:
    """Frozen timing and objective defaults for the first arm-MPC version."""

    dt: float = ARM_MPC_DT
    horizon_steps: int = ARM_MPC_HORIZON_STEPS
    pose_weight: float = 200.0
    twist_weight: float = 5.0
    acceleration_weight: float = 0.05
    acceleration_slew_weight: float = 0.02
    rest_posture_weight: float = 0.1
    hessian_regularization: float = 1.0e-8
    qp_tolerance: float = 1.0e-7
    qp_max_iterations: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.dt, bool) or not isinstance(self.dt, (int, float)):
            raise TypeError("dt must be a real number")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        if isinstance(self.horizon_steps, bool) or not isinstance(
            self.horizon_steps, int
        ):
            raise TypeError("horizon_steps must be an integer")
        if self.horizon_steps != ARM_MPC_HORIZON_STEPS:
            raise ValueError(
                f"horizon_steps must equal the frozen value {ARM_MPC_HORIZON_STEPS}"
            )
        for name in (
            "pose_weight",
            "twist_weight",
            "acceleration_weight",
            "acceleration_slew_weight",
            "rest_posture_weight",
            "hessian_regularization",
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

    @property
    def horizon_seconds(self) -> float:
        return float(self.dt) * self.horizon_steps


@dataclass(frozen=True)
class ArmMpcInput:
    """One atomic, canonical base-frame snapshot consumed by arm MPC."""

    q: torch.Tensor
    qd: torch.Tensor
    ee_pose_b: torch.Tensor
    ee_twist_b: torch.Tensor
    target_pose_b: torch.Tensor
    target_twist_b: torch.Tensor
    jacobian_b: torch.Tensor
    arm_mass_matrix: torch.Tensor
    arm_bias: torch.Tensor
    base_arm_coupling: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    qd_max: torch.Tensor
    qdd_max: torch.Tensor
    effort_max: torch.Tensor

    def __post_init__(self) -> None:
        _require_exact_cpu64("q", self.q, (ARM_DOF,))
        _require_exact_cpu64("qd", self.qd, (ARM_DOF,))
        _require_exact_cpu64("ee_pose_b", self.ee_pose_b, (ARM_TASK_DOF,))
        _require_exact_cpu64("ee_twist_b", self.ee_twist_b, (ARM_TASK_DOF,))
        _require_exact_cpu64(
            "target_pose_b",
            self.target_pose_b,
            (ARM_MPC_HORIZON_STEPS, ARM_TASK_DOF),
        )
        _require_exact_cpu64(
            "target_twist_b",
            self.target_twist_b,
            (ARM_MPC_HORIZON_STEPS, ARM_TASK_DOF),
        )
        _require_exact_cpu64(
            "jacobian_b", self.jacobian_b, (ARM_TASK_DOF, ARM_DOF)
        )
        _require_exact_cpu64(
            "arm_mass_matrix", self.arm_mass_matrix, (ARM_DOF, ARM_DOF)
        )
        _require_exact_cpu64("arm_bias", self.arm_bias, (ARM_DOF,))
        _require_exact_cpu64(
            "base_arm_coupling",
            self.base_arm_coupling,
            (ARM_TASK_DOF, ARM_DOF),
        )
        _require_exact_cpu64("q_min", self.q_min, (ARM_DOF,))
        _require_exact_cpu64("q_max", self.q_max, (ARM_DOF,))
        _require_positive_vector("qd_max", self.qd_max)
        _require_positive_vector("qdd_max", self.qdd_max)
        _require_positive_vector("effort_max", self.effort_max)
        if not torch.all(self.q_min < self.q_max).item():
            raise ValueError("q_min must be strictly below q_max")


@dataclass(frozen=True)
class LinearizedArmRollout:
    q: torch.Tensor
    qd: torch.Tensor
    pose_delta_b: torch.Tensor
    ee_twist_b: torch.Tensor


@dataclass(frozen=True)
class CondensedArmDynamics:
    q_offset: torch.Tensor
    qd_offset: torch.Tensor
    q_from_qdd: torch.Tensor
    qd_from_qdd: torch.Tensor


@dataclass(frozen=True)
class ArmMpcDiagnostics:
    feasible: bool
    fallback_used: bool
    fallback_reason: str | None
    iterations: int
    saturation_fraction: float
    sigma_min: float
    min_joint_margin: float
    mean_joint_margin: float
    ee_position_error: float
    ee_orientation_error: float


@dataclass(frozen=True)
class ArmMpcSolution:
    q_ref: torch.Tensor
    qd_ref: torch.Tensor
    qdd: torch.Tensor
    predicted_q: torch.Tensor
    predicted_qd: torch.Tensor
    predicted_pose_b: torch.Tensor
    predicted_twist_b: torch.Tensor
    predicted_dynamic_mount_wrench_b: torch.Tensor
    diagnostics: ArmMpcDiagnostics


def _validate_rollout_inputs(
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    jacobian_b: torch.Tensor,
    dt: float,
) -> None:
    _require_exact_cpu64("q", q, (ARM_DOF,))
    _require_exact_cpu64("qd", qd, (ARM_DOF,))
    require_tensor(
        "qdd", qdd, trailing_shape=(), dtype=torch.float64, device="cpu"
    )
    if qdd.ndim != 2 or qdd.shape[0] <= 0 or qdd.shape[1] != ARM_DOF:
        raise ValueError(f"qdd must have shape (horizon, {ARM_DOF}); got {tuple(qdd.shape)}")
    _require_exact_cpu64("jacobian_b", jacobian_b, (ARM_TASK_DOF, ARM_DOF))
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError("dt must be a real number")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")


def rollout_linearized_arm(
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    jacobian_b: torch.Tensor,
    dt: float,
) -> LinearizedArmRollout:
    """Roll out a frozen-Jacobian double integrator for one MPC horizon."""

    _validate_rollout_inputs(q, qd, qdd, jacobian_b, dt)
    step_dt = float(dt)
    q_steps: list[torch.Tensor] = []
    qd_steps: list[torch.Tensor] = []
    current_q = q
    current_qd = qd
    for acceleration in qdd:
        current_q = current_q + step_dt * current_qd + 0.5 * step_dt**2 * acceleration
        current_qd = current_qd + step_dt * acceleration
        q_steps.append(current_q)
        qd_steps.append(current_qd)
    predicted_q = torch.stack(q_steps)
    predicted_qd = torch.stack(qd_steps)
    return LinearizedArmRollout(
        q=predicted_q,
        qd=predicted_qd,
        pose_delta_b=(predicted_q - q) @ jacobian_b.transpose(0, 1),
        ee_twist_b=predicted_qd @ jacobian_b.transpose(0, 1),
    )


def condense_arm_dynamics(
    q: torch.Tensor, qd: torch.Tensor, *, horizon_steps: int, dt: float
) -> CondensedArmDynamics:
    """Return affine maps from flattened joint accelerations to q and qd."""

    _require_exact_cpu64("q", q, (ARM_DOF,))
    _require_exact_cpu64("qd", qd, (ARM_DOF,))
    if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int):
        raise TypeError("horizon_steps must be an integer")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError("dt must be a real number")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")

    step_dt = float(dt)
    identity = torch.eye(ARM_DOF, dtype=torch.float64)
    q_from_qdd = torch.zeros(
        (horizon_steps * ARM_DOF, horizon_steps * ARM_DOF), dtype=torch.float64
    )
    qd_from_qdd = torch.zeros_like(q_from_qdd)
    for step in range(horizon_steps):
        for control in range(step + 1):
            row = slice(step * ARM_DOF, (step + 1) * ARM_DOF)
            column = slice(control * ARM_DOF, (control + 1) * ARM_DOF)
            q_from_qdd[row, column] = step_dt**2 * (step - control + 0.5) * identity
            qd_from_qdd[row, column] = step_dt * identity
    elapsed = step_dt * torch.arange(1, horizon_steps + 1, dtype=torch.float64)
    return CondensedArmDynamics(
        q_offset=q.repeat(horizon_steps) + (elapsed[:, None] * qd).reshape(-1),
        qd_offset=qd.repeat(horizon_steps),
        q_from_qdd=q_from_qdd,
        qd_from_qdd=qd_from_qdd,
    )


def _block_diagonal(matrix: torch.Tensor, count: int) -> torch.Tensor:
    result = torch.zeros(
        (count * matrix.shape[0], count * matrix.shape[1]), dtype=torch.float64
    )
    for index in range(count):
        rows = slice(index * matrix.shape[0], (index + 1) * matrix.shape[0])
        columns = slice(index * matrix.shape[1], (index + 1) * matrix.shape[1])
        result[rows, columns] = matrix
    return result


def _quadratic_term(
    matrix: torch.Tensor, offset: torch.Tensor, weight: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        2.0 * weight * matrix.transpose(0, 1) @ matrix,
        2.0 * weight * matrix.transpose(0, 1) @ offset,
    )


class LinearizedArmMpc:
    """Stateful deterministic arm MPC with last-safe-reference fallback."""

    def __init__(self, cfg: ArmMpcCfg | None = None) -> None:
        self.cfg = ArmMpcCfg() if cfg is None else cfg
        if not isinstance(self.cfg, ArmMpcCfg):
            raise TypeError("cfg must be an ArmMpcCfg")
        self._last_safe: ArmMpcSolution | None = None

    def _problem(self, sample: ArmMpcInput) -> DenseQpProblem:
        horizon = self.cfg.horizon_steps
        dimension = horizon * ARM_DOF
        dynamics = condense_arm_dynamics(
            sample.q, sample.qd, horizon_steps=horizon, dt=self.cfg.dt
        )
        jacobian = _block_diagonal(sample.jacobian_b, horizon)
        q_initial = sample.q.repeat(horizon)
        pose_matrix = jacobian @ dynamics.q_from_qdd
        pose_offset = (
            sample.ee_pose_b.repeat(horizon)
            + jacobian @ (dynamics.q_offset - q_initial)
            - sample.target_pose_b.reshape(-1)
        )
        twist_matrix = jacobian @ dynamics.qd_from_qdd
        twist_offset = (
            jacobian @ dynamics.qd_offset - sample.target_twist_b.reshape(-1)
        )

        hessian, gradient = _quadratic_term(
            pose_matrix, pose_offset, self.cfg.pose_weight
        )
        term_h, term_g = _quadratic_term(
            twist_matrix, twist_offset, self.cfg.twist_weight
        )
        hessian += term_h
        gradient += term_g

        identity = torch.eye(dimension, dtype=torch.float64)
        hessian += 2.0 * self.cfg.acceleration_weight * identity
        slew = identity.clone()
        for step in range(1, horizon):
            rows = slice(step * ARM_DOF, (step + 1) * ARM_DOF)
            previous = slice((step - 1) * ARM_DOF, step * ARM_DOF)
            slew[rows, previous] = -torch.eye(ARM_DOF, dtype=torch.float64)
        hessian += (
            2.0
            * self.cfg.acceleration_slew_weight
            * slew.transpose(0, 1)
            @ slew
        )
        rest_offset = dynamics.q_offset - q_initial
        term_h, term_g = _quadratic_term(
            dynamics.q_from_qdd, rest_offset, self.cfg.rest_posture_weight
        )
        hessian += term_h + 2.0 * self.cfg.hessian_regularization * identity
        gradient += term_g

        q_min = sample.q_min.repeat(horizon)
        q_max = sample.q_max.repeat(horizon)
        qd_max = sample.qd_max.repeat(horizon)
        mass = _block_diagonal(sample.arm_mass_matrix, horizon)
        bias = sample.arm_bias.repeat(horizon)
        effort = sample.effort_max.repeat(horizon)
        inequality_matrix = torch.cat(
            (
                dynamics.q_from_qdd,
                -dynamics.q_from_qdd,
                dynamics.qd_from_qdd,
                -dynamics.qd_from_qdd,
                mass,
                -mass,
            ),
            dim=0,
        )
        inequality_upper = torch.cat(
            (
                q_max - dynamics.q_offset,
                dynamics.q_offset - q_min,
                qd_max - dynamics.qd_offset,
                qd_max + dynamics.qd_offset,
                effort - bias,
                effort + bias,
            )
        )
        return DenseQpProblem(
            hessian=hessian,
            gradient=gradient,
            equality_matrix=torch.empty((0, dimension), dtype=torch.float64),
            equality_rhs=torch.empty(0, dtype=torch.float64),
            inequality_matrix=inequality_matrix,
            inequality_upper=inequality_upper,
            lower_bound=-sample.qdd_max.repeat(horizon),
            upper_bound=sample.qdd_max.repeat(horizon),
        )

    def _diagnostics(
        self,
        sample: ArmMpcInput,
        rollout: LinearizedArmRollout,
        qdd: torch.Tensor,
        *,
        iterations: int,
    ) -> ArmMpcDiagnostics:
        margin = torch.minimum(
            rollout.q - sample.q_min, sample.q_max - rollout.q
        )
        normalized = qdd.abs() / sample.qdd_max
        pose_error = rollout.pose_delta_b + sample.ee_pose_b - sample.target_pose_b
        singular_values = torch.linalg.svdvals(sample.jacobian_b)
        return ArmMpcDiagnostics(
            feasible=True,
            fallback_used=False,
            fallback_reason=None,
            iterations=iterations,
            saturation_fraction=float((normalized >= 1.0 - 1.0e-6).double().mean().item()),
            sigma_min=float(singular_values.min().item()),
            min_joint_margin=float(margin.min().item()),
            mean_joint_margin=float(margin.mean().item()),
            ee_position_error=float(torch.linalg.vector_norm(pose_error[-1, :3]).item()),
            ee_orientation_error=float(torch.linalg.vector_norm(pose_error[-1, 3:]).item()),
        )

    @staticmethod
    def _clone_solution(solution: ArmMpcSolution) -> ArmMpcSolution:
        return ArmMpcSolution(
            q_ref=solution.q_ref.clone(),
            qd_ref=solution.qd_ref.clone(),
            qdd=solution.qdd.clone(),
            predicted_q=solution.predicted_q.clone(),
            predicted_qd=solution.predicted_qd.clone(),
            predicted_pose_b=solution.predicted_pose_b.clone(),
            predicted_twist_b=solution.predicted_twist_b.clone(),
            predicted_dynamic_mount_wrench_b=solution.predicted_dynamic_mount_wrench_b.clone(),
            diagnostics=solution.diagnostics,
        )

    def _fallback(self, sample: ArmMpcInput, reason: str) -> ArmMpcSolution:
        if self._last_safe is None:
            q_ref = sample.q.clone()
            qd_ref = torch.zeros_like(sample.qd)
            qdd = torch.zeros((self.cfg.horizon_steps, ARM_DOF), dtype=torch.float64)
            rollout = rollout_linearized_arm(
                q_ref, qd_ref, qdd, sample.jacobian_b, self.cfg.dt
            )
            predicted_pose = sample.ee_pose_b + rollout.pose_delta_b
            predicted_twist = rollout.ee_twist_b
        else:
            safe = self._last_safe
            q_ref = safe.q_ref.clone()
            qd_ref = safe.qd_ref.clone()
            qdd = safe.qdd.clone()
            rollout = LinearizedArmRollout(
                q=safe.predicted_q.clone(),
                qd=safe.predicted_qd.clone(),
                pose_delta_b=safe.predicted_pose_b.clone() - sample.ee_pose_b,
                ee_twist_b=safe.predicted_twist_b.clone(),
            )
            predicted_pose = safe.predicted_pose_b.clone()
            predicted_twist = safe.predicted_twist_b.clone()
        pose_error = predicted_pose - sample.target_pose_b
        diagnostics = ArmMpcDiagnostics(
            feasible=False,
            fallback_used=True,
            fallback_reason=reason,
            iterations=0,
            saturation_fraction=0.0,
            sigma_min=float(torch.linalg.svdvals(sample.jacobian_b).min().item()),
            min_joint_margin=float(
                torch.minimum(rollout.q - sample.q_min, sample.q_max - rollout.q)
                .min()
                .item()
            ),
            mean_joint_margin=float(
                torch.minimum(rollout.q - sample.q_min, sample.q_max - rollout.q)
                .mean()
                .item()
            ),
            ee_position_error=float(
                torch.linalg.vector_norm(pose_error[-1, :3]).item()
            ),
            ee_orientation_error=float(
                torch.linalg.vector_norm(pose_error[-1, 3:]).item()
            ),
        )
        return ArmMpcSolution(
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd=qdd,
            predicted_q=rollout.q,
            predicted_qd=rollout.qd,
            predicted_pose_b=predicted_pose,
            predicted_twist_b=predicted_twist,
            predicted_dynamic_mount_wrench_b=torch.zeros(
                ARM_TASK_DOF, dtype=torch.float64
            ),
            diagnostics=diagnostics,
        )

    def plan(self, sample: ArmMpcInput) -> ArmMpcSolution:
        if not isinstance(sample, ArmMpcInput):
            raise TypeError("sample must be an ArmMpcInput")
        result = solve_reference_qp(
            self._problem(sample),
            tolerance=self.cfg.qp_tolerance,
            max_iterations=self.cfg.qp_max_iterations,
        )
        if not result.success or not torch.isfinite(result.solution).all().item():
            return self._fallback(sample, "qp_infeasible")
        qdd = result.solution.reshape(self.cfg.horizon_steps, ARM_DOF)
        rollout = rollout_linearized_arm(
            sample.q, sample.qd, qdd, sample.jacobian_b, self.cfg.dt
        )
        predicted_pose = sample.ee_pose_b + rollout.pose_delta_b
        solution = ArmMpcSolution(
            q_ref=rollout.q[0].clone(),
            qd_ref=rollout.qd[0].clone(),
            qdd=qdd,
            predicted_q=rollout.q,
            predicted_qd=rollout.qd,
            predicted_pose_b=predicted_pose,
            predicted_twist_b=rollout.ee_twist_b,
            predicted_dynamic_mount_wrench_b=sample.base_arm_coupling @ qdd[0],
            diagnostics=self._diagnostics(
                sample, rollout, qdd, iterations=result.iterations
            ),
        )
        tensors = (
            solution.q_ref,
            solution.qd_ref,
            solution.qdd,
            solution.predicted_q,
            solution.predicted_qd,
            solution.predicted_pose_b,
            solution.predicted_twist_b,
            solution.predicted_dynamic_mount_wrench_b,
        )
        if not all(torch.isfinite(value).all().item() for value in tensors):
            return self._fallback(sample, "nonfinite_solution")
        self._last_safe = self._clone_solution(solution)
        return solution
