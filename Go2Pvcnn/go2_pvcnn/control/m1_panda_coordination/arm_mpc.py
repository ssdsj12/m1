"""Pure float64 contracts and linearized dynamics for the Panda arm MPC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import require_tensor


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
