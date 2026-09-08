"""Hard 200 Hz projection for learned or teacher 43-effort candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    solve_reference_qp,
)

from .contracts import BimanualPhase, FullDynamicsState, _float64
from .reduced_dynamics import condense_constrained_dynamics


ACTIVE_DOF = 43
GENERALIZED_DOF = 59
LOADED_PHASES = {
    BimanualPhase.GRASP,
    BimanualPhase.LIFT,
    BimanualPhase.HOLD,
    BimanualPhase.LOWER,
    BimanualPhase.HOLD_SAFE,
    BimanualPhase.LOWER_SAFE,
}


@dataclass(frozen=True)
class SafetyInput:
    candidate_effort: torch.Tensor
    safe_effort: torch.Tensor
    dynamics: FullDynamicsState
    active_generalized_ids: torch.Tensor
    active_q: torch.Tensor
    active_qd: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    qd_max: torch.Tensor
    effort_limits: torch.Tensor
    collision_distances: torch.Tensor
    collision_jacobian: torch.Tensor
    base_error: torch.Tensor
    force_closure_margin: float
    phase: BimanualPhase

    def __post_init__(self) -> None:
        candidate = self.candidate_effort
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("candidate_effort must be a torch.Tensor")
        if candidate.dtype != torch.float64 or candidate.device.type != "cpu" or candidate.shape != (43,):
            raise ValueError("candidate_effort must be CPU float64 with shape (43,)")
        object.__setattr__(self, "candidate_effort", candidate.clone())
        for name, shape in (
            ("safe_effort", (43,)),
            ("active_q", (43,)),
            ("active_qd", (43,)),
            ("q_min", (43,)),
            ("q_max", (43,)),
            ("qd_max", (43,)),
            ("effort_limits", (43,)),
            ("base_error", (6,)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))
        if not isinstance(self.dynamics, FullDynamicsState):
            raise TypeError("dynamics must be FullDynamicsState")
        ids = self.active_generalized_ids
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.device.type != "cpu" or ids.shape != (43,):
            raise ValueError("active_generalized_ids must be CPU int64 with shape (43,)")
        if torch.any((ids < 0) | (ids >= GENERALIZED_DOF)).item() or torch.unique(ids).numel() != ACTIVE_DOF:
            raise ValueError("active_generalized_ids must be unique and in [0, 59)")
        object.__setattr__(self, "active_generalized_ids", ids.clone())
        distances = self.collision_distances
        jacobian = self.collision_jacobian
        if distances.dtype != torch.float64 or distances.device.type != "cpu" or distances.ndim != 1:
            raise ValueError("collision_distances must be a CPU float64 vector")
        if jacobian.dtype != torch.float64 or jacobian.device.type != "cpu" or jacobian.shape != (distances.numel(), 43):
            raise ValueError("collision_jacobian must have shape (collision_count, 43)")
        if not torch.isfinite(distances).all().item() or not torch.isfinite(jacobian).all().item():
            raise ValueError("collision inputs must be finite")
        object.__setattr__(self, "collision_distances", distances.clone())
        object.__setattr__(self, "collision_jacobian", jacobian.clone())
        if not torch.all(self.q_min < self.q_max).item():
            raise ValueError("q_min must be below q_max")
        if not torch.all(self.qd_max > 0.0).item() or not torch.all(self.effort_limits > 0.0).item():
            raise ValueError("velocity and effort limits must be positive")
        if not math.isfinite(self.force_closure_margin):
            raise ValueError("force_closure_margin must be finite")
        if not isinstance(self.phase, BimanualPhase):
            raise TypeError("phase must be BimanualPhase")


@dataclass(frozen=True)
class SafetyResult:
    effort: torch.Tensor
    feasible: bool
    active_constraints: tuple[str, ...]
    fallback_reason: str | None
    dynamics_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", _float64("effort", self.effort, (43,)))
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be bool")
        if self.fallback_reason is not None and not self.fallback_reason.strip():
            raise ValueError("fallback_reason must be None or non-empty")
        if not math.isfinite(self.dynamics_residual) or self.dynamics_residual < 0.0:
            raise ValueError("dynamics_residual must be finite and non-negative")


class SafetyProjection:
    def __init__(
        self,
        *,
        dt: float = 0.005,
        collision_margin_m: float = 0.02,
        joint_position_margin_rad: float = 0.0,
        joint_velocity_margin_fraction: float = 0.0,
        externally_servoed_hand: bool = False,
        wheel_contact_acceleration_tolerance: float = 1.0e-6,
        base_acceleration_tolerance: float = 5.0,
        base_kp: float = 20.0,
        qp_tolerance: float = 1.0e-7,
    ) -> None:
        self.dt = float(dt)
        self.collision_margin_m = float(collision_margin_m)
        self.joint_position_margin_rad = float(joint_position_margin_rad)
        if (
            not math.isfinite(self.joint_position_margin_rad)
            or self.joint_position_margin_rad < 0.0
        ):
            raise ValueError("joint_position_margin_rad must be finite and non-negative")
        self.joint_velocity_margin_fraction = float(
            joint_velocity_margin_fraction
        )
        if (
            not math.isfinite(self.joint_velocity_margin_fraction)
            or not 0.0 <= self.joint_velocity_margin_fraction < 1.0
        ):
            raise ValueError(
                "joint_velocity_margin_fraction must be in [0, 1)"
            )
        if not isinstance(externally_servoed_hand, bool):
            raise TypeError("externally_servoed_hand must be bool")
        self.externally_servoed_hand = externally_servoed_hand
        self.wheel_contact_acceleration_tolerance = float(
            wheel_contact_acceleration_tolerance
        )
        self.base_acceleration_tolerance = float(base_acceleration_tolerance)
        self.base_kp = float(base_kp)
        self.qp_tolerance = float(qp_tolerance)

    @staticmethod
    def _fallback(sample: SafetyInput, reason: str) -> SafetyResult:
        safe = torch.clamp(
            sample.safe_effort,
            min=-sample.effort_limits,
            max=sample.effort_limits,
        )
        safe[12:16] = 0.0
        return SafetyResult(
            effort=safe,
            feasible=False,
            active_constraints=("wheel_lock",),
            fallback_reason=reason,
            dynamics_residual=0.0,
        )

    def project(self, sample: SafetyInput) -> SafetyResult:
        if not isinstance(sample, SafetyInput):
            raise TypeError("sample must be SafetyInput")
        if not torch.isfinite(sample.candidate_effort).all().item():
            return self._fallback(sample, "nonfinite_candidate")
        if sample.phase in LOADED_PHASES and sample.force_closure_margin <= 0.0:
            return self._fallback(sample, "force_closure_lost")
        reduced = condense_constrained_dynamics(sample.dynamics)
        ids = sample.active_generalized_ids
        qdd_offset = reduced.qdd_offset[ids]
        qdd_map = reduced.qdd_from_effort[ids]
        position_offset = sample.active_q + self.dt * sample.active_qd + 0.5 * self.dt**2 * qdd_offset
        position_map = 0.5 * self.dt**2 * qdd_map
        joint_margin = torch.minimum(
            self.joint_position_margin_rad * torch.ones(43, dtype=torch.float64),
            0.25 * (sample.q_max - sample.q_min),
        )
        safe_q_min = sample.q_min + joint_margin
        safe_q_max = sample.q_max - joint_margin
        velocity_offset = sample.active_qd + self.dt * qdd_offset
        velocity_map = self.dt * qdd_map
        safe_qd_max = (
            1.0 - self.joint_velocity_margin_fraction
        ) * sample.qd_max
        inequality_rows = [position_map, -position_map, velocity_map, -velocity_map]
        inequality_upper = [
            safe_q_max - position_offset,
            -(safe_q_min - position_offset),
            safe_qd_max - velocity_offset,
            safe_qd_max + velocity_offset,
        ]
        if self.externally_servoed_hand:
            for upper_vector in inequality_upper:
                upper_vector[31:43] = 1.0e12
        contact_offset = (
            sample.dynamics.wheel_contact_jacobian @ reduced.qdd_offset
            + sample.dynamics.wheel_contact_bias
        )
        contact_map = sample.dynamics.wheel_contact_jacobian @ reduced.qdd_from_effort
        contact_map[:, 31:43] = 0.0
        contact_tolerance = self.wheel_contact_acceleration_tolerance * torch.ones(12, dtype=torch.float64)
        inequality_rows.extend((contact_map, -contact_map))
        inequality_upper.extend((contact_tolerance - contact_offset, contact_tolerance + contact_offset))
        base_target = -self.base_kp * sample.base_error
        base_offset = reduced.qdd_offset[:6]
        base_map = reduced.qdd_from_effort[:6].clone()
        base_map[:, 31:43] = 0.0
        base_tolerance = self.base_acceleration_tolerance * torch.ones(6, dtype=torch.float64)
        inequality_rows.extend((base_map, -base_map))
        inequality_upper.extend((base_target + base_tolerance - base_offset, -base_target + base_tolerance + base_offset))
        collision_map = -(self.dt**2) * sample.collision_jacobian
        inequality_rows.append(collision_map)
        inequality_upper.append(sample.collision_distances - self.collision_margin_m)
        lower = -sample.effort_limits.clone()
        upper = sample.effort_limits.clone()
        lower[12:16] = 0.0
        upper[12:16] = 0.0
        inequality_matrix = torch.cat(inequality_rows)
        inequality_upper_vector = torch.cat(inequality_upper)
        locked_candidate = sample.candidate_effort.clone()
        locked_candidate[12:16] = 0.0
        candidate_is_feasible = bool(
            torch.all(locked_candidate >= lower - self.qp_tolerance).item()
            and torch.all(locked_candidate <= upper + self.qp_tolerance).item()
            and torch.all(
                inequality_matrix @ locked_candidate
                <= inequality_upper_vector + self.qp_tolerance
            ).item()
        )
        if candidate_is_feasible:
            qdd = reduced.qdd_offset + reduced.qdd_from_effort @ locked_candidate
            contact_force = (
                reduced.contact_offset
                + reduced.contact_from_effort @ locked_candidate
            )
            residual = (
                sample.dynamics.mass_matrix @ qdd
                + sample.dynamics.bias
                - sample.dynamics.actuation_matrix @ locked_candidate
                - sample.dynamics.wheel_contact_jacobian.T @ contact_force
            )
            residual_norm = float(torch.max(torch.abs(residual)).item())
            if residual_norm <= 1.0e-6:
                return SafetyResult(
                    effort=locked_candidate,
                    feasible=True,
                    active_constraints=(
                        "effort_limits",
                        "wheel_lock",
                        "joint_position",
                        "joint_velocity",
                        "wheel_contact",
                        "base_reference",
                        "collision",
                    ),
                    fallback_reason=None,
                    dynamics_residual=residual_norm,
                )
        result = solve_reference_qp(
            DenseQpProblem(
                hessian=2.0 * torch.eye(ACTIVE_DOF, dtype=torch.float64),
                gradient=-2.0 * sample.candidate_effort,
                equality_matrix=torch.empty((0, ACTIVE_DOF), dtype=torch.float64),
                equality_rhs=torch.empty(0, dtype=torch.float64),
                inequality_matrix=inequality_matrix,
                inequality_upper=inequality_upper_vector,
                lower_bound=lower,
                upper_bound=upper,
            ),
            tolerance=self.qp_tolerance,
            max_iterations=512,
        )
        if not result.success or not torch.isfinite(result.solution).all().item():
            return self._fallback(sample, "safety_qp_infeasible")
        effort = result.solution.clone()
        effort[12:16] = 0.0
        qdd = reduced.qdd_offset + reduced.qdd_from_effort @ effort
        contact_force = reduced.contact_offset + reduced.contact_from_effort @ effort
        residual = sample.dynamics.mass_matrix @ qdd + sample.dynamics.bias - sample.dynamics.actuation_matrix @ effort - sample.dynamics.wheel_contact_jacobian.T @ contact_force
        residual_norm = float(torch.max(torch.abs(residual)).item())
        if residual_norm > 1.0e-6:
            return self._fallback(sample, "forward_dynamics_residual")
        return SafetyResult(
            effort=effort,
            feasible=True,
            active_constraints=(
                "effort_limits",
                "wheel_lock",
                "joint_position",
                "joint_velocity",
                "wheel_contact",
                "base_reference",
                "collision",
            ),
            fallback_reason=None,
            dynamics_residual=residual_norm,
        )


__all__ = ["SafetyInput", "SafetyProjection", "SafetyResult"]
