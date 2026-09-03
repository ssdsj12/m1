"""Six-active-axis O6 fingertip contact MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    DenseQpResult,
    solve_reference_qp,
)
from .contracts import BimanualPhase
from .o6_contact_kinematics import PrecontactHandController


HAND_ACTIVE_DOF = 6
HAND_FINGERTIP_COUNT = 5
HAND_FORCE_DOF = 15
HAND_MPC_DT = 0.01
HAND_MPC_HORIZON_STEPS = 20
O6_ACTIVE_AXIS_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _float64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on device cpu; got {value.device}")
    if tuple(value.shape) != shape:
        qualifier = " for six active O6 axes" if name in {"q", "qd", "q_min", "q_max", "qd_max"} else ""
        raise ValueError(f"{name} must have exact shape {shape}{qualifier}; got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value.clone()


def _mask(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("contact_mask must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"contact_mask must have dtype torch.bool; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"contact_mask must be on device cpu; got {value.device}")
    if tuple(value.shape) != (HAND_FINGERTIP_COUNT,):
        raise ValueError("contact_mask must have exact shape (5,)")
    return value.clone()


@dataclass(frozen=True)
class HandMpcCfg:
    dt: float = HAND_MPC_DT
    horizon_steps: int = HAND_MPC_HORIZON_STEPS
    wrench_weight: float = 500.0
    force_slew_weight: float = 1.0
    joint_rate_weight: float = 0.1
    contact_stiffness_n_m: float = 100.0
    normal_force_max: float = 10.0
    friction_coefficient: float = 0.8
    hessian_regularization: float = 1.0e-8
    qp_tolerance: float = 1.0e-7
    qp_max_iterations: int = 256

    def __post_init__(self) -> None:
        _positive("dt", self.dt)
        if isinstance(self.horizon_steps, bool) or not isinstance(
            self.horizon_steps, int
        ):
            raise TypeError("horizon_steps must be an integer")
        if self.horizon_steps != HAND_MPC_HORIZON_STEPS:
            raise ValueError(
                f"horizon_steps must equal the frozen value {HAND_MPC_HORIZON_STEPS}"
            )
        for name in (
            "wrench_weight",
            "force_slew_weight",
            "joint_rate_weight",
            "contact_stiffness_n_m",
            "normal_force_max",
            "friction_coefficient",
            "hessian_regularization",
            "qp_tolerance",
        ):
            _positive(name, getattr(self, name))
        if isinstance(self.qp_max_iterations, bool) or not isinstance(
            self.qp_max_iterations, int
        ) or self.qp_max_iterations <= 0:
            raise ValueError("qp_max_iterations must be a positive integer")

    @property
    def horizon_seconds(self) -> float:
        return float(self.dt) * self.horizon_steps


@dataclass(frozen=True)
class HandMpcDiagnostics:
    feasible: bool
    fallback_used: bool
    fallback_reason: str | None
    wrench_error_norm: float
    slip_margin: float
    iterations: int


@dataclass(frozen=True)
class HandMpcInput:
    q: torch.Tensor
    qd: torch.Tensor
    fingertip_forces_b: torch.Tensor
    contact_mask: torch.Tensor
    contact_jacobian: torch.Tensor
    wrench_map: torch.Tensor
    target_wrench_b: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    qd_max: torch.Tensor
    phase: BimanualPhase = BimanualPhase.GRASP

    def __post_init__(self) -> None:
        for name, shape in (
            ("q", (HAND_ACTIVE_DOF,)),
            ("qd", (HAND_ACTIVE_DOF,)),
            ("fingertip_forces_b", (HAND_FINGERTIP_COUNT, 3)),
            ("contact_jacobian", (HAND_FORCE_DOF, HAND_ACTIVE_DOF)),
            ("wrench_map", (6, HAND_FORCE_DOF)),
            ("target_wrench_b", (6,)),
            ("q_min", (HAND_ACTIVE_DOF,)),
            ("q_max", (HAND_ACTIVE_DOF,)),
            ("qd_max", (HAND_ACTIVE_DOF,)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))
        object.__setattr__(self, "contact_mask", _mask(self.contact_mask))
        if not isinstance(self.phase, BimanualPhase):
            raise TypeError("phase must be BimanualPhase")
        if not torch.all(self.q_min < self.q_max).item():
            raise ValueError("q_min must be strictly below q_max")
        if not torch.all(self.qd_max > 0.0).item():
            raise ValueError("qd_max must be strictly positive")


@dataclass(frozen=True)
class HandMpcSolution:
    q_ref: torch.Tensor
    qd_ref: torch.Tensor
    predicted_forces_b: torch.Tensor
    predicted_wrench_b: torch.Tensor
    diagnostics: HandMpcDiagnostics

    def __post_init__(self) -> None:
        for name, shape in (
            ("q_ref", (HAND_ACTIVE_DOF,)),
            ("qd_ref", (HAND_ACTIVE_DOF,)),
            ("predicted_forces_b", (HAND_FINGERTIP_COUNT, 3)),
            ("predicted_wrench_b", (6,)),
        ):
            object.__setattr__(self, name, _float64(name, getattr(self, name), shape))
        if not isinstance(self.diagnostics, HandMpcDiagnostics):
            raise TypeError("diagnostics must be HandMpcDiagnostics")


def build_hand_contact_qp(sample: HandMpcInput, cfg: HandMpcCfg) -> DenseQpProblem:
    """Build a linear-compliance QP over six rates and five 3D forces."""

    if not isinstance(sample, HandMpcInput):
        raise TypeError("sample must be HandMpcInput")
    if not isinstance(cfg, HandMpcCfg):
        raise TypeError("cfg must be HandMpcCfg")
    dimension = HAND_ACTIVE_DOF + HAND_FORCE_DOF
    rate_slice = slice(0, HAND_ACTIVE_DOF)
    force_slice = slice(HAND_ACTIVE_DOF, dimension)
    hessian = 2.0 * cfg.hessian_regularization * torch.eye(
        dimension, dtype=torch.float64
    )
    gradient = torch.zeros(dimension, dtype=torch.float64)

    wrench_matrix = torch.zeros((6, dimension), dtype=torch.float64)
    wrench_matrix[:, force_slice] = sample.wrench_map
    hessian += 2.0 * cfg.wrench_weight * wrench_matrix.T @ wrench_matrix
    gradient += -2.0 * cfg.wrench_weight * wrench_matrix.T @ sample.target_wrench_b

    force_selector = torch.zeros((HAND_FORCE_DOF, dimension), dtype=torch.float64)
    force_selector[:, force_slice] = torch.eye(HAND_FORCE_DOF, dtype=torch.float64)
    current_force = sample.fingertip_forces_b.reshape(-1).clone()
    current_force[~sample.contact_mask.repeat_interleave(3)] = 0.0
    hessian += 2.0 * cfg.force_slew_weight * force_selector.T @ force_selector
    gradient += -2.0 * cfg.force_slew_weight * force_selector.T @ current_force
    hessian[rate_slice, rate_slice] += 2.0 * cfg.joint_rate_weight * torch.eye(
        HAND_ACTIVE_DOF, dtype=torch.float64
    )

    equality_matrix = torch.zeros((HAND_FORCE_DOF, dimension), dtype=torch.float64)
    equality_rhs = torch.zeros(HAND_FORCE_DOF, dtype=torch.float64)
    compliance = cfg.contact_stiffness_n_m * cfg.dt
    for fingertip in range(HAND_FINGERTIP_COUNT):
        rows = slice(3 * fingertip, 3 * (fingertip + 1))
        equality_matrix[rows, force_slice] = torch.eye(
            HAND_FORCE_DOF, dtype=torch.float64
        )[rows]
        if sample.contact_mask[fingertip].item():
            equality_matrix[rows, rate_slice] = -compliance * sample.contact_jacobian[rows]
            equality_rhs[rows] = sample.fingertip_forces_b[fingertip]

    inequalities = []
    for fingertip in range(HAND_FINGERTIP_COUNT):
        start = HAND_ACTIVE_DOF + 3 * fingertip
        if not sample.contact_mask[fingertip].item():
            continue
        for tangent in (0, 1):
            for sign in (-1.0, 1.0):
                row = torch.zeros(dimension, dtype=torch.float64)
                row[start + tangent] = sign
                row[start + 2] = -cfg.friction_coefficient
                inequalities.append(row)
    inequality_matrix = (
        torch.stack(inequalities)
        if inequalities
        else torch.empty((0, dimension), dtype=torch.float64)
    )
    lower = torch.full((dimension,), -torch.inf, dtype=torch.float64)
    upper = torch.full((dimension,), torch.inf, dtype=torch.float64)
    lower[rate_slice] = torch.maximum(
        -sample.qd_max, (sample.q_min - sample.q) / cfg.dt
    )
    upper[rate_slice] = torch.minimum(
        sample.qd_max, (sample.q_max - sample.q) / cfg.dt
    )
    for fingertip in range(HAND_FINGERTIP_COUNT):
        start = HAND_ACTIVE_DOF + 3 * fingertip
        if sample.contact_mask[fingertip].item():
            tangential = cfg.friction_coefficient * cfg.normal_force_max
            lower[start : start + 2] = -tangential
            upper[start : start + 2] = tangential
            lower[start + 2] = 0.0
            upper[start + 2] = cfg.normal_force_max
        else:
            lower[start : start + 3] = 0.0
            upper[start : start + 3] = 0.0
    return DenseQpProblem(
        hessian=hessian,
        gradient=gradient,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_upper=torch.zeros(inequality_matrix.shape[0], dtype=torch.float64),
        lower_bound=lower,
        upper_bound=upper,
    )


def _slip_margin(forces: torch.Tensor, mask: torch.Tensor, mu: float) -> float:
    if not mask.any().item():
        return 0.0
    active = forces[mask]
    margin = mu * active[:, 2] - active[:, :2].abs().max(dim=1).values
    return float(margin.min().item())


class O6HandMpc:
    """Stateful receding-horizon O6 contact controller with safe hold fallback."""

    def __init__(self, cfg: HandMpcCfg | None = None) -> None:
        self.cfg = HandMpcCfg() if cfg is None else cfg
        if not isinstance(self.cfg, HandMpcCfg):
            raise TypeError("cfg must be HandMpcCfg")
        self._last_safe: HandMpcSolution | None = None
        self.precontact = PrecontactHandController()

    @staticmethod
    def _clone(
        solution: HandMpcSolution, diagnostics: HandMpcDiagnostics | None = None
    ) -> HandMpcSolution:
        return HandMpcSolution(
            q_ref=solution.q_ref,
            qd_ref=solution.qd_ref,
            predicted_forces_b=solution.predicted_forces_b,
            predicted_wrench_b=solution.predicted_wrench_b,
            diagnostics=solution.diagnostics if diagnostics is None else diagnostics,
        )

    def _fallback(self, sample: HandMpcInput, reason: str) -> HandMpcSolution:
        if self._last_safe is not None:
            safe = self._last_safe
            diagnostics = HandMpcDiagnostics(
                feasible=False,
                fallback_used=True,
                fallback_reason=reason,
                wrench_error_norm=float(
                    torch.linalg.vector_norm(safe.predicted_wrench_b - sample.target_wrench_b).item()
                ),
                slip_margin=safe.diagnostics.slip_margin,
                iterations=0,
            )
            return self._clone(safe, diagnostics)
        forces = sample.fingertip_forces_b.clone()
        forces[~sample.contact_mask] = 0.0
        wrench = sample.wrench_map @ forces.reshape(-1)
        return HandMpcSolution(
            q_ref=sample.q,
            qd_ref=torch.zeros(HAND_ACTIVE_DOF, dtype=torch.float64),
            predicted_forces_b=forces,
            predicted_wrench_b=wrench,
            diagnostics=HandMpcDiagnostics(
                feasible=False,
                fallback_used=True,
                fallback_reason=reason,
                wrench_error_norm=float(
                    torch.linalg.vector_norm(wrench - sample.target_wrench_b).item()
                ),
                slip_margin=_slip_margin(
                    forces, sample.contact_mask, self.cfg.friction_coefficient
                ),
                iterations=0,
            ),
        )

    def _solution(
        self, sample: HandMpcInput, result: DenseQpResult
    ) -> HandMpcSolution:
        rate = result.solution[:HAND_ACTIVE_DOF]
        forces = result.solution[HAND_ACTIVE_DOF:].reshape(HAND_FINGERTIP_COUNT, 3).clone()
        # KKT least-squares can leave subnormal residuals on equality-fixed
        # inactive contacts.  The public contact contract is exactly zero.
        forces[~sample.contact_mask] = 0.0
        wrench = sample.wrench_map @ forces.reshape(-1)
        return HandMpcSolution(
            q_ref=sample.q + self.cfg.dt * rate,
            qd_ref=rate,
            predicted_forces_b=forces,
            predicted_wrench_b=wrench,
            diagnostics=HandMpcDiagnostics(
                feasible=True,
                fallback_used=False,
                fallback_reason=None,
                wrench_error_norm=float(
                    torch.linalg.vector_norm(wrench - sample.target_wrench_b).item()
                ),
                slip_margin=_slip_margin(
                    forces, sample.contact_mask, self.cfg.friction_coefficient
                ),
                iterations=result.iterations,
            ),
        )

    def plan(self, sample: HandMpcInput) -> HandMpcSolution:
        if not isinstance(sample, HandMpcInput):
            raise TypeError("sample must be HandMpcInput")
        if sample.phase in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD}:
            q_ref, qd_ref = self.precontact.reference(
                sample.q, sample.contact_mask, sample.phase
            )
            forces = sample.fingertip_forces_b.clone()
            forces[~sample.contact_mask] = 0.0
            wrench = sample.wrench_map @ forces.reshape(-1)
            solution = HandMpcSolution(
                q_ref=q_ref,
                qd_ref=qd_ref,
                predicted_forces_b=forces,
                predicted_wrench_b=wrench,
                diagnostics=HandMpcDiagnostics(
                    feasible=True,
                    fallback_used=False,
                    fallback_reason=None,
                    wrench_error_norm=float(
                        torch.linalg.vector_norm(
                            wrench - sample.target_wrench_b
                        ).item()
                    ),
                    slip_margin=_slip_margin(
                        forces,
                        sample.contact_mask,
                        self.cfg.friction_coefficient,
                    ),
                    iterations=0,
                ),
            )
            self._last_safe = self._clone(solution)
            return solution
        result = solve_reference_qp(
            build_hand_contact_qp(sample, self.cfg),
            tolerance=self.cfg.qp_tolerance,
            max_iterations=self.cfg.qp_max_iterations,
        )
        if not result.success or not torch.isfinite(result.solution).all().item():
            return self._fallback(sample, "qp_infeasible")
        solution = self._solution(sample, result)
        self._last_safe = self._clone(solution)
        return solution
