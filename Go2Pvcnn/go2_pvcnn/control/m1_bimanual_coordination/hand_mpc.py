"""Six-active-axis O6 fingertip contact MPC."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    DenseQpResult,
    solve_reference_qp,
)
from .contracts import BimanualPhase
from .o6_contact_kinematics import FINGER_ACTIVE_COLUMNS, PrecontactHandController


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
    fingertip_prior_weight: float = 1.0
    precontact_tracking_weight: float = 100.0

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
            "fingertip_prior_weight",
            "precontact_tracking_weight",
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
    prior_configured: bool = False
    prior_enabled: bool = False
    prior_fallback_reason: str | None = None
    prior_component: int | None = None
    prior_probability: float | None = None
    prior_inference_ms: float = 0.0
    prior_precision_min: float | None = None
    prior_precision_max: float | None = None
    prior_mahalanobis: float | None = None
    prior_cost: float | None = None
    regularized_tip_velocity_delta_norm: float | None = None
    prior_qp_accepted: bool = False


@dataclass(frozen=True)
class HandMpcInput:
    q: torch.Tensor
    qd: torch.Tensor
    fingertip_forces_b: torch.Tensor
    fingertip_positions_b: torch.Tensor
    contact_mask: torch.Tensor
    contact_jacobian: torch.Tensor
    wrench_map: torch.Tensor
    target_wrench_b: torch.Tensor
    q_min: torch.Tensor
    q_max: torch.Tensor
    qd_max: torch.Tensor
    phase: BimanualPhase = BimanualPhase.GRASP
    palm_pose_b: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name, shape in (
            ("q", (HAND_ACTIVE_DOF,)),
            ("qd", (HAND_ACTIVE_DOF,)),
            ("fingertip_forces_b", (HAND_FINGERTIP_COUNT, 3)),
            ("fingertip_positions_b", (HAND_FINGERTIP_COUNT, 3)),
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
        if self.palm_pose_b is not None:
            object.__setattr__(
                self,
                "palm_pose_b",
                _float64("palm_pose_b", self.palm_pose_b, (6,)),
            )
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


def _validated_prior_target(target: Any) -> tuple[torch.Tensor, torch.Tensor, int, float] | None:
    """Copy a duck-typed runtime target after rechecking the control boundary."""

    mean = getattr(target, "mean_velocity", None)
    precision = getattr(target, "precision", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(precision, torch.Tensor):
        return None
    if (
        mean.dtype != torch.float64
        or precision.dtype != torch.float64
        or mean.device.type != "cpu"
        or precision.device.type != "cpu"
        or tuple(mean.shape) != (HAND_FORCE_DOF,)
        or tuple(precision.shape) != (HAND_FORCE_DOF,)
        or not torch.isfinite(mean).all().item()
        or not torch.isfinite(precision).all().item()
        or not torch.all(precision >= 0.0).item()
    ):
        return None
    component = getattr(target, "component", None)
    probability = getattr(target, "probability", None)
    if type(component) is not int or not 0 <= component < 4:
        return None
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        return None
    return mean.clone(), precision.clone(), component, float(probability)


def _prior_geometry(sample: HandMpcInput) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tip positions and the folded linear Jacobian in the palm frame."""

    if sample.palm_pose_b is None:
        # Pure Hand-MPC callers may provide geometry already expressed in the
        # palm frame. The physical runtime supplies palm_pose_b explicitly.
        return sample.fingertip_positions_b, sample.contact_jacobian
    from .palm_orientation_mpc import rotvec_to_matrix

    rotation_p_to_b = rotvec_to_matrix(sample.palm_pose_b[3:])
    positions_p = (
        sample.fingertip_positions_b - sample.palm_pose_b[:3]
    ) @ rotation_p_to_b
    jacobian_b = sample.contact_jacobian.reshape(HAND_FINGERTIP_COUNT, 3, HAND_ACTIVE_DOF)
    jacobian_p = (rotation_p_to_b.T @ jacobian_b).reshape(
        HAND_FORCE_DOF, HAND_ACTIVE_DOF
    )
    return positions_p, jacobian_p


def add_fingertip_prior(
    problem: DenseQpProblem,
    sample: HandMpcInput,
    target: Any,
    weight: float,
) -> DenseQpProblem:
    """Add a PSD fingertip-velocity cost without changing any hard constraint."""

    checked = _validated_prior_target(target)
    if checked is None:
        raise ValueError("invalid fingertip prior target")
    _positive("weight", weight)
    mean, precision_values, _, _ = checked
    precision_values[sample.contact_mask.repeat_interleave(3)] = 0.0
    _, prior_jacobian = _prior_geometry(sample)
    selector = torch.zeros(
        (HAND_FORCE_DOF, problem.gradient.numel()), dtype=torch.float64
    )
    selector[:, :HAND_ACTIVE_DOF] = prior_jacobian
    precision = torch.diag(precision_values)
    hessian = (
        problem.hessian
        + 2.0 * float(weight) * selector.T @ precision @ selector
    )
    gradient = (
        problem.gradient
        - 2.0 * float(weight) * selector.T @ precision @ mean
    )
    return replace(problem, hessian=hessian, gradient=gradient)


def build_hand_contact_qp(
    sample: HandMpcInput,
    cfg: HandMpcCfg,
    *,
    prior_target: Any | None = None,
) -> DenseQpProblem:
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
    problem = DenseQpProblem(
        hessian=hessian,
        gradient=gradient,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_upper=torch.zeros(inequality_matrix.shape[0], dtype=torch.float64),
        lower_bound=lower,
        upper_bound=upper,
    )
    if prior_target is None:
        return problem
    return add_fingertip_prior(
        problem, sample, prior_target, cfg.fingertip_prior_weight
    )


def _slip_margin(forces: torch.Tensor, mask: torch.Tensor, mu: float) -> float:
    if not mask.any().item():
        return 0.0
    active = forces[mask]
    margin = mu * active[:, 2] - active[:, :2].abs().max(dim=1).values
    return float(margin.min().item())


class O6HandMpc:
    """Stateful receding-horizon O6 contact controller with safe hold fallback."""

    def __init__(
        self,
        cfg: HandMpcCfg | None = None,
        *,
        expert_prior: Any | None = None,
    ) -> None:
        self.cfg = HandMpcCfg() if cfg is None else cfg
        if not isinstance(self.cfg, HandMpcCfg):
            raise TypeError("cfg must be HandMpcCfg")
        if expert_prior is not None and not callable(getattr(expert_prior, "target", None)):
            raise TypeError("expert_prior must expose target()")
        self.expert_prior = expert_prior
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

    def _accept(self, solution: HandMpcSolution) -> HandMpcSolution:
        self._last_safe = self._clone(solution)
        return solution

    def _precontact_solution(self, sample: HandMpcInput) -> HandMpcSolution:
        q_ref, qd_ref = self.precontact.reference(
            sample.q, sample.contact_mask, sample.phase
        )
        forces = sample.fingertip_forces_b.clone()
        forces[~sample.contact_mask] = 0.0
        wrench = sample.wrench_map @ forces.reshape(-1)
        return HandMpcSolution(
            q_ref=q_ref,
            qd_ref=qd_ref,
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
                iterations=0,
            ),
        )

    @staticmethod
    def _query_diagnostics(query: Any) -> tuple[str | None, float]:
        diagnostics = getattr(query, "diagnostics", None)
        enabled = getattr(diagnostics, "enabled", None)
        reason = getattr(diagnostics, "reason", None)
        elapsed = getattr(diagnostics, "inference_ms", 0.0)
        target = getattr(query, "target", None)
        if type(enabled) is not bool or enabled != (target is not None):
            return "invalid_prior", 0.0
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            return "invalid_prior", 0.0
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            return "invalid_prior", float(elapsed)
        return reason, float(elapsed)

    def _query_prior(
        self, sample: HandMpcInput, baseline: HandMpcSolution
    ) -> tuple[tuple[Any, torch.Tensor, torch.Tensor, int, float] | None, str | None, float]:
        # Keep the frozen-prior package out of the default controller import and
        # execution path.  Task 10 owns production artifact construction.
        try:
            from .expert_fingertip_prior.runtime import O6FingertipPriorInput

            fingertip_positions_p, contact_jacobian_p = _prior_geometry(sample)
            query = self.expert_prior.target(
                O6FingertipPriorInput(
                    fingertip_positions_b=fingertip_positions_p,
                    contact_jacobian=contact_jacobian_p,
                    qd=sample.qd,
                    contact_mask=sample.contact_mask,
                    phase=sample.phase,
                ),
                baseline.qd_ref,
            )
        except BaseException:
            return None, "prior_exception", 0.0
        try:
            reason, elapsed = self._query_diagnostics(query)
            target = getattr(query, "target", None)
            if reason is not None or target is None:
                return None, reason or "invalid_prior", elapsed
            checked = _validated_prior_target(target)
        except BaseException:
            return None, "invalid_prior", 0.0
        if checked is None:
            return None, "invalid_prior", elapsed
        mean, precision, component, probability = checked
        return (target, mean, precision, component, probability), None, elapsed

    def _prior_diagnostics(
        self,
        baseline: HandMpcSolution,
        sample: HandMpcInput,
        *,
        queried: tuple[Any, torch.Tensor, torch.Tensor, int, float] | None,
        inference_ms: float,
        result: HandMpcSolution | None = None,
        fallback_reason: str | None = None,
    ) -> HandMpcDiagnostics:
        values: dict[str, Any] = {
            "prior_configured": True,
            "prior_enabled": result is not None,
            "prior_fallback_reason": fallback_reason,
            "prior_inference_ms": inference_ms,
            "prior_qp_accepted": result is not None,
        }
        if queried is not None:
            _, mean, precision, component, probability = queried
            _, prior_jacobian = _prior_geometry(sample)
            active_precision = precision.clone()
            active_precision[sample.contact_mask.repeat_interleave(3)] = 0.0
            baseline_tip = prior_jacobian @ baseline.qd_ref
            compared = baseline if result is None else result
            compared_tip = prior_jacobian @ compared.qd_ref
            values.update(
                prior_component=component,
                prior_probability=probability,
                prior_precision_min=float(active_precision.min().item()),
                prior_precision_max=float(active_precision.max().item()),
                prior_mahalanobis=float(
                    ((baseline_tip - mean).square() * active_precision).sum().item()
                ),
                prior_cost=float(
                    ((compared_tip - mean).square() * active_precision).sum().item()
                ),
                regularized_tip_velocity_delta_norm=float(
                    torch.linalg.vector_norm(compared_tip - baseline_tip).item()
                ),
            )
        source = baseline.diagnostics if result is None else result.diagnostics
        return replace(source, **values)

    def _accept_same_cycle_baseline(
        self,
        sample: HandMpcInput,
        baseline: HandMpcSolution,
        reason: str,
        *,
        queried: tuple[Any, torch.Tensor, torch.Tensor, int, float] | None = None,
        inference_ms: float = 0.0,
    ) -> HandMpcSolution:
        diagnostics = self._prior_diagnostics(
            baseline,
            sample,
            queried=queried,
            inference_ms=inference_ms,
            fallback_reason=reason,
        )
        return self._accept(self._clone(baseline, diagnostics))

    def _solve_precontact_projection(
        self,
        sample: HandMpcInput,
        baseline: HandMpcSolution,
        queried: tuple[Any, torch.Tensor, torch.Tensor, int, float],
    ) -> HandMpcSolution | None:
        _, mean, precision_values, _, _ = queried
        precision_values = precision_values.clone()
        latched_axes = torch.isfinite(self.precontact._latched_contact_q)
        latched_fingers = torch.zeros(HAND_FINGERTIP_COUNT, dtype=torch.bool)
        for finger, columns in enumerate(FINGER_ACTIVE_COLUMNS):
            latched_fingers[finger] = bool(latched_axes[list(columns)].any().item())
        precision_values[latched_fingers.repeat_interleave(3)] = 0.0
        precision = torch.diag(precision_values)
        _, prior_jacobian = _prior_geometry(sample)
        identity = torch.eye(HAND_ACTIVE_DOF, dtype=torch.float64)
        hessian = (
            2.0 * self.cfg.precontact_tracking_weight * identity
            + 2.0
            * self.cfg.fingertip_prior_weight
            * prior_jacobian.T
            @ precision
            @ prior_jacobian
            + 2.0 * self.cfg.hessian_regularization * identity
        )
        gradient = (
            -2.0 * self.cfg.precontact_tracking_weight * baseline.qd_ref
            - 2.0
            * self.cfg.fingertip_prior_weight
            * prior_jacobian.T
            @ precision
            @ mean
        )
        lower = torch.maximum(
            -sample.qd_max, (sample.q_min - sample.q) / self.cfg.dt
        )
        upper = torch.minimum(
            sample.qd_max, (sample.q_max - sample.q) / self.cfg.dt
        )
        if (
            torch.any(lower[latched_axes] > 0.0).item()
            or torch.any(upper[latched_axes] < 0.0).item()
        ):
            return None
        lower[latched_axes] = 0.0
        upper[latched_axes] = 0.0
        problem = DenseQpProblem(
            hessian=hessian,
            gradient=gradient,
            equality_matrix=torch.empty((0, HAND_ACTIVE_DOF), dtype=torch.float64),
            equality_rhs=torch.empty(0, dtype=torch.float64),
            inequality_matrix=torch.empty((0, HAND_ACTIVE_DOF), dtype=torch.float64),
            inequality_upper=torch.empty(0, dtype=torch.float64),
            lower_bound=lower,
            upper_bound=upper,
        )
        try:
            result = solve_reference_qp(
                problem,
                tolerance=self.cfg.qp_tolerance,
                max_iterations=self.cfg.qp_max_iterations,
            )
        except Exception:
            return None
        if not result.success or not torch.isfinite(result.solution).all().item():
            return None
        rate = torch.maximum(lower, torch.minimum(upper, result.solution)).clone()
        # The dense least-squares KKT solve may leave subnormal values on
        # equality-fixed bounds.  Contact-latched public axes are exactly zero.
        rate[latched_axes] = 0.0
        projected = HandMpcSolution(
            q_ref=sample.q + self.cfg.dt * rate,
            qd_ref=rate,
            predicted_forces_b=baseline.predicted_forces_b,
            predicted_wrench_b=baseline.predicted_wrench_b,
            diagnostics=replace(baseline.diagnostics, iterations=result.iterations),
        )
        return projected

    def plan(self, sample: HandMpcInput) -> HandMpcSolution:
        if not isinstance(sample, HandMpcInput):
            raise TypeError("sample must be HandMpcInput")
        if sample.phase in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD}:
            baseline = self._precontact_solution(sample)
            if self.expert_prior is None:
                return self._accept(baseline)
            queried, reason, inference_ms = self._query_prior(sample, baseline)
            if queried is None:
                return self._accept_same_cycle_baseline(
                    sample, baseline, reason or "invalid_prior", inference_ms=inference_ms
                )
            projected = self._solve_precontact_projection(sample, baseline, queried)
            if projected is None:
                return self._accept_same_cycle_baseline(
                    sample,
                    baseline,
                    "prior_qp_rejected",
                    queried=queried,
                    inference_ms=inference_ms,
                )
            diagnostics = self._prior_diagnostics(
                baseline,
                sample,
                queried=queried,
                inference_ms=inference_ms,
                result=projected,
            )
            return self._accept(self._clone(projected, diagnostics))
        baseline_result = solve_reference_qp(
            build_hand_contact_qp(sample, self.cfg),
            tolerance=self.cfg.qp_tolerance,
            max_iterations=self.cfg.qp_max_iterations,
        )
        if not baseline_result.success or not torch.isfinite(baseline_result.solution).all().item():
            return self._fallback(sample, "qp_infeasible")
        baseline = self._solution(sample, baseline_result)
        if self.expert_prior is None:
            return self._accept(baseline)
        queried, reason, inference_ms = self._query_prior(sample, baseline)
        if queried is None:
            return self._accept_same_cycle_baseline(
                sample, baseline, reason or "invalid_prior", inference_ms=inference_ms
            )
        target = queried[0]
        try:
            regularized_result = solve_reference_qp(
                build_hand_contact_qp(sample, self.cfg, prior_target=target),
                tolerance=self.cfg.qp_tolerance,
                max_iterations=self.cfg.qp_max_iterations,
            )
        except Exception:
            regularized_result = None
        if (
            regularized_result is None
            or not regularized_result.success
            or not torch.isfinite(regularized_result.solution).all().item()
        ):
            return self._accept_same_cycle_baseline(
                sample,
                baseline,
                "prior_qp_rejected",
                queried=queried,
                inference_ms=inference_ms,
            )
        regularized = self._solution(sample, regularized_result)
        diagnostics = self._prior_diagnostics(
            baseline,
            sample,
            queried=queried,
            inference_ms=inference_ms,
            result=regularized,
        )
        return self._accept(self._clone(regularized, diagnostics))
