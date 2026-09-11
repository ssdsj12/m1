"""Pure CPU geometry for the right palm orientation planner.

Rotations map palm-local vectors into the M1 base frame. All public tensor
inputs deliberately require CPU float64; no implicit transfer or casting occurs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from go2_pvcnn.control.m1_panda_coordination.qp_backend import DenseQpProblem, solve_reference_qp
from .contracts import BimanualPhase

DTYPE = torch.float64
BOX_HALF_EXTENTS_M = (.10, .09, .05)
HOUSING_SUPPORT_POINTS_LOCAL_M = tuple(
    (x, y, z) for x in (-.0200, .0200)
    for y in (-.0392, .0376) for z in (0., .1128)
)


def _tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != DTYPE or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be finite CPU float64 with shape {shape}')


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    _tensor(rotvec, (3,), 'rotvec')
    x, y, z = rotvec
    zero = torch.zeros((), dtype=DTYPE)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta = float(torch.linalg.vector_norm(rotvec))
    if theta < 1.e-4:
        square = theta * theta
        a = 1. - square / 6. + square * square / 120.
        b = .5 - square / 24. + square * square / 720.
    else:
        a = math.sin(theta) / theta
        b = (1. - math.cos(theta)) / theta**2
    result = torch.eye(3, dtype=DTYPE) + a * skew + b * (skew @ skew)
    _tensor(result, (3, 3), 'rotation result')
    return result


def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    _tensor(matrix, (3, 3), 'matrix')
    if (not torch.allclose(matrix.T @ matrix, torch.eye(3, dtype=DTYPE),
                           atol=1.e-8, rtol=0.)
            or abs(float(torch.linalg.det(matrix)) - 1.) > 1.e-8):
        raise ValueError('matrix must be a proper rotation')
    skew = torch.stack((matrix[2, 1] - matrix[1, 2],
                        matrix[0, 2] - matrix[2, 0],
                        matrix[1, 0] - matrix[0, 1])) * .5
    cosine = max(-1., min(1., float((matrix.trace() - 1.) * .5)))
    sine = float(skew.norm())
    theta = math.atan2(sine, cosine)
    if theta < 1.e-6:
        return skew * (1. + theta**2 / 6.)
    if math.pi - theta < 1.e-6:
        # The symmetric part remains well-conditioned when sin(theta) vanishes.
        outer = (matrix + matrix.T - 2. * cosine * torch.eye(3, dtype=DTYPE))
        outer /= 2. * (1. - cosine)
        index = int(torch.argmax(outer.diag()))
        axis = outer[:, index] / torch.sqrt(outer[index, index].clamp_min(0.))
        axis /= axis.norm()
        if float(axis @ skew) < 0. and sine > 1.e-14:
            axis = -axis
        return axis * theta
    return skew * (theta / sine)


def spatial_orientation_error(target_rotvec_b: torch.Tensor,
                              measured_rotvec_b: torch.Tensor) -> torch.Tensor:
    """Return the base-frame spatial error ``Log(R_target R_measured.T)``."""

    _tensor(target_rotvec_b, (3,), 'target_rotvec_b')
    _tensor(measured_rotvec_b, (3,), 'measured_rotvec_b')
    target = rotvec_to_matrix(target_rotvec_b)
    measured = rotvec_to_matrix(measured_rotvec_b)
    return matrix_to_rotvec(target @ measured.T)


def interpolate_orientation(start_rotvec_b: torch.Tensor,
                            end_rotvec_b: torch.Tensor,
                            fraction: float) -> torch.Tensor:
    """Interpolate two base-frame orientations along the shortest SO(3) path."""

    _tensor(start_rotvec_b, (3,), 'start_rotvec_b')
    _tensor(end_rotvec_b, (3,), 'end_rotvec_b')
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))

            or not math.isfinite(fraction) or not 0. <= fraction <= 1.):
        raise ValueError('fraction must be finite and in [0, 1]')
    start = rotvec_to_matrix(start_rotvec_b)
    end = rotvec_to_matrix(end_rotvec_b)
    relative = matrix_to_rotvec(end @ start.T)
    return matrix_to_rotvec(rotvec_to_matrix(relative * float(fraction)) @ start)


def spatial_angular_velocity(previous_rotvec_b: torch.Tensor,
                             next_rotvec_b: torch.Tensor,
                             dt: float) -> torch.Tensor:
    """Return the geometric base-frame angular rate between adjacent poses."""

    if (isinstance(dt, bool) or not isinstance(dt, (int, float))
            or not math.isfinite(dt) or dt <= 0.):
        raise ValueError('dt must be finite and positive')
    return spatial_orientation_error(next_rotvec_b, previous_rotvec_b) / float(dt)


def compose_orientation_horizon(entry_rotvec_b: torch.Tensor,
                                basis_local: torch.Tensor,
                                coefficients: torch.Tensor) -> torch.Tensor:
    _tensor(entry_rotvec_b, (3,), 'entry_rotvec_b')
    if not isinstance(basis_local, torch.Tensor) or basis_local.ndim != 2:
        raise ValueError('basis_local must have shape (K,3)')
    k = basis_local.shape[0]
    _tensor(basis_local, (k, 3), 'basis_local')
    if k not in (1, 2, 3) or not torch.allclose(
            basis_local @ basis_local.T, torch.eye(k, dtype=DTYPE),
            atol=1.e-10, rtol=0.):
        raise ValueError('basis_local rows must be orthonormal')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.ndim != 2
            or coefficients.shape[0] == 0):
        raise ValueError('coefficients must have nonempty shape (H,K)')
    _tensor(coefficients, (coefficients.shape[0], k), 'coefficients')
    entry = rotvec_to_matrix(entry_rotvec_b)
    return torch.stack([matrix_to_rotvec(entry @ rotvec_to_matrix(offset))
                        for offset in coefficients @ basis_local])


@dataclass(frozen=True)
class PalmLeadGeometry:
    lead_margin_m: float
    digit_support_m: float
    housing_support_m: float
    leading_digit_index: int


@dataclass(frozen=True)
class PalmAxisSelection:
    axis_local: torch.Tensor
    target_angle_rad: float
    geometry: PalmLeadGeometry


def nearest_point_on_oriented_box(palm_position_b: torch.Tensor,
                                  box_pose_b: torch.Tensor,
                                  box_half_extents_m: torch.Tensor) -> torch.Tensor:
    _tensor(palm_position_b, (3,), 'palm_position_b')
    _tensor(box_pose_b, (6,), 'box_pose_b')
    _tensor(box_half_extents_m, (3,), 'box_half_extents_m')
    if not bool((box_half_extents_m > 0.).all()):
        raise ValueError('box half extents must be positive')
    rotation = rotvec_to_matrix(box_pose_b[3:])
    local = rotation.T @ (palm_position_b - box_pose_b[:3])
    return box_pose_b[:3] + rotation @ local.clamp(-box_half_extents_m, box_half_extents_m)


def evaluate_lead_margin(rotation_b: torch.Tensor, ray_b: torch.Tensor,
                         fingertips_local: torch.Tensor,
                         housing_local: torch.Tensor) -> PalmLeadGeometry:
    _tensor(rotation_b, (3, 3), 'rotation_b')
    _tensor(ray_b, (3,), 'ray_b')
    _tensor(fingertips_local, (5, 3), 'fingertips_local')
    _tensor(housing_local, (8, 3), 'housing_local')
    if abs(float(ray_b.norm()) - 1.) > 1.e-10:
        raise ValueError('approach ray must be unit length')
    matrix_to_rotvec(rotation_b)  # Validate the geometric transform.
    local_ray = rotation_b.T @ ray_b
    supports = fingertips_local @ local_ray
    index = int(torch.argmax(supports))
    digit = float(supports[index])
    housing = float(torch.max(housing_local @ local_ray))
    return PalmLeadGeometry(digit - housing, digit, housing, index)


def select_single_axis(entry_rotation_b: torch.Tensor, ray_b: torch.Tensor,
                        fingertips_local: torch.Tensor, housing_local: torch.Tensor,
                        *, axis_local: torch.Tensor | None = None,
                        theta_max_rad: float = .35,
                        candidate_count: int = 29) -> PalmAxisSelection:
    if (not math.isfinite(theta_max_rad) or theta_max_rad <= 0.
            or type(candidate_count) is not int or candidate_count < 3
            or candidate_count % 2 != 1):
        raise ValueError('candidate grid must be finite, symmetric and contain zero')
    if axis_local is None:
        axes = torch.eye(3, dtype=DTYPE)
    else:
        _tensor(axis_local, (3,), 'axis_local')
        if abs(float(axis_local.norm()) - 1.) > 1.e-10:
            raise ValueError('axis_local must have unit length')
        axes = axis_local.reshape(1, 3)
    angles = torch.linspace(-theta_max_rad, theta_max_rad, candidate_count, dtype=DTYPE)
    angles[candidate_count // 2] = 0.
    candidates = []
    for index, axis in enumerate(axes):
        for value in angles:
            angle = float(value)
            geometry = evaluate_lead_margin(
                entry_rotation_b @ rotvec_to_matrix(axis * angle), ray_b,
                fingertips_local, housing_local)
            rank = (-geometry.lead_margin_m, abs(angle), index, angle)
            candidates.append((rank, PalmAxisSelection(axis.clone(), angle, geometry)))
    return min(candidates, key=lambda item: item[0])[1]


@dataclass(frozen=True)
class PalmOrientationMpcCfg:
    dt: float = .04
    horizon_steps: int = 25
    theta_max_rad: float = .35
    angular_rate_max_rad_s: float = .35
    candidate_count: int = 29
    active_basis_dim: int = 1
    tracking_weight: float = 20.
    slew_weight: float = 2.
    smoothness_weight: float = 5.
    regularization: float = 1.e-8
    qp_tolerance: float = 1.e-8
    qp_max_iterations: int = 256

    def __post_init__(self) -> None:
        for name in ('dt', 'theta_max_rad', 'angular_rate_max_rad_s',
                     'tracking_weight', 'slew_weight', 'smoothness_weight',
                     'regularization', 'qp_tolerance'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0.):
                raise ValueError(f'{name} must be finite and positive')
        for name in ('horizon_steps', 'candidate_count', 'active_basis_dim', 'qp_max_iterations'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.horizon_steps != 25 or self.active_basis_dim != 1:
            raise ValueError('runtime requires a 25-node scalar orientation QP')
        if self.candidate_count < 29 or self.candidate_count % 2 != 1:
            raise ValueError('candidate grid requires at least 29 odd samples')


def build_palm_orientation_qp(current_angle: float, target_angle: float,
                              cfg: PalmOrientationMpcCfg) -> DenseQpProblem:
    if any(not math.isfinite(value) or abs(value) > cfg.theta_max_rad
           for value in (current_angle, target_angle)):
        raise ValueError('angles must be finite and within the calibrated interval')
    n = cfg.horizon_steps
    eye = torch.eye(n, dtype=DTYPE)
    d1 = eye - torch.diag(torch.ones(n - 1, dtype=DTYPE), diagonal=-1)
    b = torch.zeros(n, dtype=DTYPE)
    b[0] = current_angle
    d2 = d1[1:] - d1[:-1]
    boundary2 = b[1:] - b[:-1]
    hessian = 2. * (cfg.tracking_weight * eye + cfg.slew_weight * d1.T @ d1
                    + cfg.smoothness_weight * d2.T @ d2 + cfg.regularization * eye)
    gradient = (-2. * cfg.tracking_weight * target_angle * torch.ones(n, dtype=DTYPE)
                - 2. * cfg.slew_weight * d1.T @ b
                - 2. * cfg.smoothness_weight * d2.T @ boundary2)
    delta = torch.full((n,), cfg.angular_rate_max_rad_s * cfg.dt, dtype=DTYPE)
    return DenseQpProblem(
        hessian=hessian, gradient=gradient,
        equality_matrix=torch.zeros(0, n, dtype=DTYPE), equality_rhs=torch.zeros(0, dtype=DTYPE),
        inequality_matrix=torch.cat((d1, -d1)), inequality_upper=torch.cat((delta + b, delta - b)),
        lower_bound=torch.full((n,), -cfg.theta_max_rad, dtype=DTYPE),
        upper_bound=torch.full((n,), cfg.theta_max_rad, dtype=DTYPE))


@dataclass(frozen=True)
class PalmOrientationInput:
    palm_pose_b: torch.Tensor
    fingertip_positions_b: torch.Tensor
    box_pose_b: torch.Tensor
    contact_mask: torch.Tensor
    phase: BimanualPhase


@dataclass(frozen=True)
class PalmOrientationDiagnostics:
    feasible: bool
    fallback_reason: str | None = None
    basis_dimension: int = 1
    selected_axis_local: tuple[float, float, float] | None = None
    candidate_angle_rad: float = 0.
    committed_angle_rad: float = 0.
    lead_margin_m: float | None = None
    leading_digit_index: int | None = None
    housing_support_m: float | None = None
    digit_support_m: float | None = None
    iterations: int = 0
    contact_latched: bool = False


@dataclass(frozen=True)
class PalmOrientationSolution:
    orientation_rotvec_b: torch.Tensor
    angle_rad: torch.Tensor
    diagnostics: PalmOrientationDiagnostics


class RightPalmOrientationMpc:
    def __init__(self, cfg: PalmOrientationMpcCfg | None = None):
        self.cfg = cfg or PalmOrientationMpcCfg()
        self.reset()

    def reset(self) -> None:
        self._entry_rotation = None
        self._axis_local = None
        self._committed_angle = 0.
        self._contact_latched = False
        self._last_safe = None
        self.last_diagnostics = None

    @staticmethod
    def _clone(solution: PalmOrientationSolution) -> PalmOrientationSolution:
        return replace(solution, orientation_rotvec_b=solution.orientation_rotvec_b.clone(),
                       angle_rad=solution.angle_rad.clone())

    def _fallback(self, sample: PalmOrientationInput, reason: str) -> PalmOrientationSolution:
        if self._last_safe is not None:
            solution = self._clone(self._last_safe)
            solution = replace(solution, diagnostics=replace(solution.diagnostics,
                                                              feasible=False, fallback_reason=reason))
        else:
            # No finite hold can be inferred from a corrupt initial palm pose.
            _tensor(sample.palm_pose_b, (6,), 'initial fallback palm pose')
            solution = PalmOrientationSolution(sample.palm_pose_b[3:].repeat(25, 1),
                torch.zeros(25, dtype=DTYPE), PalmOrientationDiagnostics(False, reason))
        self.last_diagnostics = solution.diagnostics
        return solution

    def plan(self, sample: PalmOrientationInput) -> PalmOrientationSolution:
        try:
            _tensor(sample.palm_pose_b, (6,), 'palm_pose_b')
            _tensor(sample.fingertip_positions_b, (5, 3), 'fingertip_positions_b')
            _tensor(sample.box_pose_b, (6,), 'box_pose_b')
            mask = sample.contact_mask
            if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
                    or mask.dtype != torch.bool or tuple(mask.shape) != (5,)
                    or not isinstance(sample.phase, BimanualPhase)):
                raise ValueError('invalid contact mask or phase')
            measured = rotvec_to_matrix(sample.palm_pose_b[3:])
            entry = measured if self._entry_rotation is None else self._entry_rotation
            point = nearest_point_on_oriented_box(sample.palm_pose_b[:3], sample.box_pose_b,
                                                  torch.tensor(BOX_HALF_EXTENTS_M, dtype=DTYPE))
            ray = point - sample.palm_pose_b[:3]
            if float(ray.norm()) <= 1.e-10:
                raise ValueError('degenerate approach ray')
            ray = ray / ray.norm()
            tips = (sample.fingertip_positions_b - sample.palm_pose_b[:3]) @ measured
            housing = torch.tensor(HOUSING_SUPPORT_POINTS_LOCAL_M, dtype=DTYPE)
            selection = select_single_axis(entry, ray, tips, housing,
                axis_local=self._axis_local, theta_max_rad=self.cfg.theta_max_rad,
                candidate_count=self.cfg.candidate_count)
            axis = selection.axis_local
            latched = self._contact_latched
            current = self._committed_angle
            iterations = 0
            if bool(mask.any()) and not latched:
                current = float(matrix_to_rotvec(entry.T @ measured) @ axis)
                if abs(current) > self.cfg.theta_max_rad:
                    raise ValueError('contact angle outside calibrated interval')
                latched = True
            if latched or sample.phase not in (BimanualPhase.APPROACH, BimanualPhase.PRELOAD):
                angles = torch.full((25,), current, dtype=DTYPE)
            else:
                qp = build_palm_orientation_qp(current, selection.target_angle_rad, self.cfg)
                result = solve_reference_qp(qp, tolerance=self.cfg.qp_tolerance,
                                            max_iterations=self.cfg.qp_max_iterations)
                if not result.success:
                    return self._fallback(sample, 'qp_infeasible')
                angles = result.solution.clone()
                iterations = result.iterations
                _tensor(angles, (25,), 'QP angles')
            orientations = compose_orientation_horizon(matrix_to_rotvec(entry),
                                                        axis.reshape(1, 3), angles[:, None])
            geom = selection.geometry
            diagnostics = PalmOrientationDiagnostics(
                True, selected_axis_local=tuple(float(x) for x in axis),
                candidate_angle_rad=selection.target_angle_rad, committed_angle_rad=float(angles[0]),
                lead_margin_m=geom.lead_margin_m, leading_digit_index=geom.leading_digit_index,
                housing_support_m=geom.housing_support_m, digit_support_m=geom.digit_support_m,
                iterations=iterations, contact_latched=latched)
            solution = PalmOrientationSolution(orientations, angles, diagnostics)
        except (ValueError, TypeError, RuntimeError) as error:
            return self._fallback(sample, f'invalid_orientation_proposal: {error}')
        self._entry_rotation = entry.clone()
        self._axis_local = axis.clone()
        self._committed_angle = float(angles[0])
        self._contact_latched = latched
        self._last_safe = self._clone(solution)
        self.last_diagnostics = diagnostics
        return solution
