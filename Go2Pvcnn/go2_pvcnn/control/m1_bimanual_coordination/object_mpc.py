"""Condensed rigid-body MPC for the shared box and bimanual grasp targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy

from .palm_orientation_mpc import PalmOrientationInput, RightPalmOrientationMpc
import math

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    DenseQpResult,
    solve_reference_qp,
)

from .contracts import BimanualPhase, BimanualSnapshot


OBJECT_MPC_DT = 0.04
OBJECT_MPC_HORIZON_STEPS = 25
_WRENCH_DOF = 6
_VARIABLES_PER_STEP = 13
_LEFT = slice(0, 6)
_RIGHT = slice(6, 12)
_YAW = 12
_GRAVITY = 9.81
_GRASP_PHASES = frozenset(
    {
        BimanualPhase.GRASP,
        BimanualPhase.LIFT,
        BimanualPhase.HOLD,
        BimanualPhase.LOWER,
        BimanualPhase.HOLD_SAFE,
        BimanualPhase.LOWER_SAFE,
    }
)


def _real_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _exact_cpu64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on device cpu; got {value.device}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value.clone()


@dataclass(frozen=True)
class ObjectMpcCfg:
    dt: float = OBJECT_MPC_DT
    horizon_steps: int = OBJECT_MPC_HORIZON_STEPS
    box_pose_weight: float = 2000.0
    box_twist_weight: float = 100.0
    wrench_slew_weight: float = 0.1
    wrench_nominal_weight: float = 0.02
    platform_motion_weight: float = 10.0
    hessian_regularization: float = 1.0e-8
    friction_coefficient: float = 0.8
    per_hand_normal_force_min: float = 8.0
    per_hand_normal_force_max: float = 15.0
    preload_normal_force: float = 2.0
    grasp_half_width_m: float = 0.09
    palm_lateral_offset_m: float = 0.20
    palm_reach_offset_m: float = 0.20
    preload_palm_inward_speed_m_s: float = 0.025
    left_preload_inward_limit_m: float = 0.04
    right_preload_inward_limit_m: float = 0.115
    preload_forward_limit_m: float = 0.025
    right_palm_height_offset_m: float = 0.0
    right_preload_height_offset_m: float = 0.045
    platform_yaw_limit_rad: float = math.pi / 2.0
    platform_velocity_limit_rad_s: float = 0.25
    max_target_translation_m: float = 0.5
    max_target_orientation_rad: float = 1.2
    moment_limit_nm: float = 3.0
    qp_tolerance: float = 1.0e-7
    qp_max_iterations: int = 512

    def __post_init__(self) -> None:
        _real_positive("dt", self.dt)
        if self.horizon_steps != OBJECT_MPC_HORIZON_STEPS:
            raise ValueError(
                f"horizon_steps must equal the frozen value {OBJECT_MPC_HORIZON_STEPS}"
            )
        for name in (
            "box_pose_weight",
            "box_twist_weight",
            "wrench_slew_weight",
            "wrench_nominal_weight",
            "platform_motion_weight",
            "hessian_regularization",
            "friction_coefficient",
            "per_hand_normal_force_min",
            "per_hand_normal_force_max",
            "preload_normal_force",
            "grasp_half_width_m",
            "palm_lateral_offset_m",
            "palm_reach_offset_m",
            "preload_palm_inward_speed_m_s",
            "left_preload_inward_limit_m",
            "right_preload_inward_limit_m",
            "preload_forward_limit_m",
            "right_preload_height_offset_m",
            "platform_yaw_limit_rad",
            "platform_velocity_limit_rad_s",
            "max_target_translation_m",
            "max_target_orientation_rad",
            "moment_limit_nm",
            "qp_tolerance",
        ):
            _real_positive(name, getattr(self, name))
        if (
            isinstance(self.right_palm_height_offset_m, bool)
            or not isinstance(self.right_palm_height_offset_m, (int, float))
            or not math.isfinite(float(self.right_palm_height_offset_m))
            or float(self.right_palm_height_offset_m) < 0.0
        ):
            raise ValueError("right_palm_height_offset_m must be finite and non-negative")
        if self.per_hand_normal_force_min >= self.per_hand_normal_force_max:
            raise ValueError("per-hand normal force minimum must be below maximum")
        if self.preload_normal_force >= self.per_hand_normal_force_min:
            raise ValueError("preload_normal_force must be below grasp minimum")
        if isinstance(self.qp_max_iterations, bool) or not isinstance(
            self.qp_max_iterations, int
        ) or self.qp_max_iterations <= 0:
            raise ValueError("qp_max_iterations must be a positive integer")

    @property
    def horizon_seconds(self) -> float:
        return float(self.dt) * self.horizon_steps


@dataclass(frozen=True)
class ObjectMpcDiagnostics:
    feasible: bool
    fallback_used: bool
    fallback_reason: str | None
    force_closure_margin: float
    saturation_fraction: float
    iterations: int


@dataclass(frozen=True)
class ObjectMpcSolution:
    box_pose: torch.Tensor
    box_twist: torch.Tensor
    platform_yaw: torch.Tensor
    left_palm_pose: torch.Tensor
    right_palm_pose: torch.Tensor
    left_wrench: torch.Tensor
    right_wrench: torch.Tensor
    diagnostics: ObjectMpcDiagnostics

    def __post_init__(self) -> None:
        horizon = OBJECT_MPC_HORIZON_STEPS
        for name, shape in (
            ("box_pose", (horizon, 6)),
            ("box_twist", (horizon, 6)),
            ("platform_yaw", (horizon,)),
            ("left_palm_pose", (horizon, 6)),
            ("right_palm_pose", (horizon, 6)),
            ("left_wrench", (horizon, 6)),
            ("right_wrench", (horizon, 6)),
        ):
            object.__setattr__(self, name, _exact_cpu64(name, getattr(self, name), shape))
        if not isinstance(self.diagnostics, ObjectMpcDiagnostics):
            raise TypeError("diagnostics must be ObjectMpcDiagnostics")


@dataclass(frozen=True)
class ObjectMpcInput:
    snapshot: BimanualSnapshot
    target_box_pose_b: torch.Tensor | None = None
    phase: BimanualPhase = BimanualPhase.APPROACH
    previous_solution: ObjectMpcSolution | None = None
    target_object_pose_b: torch.Tensor | None = None
    obstacle_object_poses_b: tuple[torch.Tensor, ...] | torch.Tensor = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if not isinstance(self.phase, BimanualPhase):
            raise TypeError("phase must be BimanualPhase")
        if self.previous_solution is not None and not isinstance(
            self.previous_solution, ObjectMpcSolution
        ):
            raise TypeError("previous_solution must be ObjectMpcSolution or None")
        target_box_pose = self.target_box_pose_b
        target_object_pose = self.target_object_pose_b
        if target_box_pose is None and target_object_pose is None:
            raise ValueError(
                "one of target_box_pose_b or target_object_pose_b is required"
            )
        if target_box_pose is None:
            target_box_pose = target_object_pose
        elif target_object_pose is not None:
            target_object_pose = _exact_cpu64(
                "target_object_pose_b",
                target_object_pose,
                (OBJECT_MPC_HORIZON_STEPS, 6),
            )
            if not torch.equal(
                _exact_cpu64(
                    "target_box_pose_b",
                    target_box_pose,
                    (OBJECT_MPC_HORIZON_STEPS, 6),
                ),
                target_object_pose,
            ):
                raise ValueError(
                    "target_box_pose_b and target_object_pose_b must match"
                )
        target_box_pose = _exact_cpu64(
            "target_box_pose_b",
            target_box_pose,
            (OBJECT_MPC_HORIZON_STEPS, 6),
        )
        object.__setattr__(self, "target_box_pose_b", target_box_pose)
        object.__setattr__(self, "target_object_pose_b", target_box_pose.clone())

        obstacle_poses = self.obstacle_object_poses_b
        if isinstance(obstacle_poses, torch.Tensor):
            if obstacle_poses.ndim != 3 or obstacle_poses.shape[1:] != (
                OBJECT_MPC_HORIZON_STEPS,
                6,
            ):
                raise ValueError(
                    "obstacle_object_poses_b tensor must have shape "
                    "(N, horizon, 6)"
                )
            obstacle_poses = tuple(obstacle_poses[index] for index in range(obstacle_poses.shape[0]))
        elif isinstance(obstacle_poses, (tuple, list)):
            obstacle_poses = tuple(obstacle_poses)
        else:
            raise TypeError(
                "obstacle_object_poses_b must be a tensor or sequence of tensors"
            )
        object.__setattr__(
            self,
            "obstacle_object_poses_b",
            tuple(
                _exact_cpu64(
                    f"obstacle_object_poses_b[{index}]",
                    value,
                    (OBJECT_MPC_HORIZON_STEPS, 6),
                )
                for index, value in enumerate(obstacle_poses)
            ),
        )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.tolist()
    return torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64
    )


def _acceleration_map(sample: ObjectMpcInput, cfg: ObjectMpcCfg) -> torch.Tensor:
    mass = float(sample.snapshot.box.mass.item())
    inverse_inertia = torch.linalg.inv(sample.snapshot.box.inertia_b)
    result = torch.zeros((6, _VARIABLES_PER_STEP), dtype=torch.float64)
    result[:3, 0:3] = torch.eye(3, dtype=torch.float64) / mass
    result[:3, 6:9] = torch.eye(3, dtype=torch.float64) / mass
    left_offset = torch.tensor([0.0, cfg.grasp_half_width_m, 0.0], dtype=torch.float64)
    right_offset = -left_offset
    result[3:, 0:3] = inverse_inertia @ _skew(left_offset)
    result[3:, 3:6] = inverse_inertia
    result[3:, 6:9] = inverse_inertia @ _skew(right_offset)
    result[3:, 9:12] = inverse_inertia
    return result


def _desired_acceleration(sample: ObjectMpcInput, cfg: ObjectMpcCfg) -> torch.Tensor:
    pose = sample.snapshot.box.pose_b.clone()
    twist = sample.snapshot.box.twist_b.clone()
    accelerations = []
    for target in sample.target_box_pose_b:
        acceleration = 2.0 * (target - pose - cfg.dt * twist) / (cfg.dt**2)
        accelerations.append(acceleration)
        twist = twist + cfg.dt * acceleration
        pose = pose + cfg.dt * (twist - cfg.dt * acceleration) + 0.5 * cfg.dt**2 * acceleration
    return torch.stack(accelerations)


def _condensed_dynamics(
    sample: ObjectMpcInput, cfg: ObjectMpcCfg, acceleration_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = cfg.horizon_steps
    dimension = horizon * _VARIABLES_PER_STEP
    pose_map = torch.zeros((horizon * 6, dimension), dtype=torch.float64)
    twist_map = torch.zeros_like(pose_map)
    gravity = torch.tensor([0.0, 0.0, -_GRAVITY, 0.0, 0.0, 0.0], dtype=torch.float64)
    if sample.phase not in _GRASP_PHASES and sample.snapshot.box.supported:
        gravity.zero_()
    pose_offset_rows = []
    twist_offset_rows = []
    for step in range(horizon):
        elapsed = (step + 1) * cfg.dt
        pose_offset_rows.append(
            sample.snapshot.box.pose_b
            + elapsed * sample.snapshot.box.twist_b
            + 0.5 * elapsed**2 * gravity
        )
        twist_offset_rows.append(sample.snapshot.box.twist_b + elapsed * gravity)
        row = slice(step * 6, (step + 1) * 6)
        for control in range(step + 1):
            column = slice(
                control * _VARIABLES_PER_STEP,
                (control + 1) * _VARIABLES_PER_STEP,
            )
            pose_map[row, column] = (
                cfg.dt**2 * (step - control + 0.5) * acceleration_map
            )
            twist_map[row, column] = cfg.dt * acceleration_map
    return (
        pose_map,
        torch.cat(pose_offset_rows),
        twist_map,
        torch.cat(twist_offset_rows),
    )


def _quadratic(
    matrix: torch.Tensor, offset: torch.Tensor, weight: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        2.0 * weight * matrix.T @ matrix,
        2.0 * weight * matrix.T @ offset,
    )


def _normal_limits(phase: BimanualPhase, cfg: ObjectMpcCfg) -> tuple[float, float]:
    if phase in _GRASP_PHASES:
        return cfg.per_hand_normal_force_min, cfg.per_hand_normal_force_max
    if phase is BimanualPhase.PRELOAD:
        return cfg.preload_normal_force, cfg.preload_normal_force
    return 0.0, 0.0


def build_object_qp(sample: ObjectMpcInput, cfg: ObjectMpcCfg) -> DenseQpProblem:
    """Build the horizon QP with condensed rigid-body and contact constraints."""

    if not isinstance(sample, ObjectMpcInput):
        raise TypeError("sample must be ObjectMpcInput")
    if not isinstance(cfg, ObjectMpcCfg):
        raise TypeError("cfg must be ObjectMpcCfg")
    horizon = cfg.horizon_steps
    dimension = horizon * _VARIABLES_PER_STEP
    acceleration_map = _acceleration_map(sample, cfg)
    pose_map, pose_offset, twist_map, twist_offset = _condensed_dynamics(
        sample, cfg, acceleration_map
    )
    hessian, gradient = _quadratic(
        pose_map,
        pose_offset - sample.target_box_pose_b.reshape(-1),
        cfg.box_pose_weight,
    )
    target_twist = torch.cat(
        (
            (sample.target_box_pose_b[0] - sample.snapshot.box.pose_b)[None, :],
            sample.target_box_pose_b[1:] - sample.target_box_pose_b[:-1],
        ),
        dim=0,
    ) / cfg.dt
    term_h, term_g = _quadratic(
        twist_map,
        twist_offset - target_twist.reshape(-1),
        cfg.box_twist_weight,
    )
    hessian += term_h
    gradient += term_g

    desired_acceleration = _desired_acceleration(sample, cfg)
    nominal = torch.zeros((horizon, _VARIABLES_PER_STEP), dtype=torch.float64)
    normal_min, normal_max = _normal_limits(sample.phase, cfg)
    nominal_normal = 0.5 * (normal_min + normal_max)
    nominal[:, 1] = -nominal_normal
    nominal[:, 7] = nominal_normal
    gravity_compensation = _GRAVITY if sample.phase in _GRASP_PHASES else 0.0
    nominal_vertical = 0.5 * float(sample.snapshot.box.mass) * (
        desired_acceleration[:, 2] + gravity_compensation
    )
    nominal[:, 2] = nominal_vertical
    nominal[:, 8] = nominal_vertical
    nominal[:, _YAW] = sample.snapshot.platform_q_qd[0]
    identity = torch.eye(dimension, dtype=torch.float64)
    hessian += 2.0 * cfg.wrench_nominal_weight * identity
    gradient += -2.0 * cfg.wrench_nominal_weight * nominal.reshape(-1)

    slew = torch.zeros((horizon * 12, dimension), dtype=torch.float64)
    for step in range(horizon):
        rows = slice(step * 12, (step + 1) * 12)
        current = slice(step * _VARIABLES_PER_STEP, step * _VARIABLES_PER_STEP + 12)
        slew[rows, current] = torch.eye(12, dtype=torch.float64)
        if step:
            previous = slice(
                (step - 1) * _VARIABLES_PER_STEP,
                (step - 1) * _VARIABLES_PER_STEP + 12,
            )
            slew[rows, previous] = -torch.eye(12, dtype=torch.float64)
    hessian += 2.0 * cfg.wrench_slew_weight * slew.T @ slew
    yaw_selector = torch.zeros((horizon, dimension), dtype=torch.float64)
    for step in range(horizon):
        yaw_selector[step, step * _VARIABLES_PER_STEP + _YAW] = 1.0
    yaw_offset = -sample.snapshot.platform_q_qd[0].repeat(horizon)
    term_h, term_g = _quadratic(
        yaw_selector, yaw_offset, cfg.platform_motion_weight
    )
    hessian += term_h + 2.0 * cfg.hessian_regularization * identity
    gradient += term_g

    gravity = torch.tensor([0.0, 0.0, -_GRAVITY, 0.0, 0.0, 0.0], dtype=torch.float64)
    if sample.phase not in _GRASP_PHASES and sample.snapshot.box.supported:
        gravity.zero_()
    equality_matrix = torch.zeros((horizon * 6, dimension), dtype=torch.float64)
    for step in range(horizon):
        rows = slice(step * 6, (step + 1) * 6)
        columns = slice(step * _VARIABLES_PER_STEP, (step + 1) * _VARIABLES_PER_STEP)
        equality_matrix[rows, columns] = acceleration_map
    equality_rhs = (desired_acceleration - gravity).reshape(-1)

    inequalities = []
    inequality_upper = []
    mu = cfg.friction_coefficient
    yaw_step = cfg.platform_velocity_limit_rad_s * cfg.dt
    for step in range(horizon):
        offset = step * _VARIABLES_PER_STEP
        for tangent in (0, 2):
            for sign in (-1.0, 1.0):
                left = torch.zeros(dimension, dtype=torch.float64)
                left[offset + tangent] = sign
                left[offset + 1] = mu
                inequalities.append(left)
                inequality_upper.append(0.0)
                right = torch.zeros(dimension, dtype=torch.float64)
                right[offset + 6 + tangent] = sign
                right[offset + 7] = -mu
                inequalities.append(right)
                inequality_upper.append(0.0)
        for sign in (-1.0, 1.0):
            row = torch.zeros(dimension, dtype=torch.float64)
            row[offset + _YAW] = sign
            if step:
                row[offset - _VARIABLES_PER_STEP + _YAW] = -sign
                upper = yaw_step
            else:
                upper = yaw_step + sign * float(sample.snapshot.platform_q_qd[0])
            inequalities.append(row)
            inequality_upper.append(upper)

    lower = torch.full((dimension,), -torch.inf, dtype=torch.float64)
    upper = torch.full((dimension,), torch.inf, dtype=torch.float64)
    force_bound = max(cfg.per_hand_normal_force_max, cfg.friction_coefficient * cfg.per_hand_normal_force_max)
    for step in range(horizon):
        offset = step * _VARIABLES_PER_STEP
        for start in (0, 6):
            lower[offset + start : offset + start + 3] = -force_bound
            upper[offset + start : offset + start + 3] = force_bound
            lower[offset + start + 3 : offset + start + 6] = -cfg.moment_limit_nm
            upper[offset + start + 3 : offset + start + 6] = cfg.moment_limit_nm
        lower[offset + 1], upper[offset + 1] = -normal_max, -normal_min
        lower[offset + 7], upper[offset + 7] = normal_min, normal_max
        lower[offset + _YAW] = -cfg.platform_yaw_limit_rad
        upper[offset + _YAW] = cfg.platform_yaw_limit_rad
    return DenseQpProblem(
        hessian=hessian,
        gradient=gradient,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=torch.stack(inequalities),
        inequality_upper=torch.tensor(inequality_upper, dtype=torch.float64),
        lower_bound=lower,
        upper_bound=upper,
    )


def _palm_targets(
    box_pose: torch.Tensor,
    half_width: float,
    reach_offset: float,
    left_orientation: torch.Tensor,
    right_orientation: torch.Tensor,
    right_height_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = box_pose.clone()
    right = box_pose.clone()
    # Palm origins sit behind the O6 fingertips; contact targets are defined
    # on the box while arm MPC must track the corresponding palm origins.
    left[:, 0] -= reach_offset
    right[:, 0] -= reach_offset
    left[:, 1] += half_width
    right[:, 1] -= half_width
    right[:, 2] += right_height_offset
    # The O6 mounting transform is not the box frame.  For this fixed first
    # task, retain each calibrated mounted-hand orientation while translating
    # the palms to the two object faces.
    left[:, 3:] = left_orientation
    right[:, 3:] = right_orientation
    return left, right


class BimanualObjectMpc:
    """Stateful 25 Hz object MPC with atomic last-safe fallback."""

    def __init__(self, cfg: ObjectMpcCfg | None = None,
                 right_orientation_mpc: RightPalmOrientationMpc | None = None) -> None:
        self.cfg = ObjectMpcCfg() if cfg is None else cfg
        self.right_orientation_mpc = right_orientation_mpc if right_orientation_mpc is not None else RightPalmOrientationMpc()
        self._preload_palm_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        self._preload_inward_travel = [0.0, 0.0]
        self._right_preload_forward_travel = 0.0
        self._preload_contact_latched = [False, False]
        if not isinstance(self.cfg, ObjectMpcCfg):
            raise TypeError("cfg must be ObjectMpcCfg")
        self._last_safe: ObjectMpcSolution | None = None

    def _target_reachable(self, sample: ObjectMpcInput) -> bool:
        delta = sample.target_box_pose_b - sample.snapshot.box.pose_b
        translation = torch.linalg.vector_norm(delta[:, :3], dim=-1)
        orientation = torch.linalg.vector_norm(delta[:, 3:], dim=-1)
        return bool(
            (translation <= self.cfg.max_target_translation_m).all()
            and (orientation <= self.cfg.max_target_orientation_rad).all()
        )

    @staticmethod
    def _clone(solution: ObjectMpcSolution, diagnostics: ObjectMpcDiagnostics | None = None) -> ObjectMpcSolution:
        return ObjectMpcSolution(
            box_pose=solution.box_pose,
            box_twist=solution.box_twist,
            platform_yaw=solution.platform_yaw,
            left_palm_pose=solution.left_palm_pose,
            right_palm_pose=solution.right_palm_pose,
            left_wrench=solution.left_wrench,
            right_wrench=solution.right_wrench,
            diagnostics=solution.diagnostics if diagnostics is None else diagnostics,
        )

    def _fallback(self, sample: ObjectMpcInput, reason: str) -> ObjectMpcSolution:
        safe = sample.previous_solution or self._last_safe
        diagnostics = ObjectMpcDiagnostics(
            feasible=False,
            fallback_used=True,
            fallback_reason=reason,
            force_closure_margin=(
                safe.diagnostics.force_closure_margin if safe is not None else 0.0
            ),
            saturation_fraction=0.0,
            iterations=0,
        )
        if safe is not None:
            return self._clone(safe, diagnostics)
        horizon = self.cfg.horizon_steps
        box_pose = sample.snapshot.box.pose_b.repeat(horizon, 1)
        left_palm, right_palm = _palm_targets(
            box_pose,
            self.cfg.palm_lateral_offset_m,
            self.cfg.palm_reach_offset_m,
            sample.snapshot.left_arm.palm_pose_b[3:],
            sample.snapshot.right_arm.palm_pose_b[3:],
            self.cfg.right_palm_height_offset_m,
        )
        return ObjectMpcSolution(
            box_pose=box_pose,
            box_twist=torch.zeros((horizon, 6), dtype=torch.float64),
            platform_yaw=sample.snapshot.platform_q_qd[0].repeat(horizon),
            left_palm_pose=left_palm,
            right_palm_pose=right_palm,
            left_wrench=torch.zeros((horizon, 6), dtype=torch.float64),
            right_wrench=torch.zeros((horizon, 6), dtype=torch.float64),
            diagnostics=diagnostics,
        )

    def _solution(
        self, sample: ObjectMpcInput, result: DenseQpResult
    ) -> ObjectMpcSolution:
        controls = result.solution.reshape(self.cfg.horizon_steps, _VARIABLES_PER_STEP)
        acceleration_map = _acceleration_map(sample, self.cfg)
        pose_map, pose_offset, twist_map, twist_offset = _condensed_dynamics(
            sample, self.cfg, acceleration_map
        )
        box_pose = (pose_offset + pose_map @ result.solution).reshape(-1, 6)
        box_twist = (twist_offset + twist_map @ result.solution).reshape(-1, 6)
        left_wrench = controls[:, _LEFT]
        right_wrench = controls[:, _RIGHT]
        left_normal = -left_wrench[:, 1]
        right_normal = right_wrench[:, 1]
        margins = torch.cat(
            (
                self.cfg.friction_coefficient * left_normal - left_wrench[:, 0].abs(),
                self.cfg.friction_coefficient * left_normal - left_wrench[:, 2].abs(),
                self.cfg.friction_coefficient * right_normal - right_wrench[:, 0].abs(),
                self.cfg.friction_coefficient * right_normal - right_wrench[:, 2].abs(),
            )
        )
        normal_min, normal_max = _normal_limits(sample.phase, self.cfg)
        saturated = torch.cat(
            (
                (left_normal - normal_min).abs() <= 1.0e-6,
                (left_normal - normal_max).abs() <= 1.0e-6,
                (right_normal - normal_min).abs() <= 1.0e-6,
                (right_normal - normal_max).abs() <= 1.0e-6,
            )
        )
        left_palm, right_palm = _palm_targets(
            box_pose,
            self.cfg.palm_lateral_offset_m,
            self.cfg.palm_reach_offset_m,
            sample.snapshot.left_arm.palm_pose_b[3:],
            sample.snapshot.right_arm.palm_pose_b[3:],
            self.cfg.right_palm_height_offset_m,
        )
        return ObjectMpcSolution(
            box_pose=box_pose,
            box_twist=box_twist,
            platform_yaw=controls[:, _YAW],
            left_palm_pose=left_palm,
            right_palm_pose=right_palm,
            left_wrench=left_wrench,
            right_wrench=right_wrench,
            diagnostics=ObjectMpcDiagnostics(
                feasible=True,
                fallback_used=False,
                fallback_reason=None,
                force_closure_margin=float(margins.min().item()),
                saturation_fraction=float(saturated.double().mean().item()),
                iterations=result.iterations,
            ),
        )

    def reset(self) -> None:
        self._preload_palm_targets = None
        self._preload_inward_travel = [0.0, 0.0]
        self._right_preload_forward_travel = 0.0
        self._preload_contact_latched = [False, False]
        self._last_safe = None
        self.right_orientation_mpc.reset()

    def plan(self, sample: ObjectMpcInput) -> ObjectMpcSolution:
        # Translation and orientation form one object proposal. Restore the
        # translation state if the subordinate orientation proposal is rejected.
        names = ('_preload_palm_targets', '_preload_inward_travel',
                 '_right_preload_forward_travel', '_preload_contact_latched', '_last_safe')
        saved = {name: copy.deepcopy(getattr(self, name)) for name in names}
        candidate = self._plan_translation(sample)
        if not candidate.diagnostics.feasible:
            for name, value in saved.items():
                setattr(self, name, value)
            return candidate
        snapshot = sample.snapshot
        orientation = self.right_orientation_mpc.plan(PalmOrientationInput(
            snapshot.right_arm.palm_pose_b, snapshot.right_hand.fingertip_positions_b,
            snapshot.box.pose_b, snapshot.right_hand.contact_mask, sample.phase))
        if not orientation.diagnostics.feasible:
            for name, value in saved.items():
                setattr(self, name, value)
            return self._fallback(sample, 'right_palm_orientation_infeasible')
        right_pose = candidate.right_palm_pose.clone()
        right_pose[:, 3:] = orientation.orientation_rotvec_b
        candidate = replace(candidate, right_palm_pose=right_pose)
        self._last_safe = self._clone(candidate)
        return candidate

    def _plan_translation(self, sample: ObjectMpcInput) -> ObjectMpcSolution:
        if not isinstance(sample, ObjectMpcInput):
            raise TypeError("sample must be ObjectMpcInput")
        if sample.phase is BimanualPhase.APPROACH:
            self._preload_palm_targets = None
            self._preload_inward_travel = [0.0, 0.0]
            self._right_preload_forward_travel = 0.0
            self._preload_contact_latched = [False, False]
            horizon = self.cfg.horizon_steps
            box_pose = sample.snapshot.box.pose_b.repeat(horizon, 1)
            left_target, right_target = _palm_targets(
                box_pose,
                self.cfg.palm_lateral_offset_m,
                self.cfg.palm_reach_offset_m,
                sample.snapshot.left_arm.palm_pose_b[3:],
                sample.snapshot.right_arm.palm_pose_b[3:],
                self.cfg.right_palm_height_offset_m,
            )
            fractions = (
                torch.arange(1, horizon + 1, dtype=torch.float64) / horizon
            ).unsqueeze(1)
            left_palm = sample.snapshot.left_arm.palm_pose_b + fractions * (
                left_target - sample.snapshot.left_arm.palm_pose_b
            )
            right_palm = sample.snapshot.right_arm.palm_pose_b + fractions * (
                right_target - sample.snapshot.right_arm.palm_pose_b
            )
            solution = ObjectMpcSolution(
                box_pose=box_pose,
                box_twist=torch.zeros((horizon, 6), dtype=torch.float64),
                platform_yaw=sample.snapshot.platform_q_qd[0].repeat(horizon),
                left_palm_pose=left_palm,
                right_palm_pose=right_palm,
                left_wrench=torch.zeros((horizon, 6), dtype=torch.float64),
                right_wrench=torch.zeros((horizon, 6), dtype=torch.float64),
                diagnostics=ObjectMpcDiagnostics(
                    feasible=True,
                    fallback_used=False,
                    fallback_reason=None,
                    force_closure_margin=0.0,
                    saturation_fraction=0.0,
                    iterations=0,
                ),
            )
            self._last_safe = self._clone(solution)
            return solution
        if sample.phase is BimanualPhase.PRELOAD and sample.snapshot.box.supported:
            horizon = self.cfg.horizon_steps
            box_pose = sample.snapshot.box.pose_b.repeat(horizon, 1)
            if self._preload_palm_targets is None:
                self._preload_palm_targets = (
                    sample.snapshot.left_arm.palm_pose_b.clone(),
                    sample.snapshot.right_arm.palm_pose_b.clone(),
                )
            fingers_preclosed = bool(
                torch.mean(sample.snapshot.left_hand.q[2:]).item() >= 0.65
                and torch.mean(sample.snapshot.right_hand.q[2:]).item() >= 0.65
            )
            if fingers_preclosed:
                right_inward_complete_at_start = (
                    self._preload_inward_travel[1]
                    >= self.cfg.right_preload_inward_limit_m
                )
                for index, (hand, limit) in enumerate(
                    (
                        (sample.snapshot.left_hand, self.cfg.left_preload_inward_limit_m),
                        (sample.snapshot.right_hand, self.cfg.right_preload_inward_limit_m),
                    )
                ):
                    if bool(hand.contact_mask.any()):
                        self._preload_contact_latched[index] = True
                    if not self._preload_contact_latched[index]:
                        self._preload_inward_travel[index] = min(
                            limit,
                            self._preload_inward_travel[index]
                            + self.cfg.preload_palm_inward_speed_m_s * self.cfg.dt,
                        )
                if (
                    right_inward_complete_at_start
                    and not self._preload_contact_latched[1]
                ):
                    self._right_preload_forward_travel = min(
                        self.cfg.preload_forward_limit_m,
                        self._right_preload_forward_travel
                        + self.cfg.preload_palm_inward_speed_m_s * self.cfg.dt,
                    )
            left_target = self._preload_palm_targets[0].clone()
            right_target = self._preload_palm_targets[1].clone()
            left_target[1] -= self._preload_inward_travel[0]
            right_target[1] += self._preload_inward_travel[1]
            right_target[2] += self.cfg.right_preload_height_offset_m * min(
                1.0,
                self._preload_inward_travel[1]
                / self.cfg.right_preload_inward_limit_m,
            )
            left_target[0] += self.cfg.preload_forward_limit_m * min(
                1.0,
                self._preload_inward_travel[0]
                / self.cfg.left_preload_inward_limit_m,
            )
            right_target[0] += self._right_preload_forward_travel
            left_palm = left_target.repeat(horizon, 1)
            right_palm = right_target.repeat(horizon, 1)
            left_wrench = torch.zeros((horizon, 6), dtype=torch.float64)
            right_wrench = torch.zeros((horizon, 6), dtype=torch.float64)
            left_wrench[:, 1] = -self.cfg.preload_normal_force
            right_wrench[:, 1] = self.cfg.preload_normal_force
            solution = ObjectMpcSolution(
                box_pose=box_pose,
                box_twist=torch.zeros((horizon, 6), dtype=torch.float64),
                platform_yaw=sample.snapshot.platform_q_qd[0].repeat(horizon),
                left_palm_pose=left_palm,
                right_palm_pose=right_palm,
                left_wrench=left_wrench,
                right_wrench=right_wrench,
                diagnostics=ObjectMpcDiagnostics(
                    feasible=True,
                    fallback_used=False,
                    fallback_reason=None,
                    force_closure_margin=(
                        self.cfg.friction_coefficient
                        * self.cfg.preload_normal_force
                    ),
                    saturation_fraction=1.0,
                    iterations=0,
                ),
            )
            self._last_safe = self._clone(solution)
            return solution
        if not self._target_reachable(sample):
            return self._fallback(sample, "target_unreachable")
        result = solve_reference_qp(
            build_object_qp(sample, self.cfg),
            tolerance=self.cfg.qp_tolerance,
            max_iterations=self.cfg.qp_max_iterations,
        )
        if not result.success or not torch.isfinite(result.solution).all().item():
            return self._fallback(sample, "qp_infeasible")
        solution = self._solution(sample, result)
        self._last_safe = self._clone(solution)
        return solution
