"""One-second full-action teacher over constrained floating-base dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    solve_reference_qp,
)

from .contracts import BimanualCommand, BimanualSnapshot, FullDynamicsState, _float64
from .reduced_dynamics import condense_constrained_dynamics


TEACHER_DT = 0.04
TEACHER_HORIZON = 25
ACTIVE_DOF = 43
TASK_ACCELERATION_DOF = 18


def _pose_with_twist(
    current_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    previous = torch.cat((current_pose.unsqueeze(0), target_pose[:-1]), dim=0)
    twist = (target_pose - previous) / TEACHER_DT
    return torch.cat((target_pose, twist), dim=1)


def build_teacher_input(
    *,
    snapshot: BimanualSnapshot,
    dynamics: FullDynamicsState,
    task_jacobian: torch.Tensor,
    baseline_command: BimanualCommand,
    latest_solutions: dict[str, object | None],
    effort_limits: torch.Tensor,
) -> "TeacherInput":
    """Bridge the existing object/arm/hand hierarchy into one full-action plan."""

    if not isinstance(snapshot, BimanualSnapshot):
        raise TypeError("snapshot must be BimanualSnapshot")
    if not isinstance(dynamics, FullDynamicsState):
        raise TypeError("dynamics must be FullDynamicsState")
    if not isinstance(baseline_command, BimanualCommand):
        raise TypeError("baseline_command must be BimanualCommand")
    required = ("object", "arm", "left_hand", "right_hand", "wbc")
    if any(latest_solutions.get(name) is None for name in required):
        raise ValueError("all hierarchical solutions are required")
    object_solution = latest_solutions["object"]
    arm_solution = latest_solutions["arm"]
    left_hand = latest_solutions["left_hand"]
    right_hand = latest_solutions["right_hand"]
    nominal = baseline_command.effort.repeat(TEACHER_HORIZON, 1)
    nominal[:, 12:16] = 0.0
    target_acceleration = torch.zeros(
        (TEACHER_HORIZON, TASK_ACCELERATION_DOF), dtype=torch.float64
    )
    for node in range(TEACHER_HORIZON):
        arm_node = min(
            int(node * arm_solution.left.qdd.shape[0] / TEACHER_HORIZON),
            arm_solution.left.qdd.shape[0] - 1,
        )
        target_acceleration[node, 6:12] = (
            snapshot.left_arm.jacobian_b @ arm_solution.left.qdd[arm_node]
        )
        target_acceleration[node, 12:18] = (
            snapshot.right_arm.jacobian_b @ arm_solution.right.qdd[arm_node]
        )
    platform_position = object_solution.platform_yaw
    platform_previous = torch.cat(
        (snapshot.platform_q_qd[:1], platform_position[:-1])
    )
    platform_velocity = (platform_position - platform_previous) / TEACHER_DT
    source_feasible = bool(
        baseline_command.feasible
        and object_solution.diagnostics.feasible
        and arm_solution.both_feasible
        and left_hand.diagnostics.feasible
        and right_hand.diagnostics.feasible
    )
    return TeacherInput(
        dynamics=dynamics,
        nominal_action_trajectory=nominal,
        task_jacobian=task_jacobian,
        task_acceleration_target=target_acceleration,
        task_bias=torch.zeros(
            (TEACHER_HORIZON, TASK_ACCELERATION_DOF), dtype=torch.float64
        ),
        task_weights=torch.tensor(
            [20.0] * 6 + [10.0] * 12, dtype=torch.float64
        ),
        effort_limits=effort_limits,
        box_trajectory_b=torch.cat(
            (object_solution.box_pose, object_solution.box_twist), dim=1
        ),
        left_palm_trajectory_b=_pose_with_twist(
            snapshot.left_arm.palm_pose_b, object_solution.left_palm_pose
        ),
        right_palm_trajectory_b=_pose_with_twist(
            snapshot.right_arm.palm_pose_b, object_solution.right_palm_pose
        ),
        left_wrench_b=object_solution.left_wrench,
        right_wrench_b=object_solution.right_wrench,
        platform_trajectory=torch.stack(
            (platform_position, platform_velocity), dim=1
        ),
        source_feasible=source_feasible,
    )


@dataclass(frozen=True)
class TeacherInput:
    dynamics: FullDynamicsState
    nominal_action_trajectory: torch.Tensor
    task_jacobian: torch.Tensor
    task_acceleration_target: torch.Tensor
    task_bias: torch.Tensor
    task_weights: torch.Tensor
    effort_limits: torch.Tensor
    box_trajectory_b: torch.Tensor
    left_palm_trajectory_b: torch.Tensor
    right_palm_trajectory_b: torch.Tensor
    left_wrench_b: torch.Tensor
    right_wrench_b: torch.Tensor
    platform_trajectory: torch.Tensor
    source_feasible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.dynamics, FullDynamicsState):
            raise TypeError("dynamics must be FullDynamicsState")
        for name, shape in (
            ("nominal_action_trajectory", (25, 43)),
            ("task_jacobian", (18, 59)),
            ("task_acceleration_target", (25, 18)),
            ("task_bias", (25, 18)),
            ("task_weights", (18,)),
            ("effort_limits", (43,)),
            ("box_trajectory_b", (25, 12)),
            ("left_palm_trajectory_b", (25, 12)),
            ("right_palm_trajectory_b", (25, 12)),
            ("left_wrench_b", (25, 6)),
            ("right_wrench_b", (25, 6)),
            ("platform_trajectory", (25, 2)),
        ):
            object.__setattr__(
                self, name, _float64(name, getattr(self, name), shape)
            )
        if not torch.all(self.task_weights > 0.0).item():
            raise ValueError("task_weights must be strictly positive")
        if not torch.all(self.effort_limits > 0.0).item():
            raise ValueError("effort_limits must be strictly positive")
        if not isinstance(self.source_feasible, bool):
            raise TypeError("source_feasible must be bool")


@dataclass(frozen=True)
class TeacherDiagnostics:
    feasible: bool
    fallback_reason: str | None
    dynamics_residual_max: float
    contact_residual_max: float
    task_acceleration_error_max: float
    qp_iterations: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be bool")
        if self.fallback_reason is not None and not self.fallback_reason.strip():
            raise ValueError("fallback_reason must be None or non-empty")
        for name in (
            "dynamics_residual_max",
            "contact_residual_max",
            "task_acceleration_error_max",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if len(self.qp_iterations) != TEACHER_HORIZON:
            raise ValueError("qp_iterations must contain 25 entries")


@dataclass(frozen=True)
class TeacherSolution:
    action_trajectory: torch.Tensor
    box_trajectory_b: torch.Tensor
    left_palm_trajectory_b: torch.Tensor
    right_palm_trajectory_b: torch.Tensor
    left_wrench_b: torch.Tensor
    right_wrench_b: torch.Tensor
    platform_trajectory: torch.Tensor
    diagnostics: TeacherDiagnostics

    def __post_init__(self) -> None:
        for name, shape in (
            ("action_trajectory", (25, 43)),
            ("box_trajectory_b", (25, 12)),
            ("left_palm_trajectory_b", (25, 12)),
            ("right_palm_trajectory_b", (25, 12)),
            ("left_wrench_b", (25, 6)),
            ("right_wrench_b", (25, 6)),
            ("platform_trajectory", (25, 2)),
        ):
            object.__setattr__(
                self, name, _float64(name, getattr(self, name), shape)
            )
        if not isinstance(self.diagnostics, TeacherDiagnostics):
            raise TypeError("diagnostics must be TeacherDiagnostics")


class FullActionTeacher:
    """Project reference actions through a full coupled-dynamics task model."""

    def __init__(
        self,
        *,
        effort_regularization: float = 1.0e-2,
        qp_tolerance: float = 1.0e-7,
        qp_max_iterations: int = 256,
        residual_tolerance: float = 1.0e-6,
        fixed_base: bool = False,
    ) -> None:
        if effort_regularization <= 0.0 or not math.isfinite(effort_regularization):
            raise ValueError("effort_regularization must be finite and positive")
        self.effort_regularization = float(effort_regularization)
        self.qp_tolerance = float(qp_tolerance)
        self.qp_max_iterations = int(qp_max_iterations)
        self.residual_tolerance = float(residual_tolerance)
        if not isinstance(fixed_base, bool):
            raise TypeError("fixed_base must be bool")
        self.fixed_base = fixed_base

    @staticmethod
    def _bounded_nominal(sample: TeacherInput) -> torch.Tensor:
        action = torch.clamp(
            sample.nominal_action_trajectory,
            min=-sample.effort_limits,
            max=sample.effort_limits,
        )
        action[:, 12:16] = 0.0
        return action

    @staticmethod
    def _solution(
        sample: TeacherInput,
        action: torch.Tensor,
        diagnostics: TeacherDiagnostics,
    ) -> TeacherSolution:
        return TeacherSolution(
            action_trajectory=action,
            box_trajectory_b=sample.box_trajectory_b,
            left_palm_trajectory_b=sample.left_palm_trajectory_b,
            right_palm_trajectory_b=sample.right_palm_trajectory_b,
            left_wrench_b=sample.left_wrench_b,
            right_wrench_b=sample.right_wrench_b,
            platform_trajectory=sample.platform_trajectory,
            diagnostics=diagnostics,
        )

    def plan(self, sample: TeacherInput) -> TeacherSolution:
        if not isinstance(sample, TeacherInput):
            raise TypeError("sample must be TeacherInput")
        bounded_nominal = self._bounded_nominal(sample)
        if not sample.source_feasible:
            return self._solution(
                sample,
                bounded_nominal,
                TeacherDiagnostics(
                    feasible=False,
                    fallback_reason="source_plan_infeasible",
                    dynamics_residual_max=0.0,
                    contact_residual_max=0.0,
                    task_acceleration_error_max=0.0,
                    qp_iterations=(0,) * TEACHER_HORIZON,
                ),
            )

        reduced = condense_constrained_dynamics(sample.dynamics)
        task_action_map = sample.task_jacobian @ reduced.qdd_from_effort
        task_offset = sample.task_jacobian @ reduced.qdd_offset
        weight = torch.diag(sample.task_weights)
        identity = torch.eye(ACTIVE_DOF, dtype=torch.float64)
        hessian = 2.0 * (
            task_action_map.T @ weight @ task_action_map
            + self.effort_regularization * identity
        )
        lower = -sample.effort_limits.clone()
        upper = sample.effort_limits.clone()
        lower[12:16] = 0.0
        upper[12:16] = 0.0
        empty_matrix = torch.empty((0, ACTIVE_DOF), dtype=torch.float64)
        empty_vector = torch.empty(0, dtype=torch.float64)
        actions: list[torch.Tensor] = []
        iterations: list[int] = []
        feasible = True
        reason: str | None = None
        task_error_max = 0.0
        dynamics_residual_max = 0.0
        contact_residual_max = 0.0
        for node in range(TEACHER_HORIZON):
            node_lower = lower.clone()
            node_upper = upper.clone()
            node_lower[17:43] = bounded_nominal[node, 17:43]
            node_upper[17:43] = bounded_nominal[node, 17:43]
            if self.fixed_base:
                node_lower[:] = bounded_nominal[node]
                node_upper[:] = bounded_nominal[node]
            desired = (
                sample.task_acceleration_target[node]
                - task_offset
                - sample.task_bias[node]
            )
            gradient = -2.0 * (
                task_action_map.T @ weight @ desired
                + self.effort_regularization * bounded_nominal[node]
            )
            result = solve_reference_qp(
                DenseQpProblem(
                    hessian=hessian,
                    gradient=gradient,
                    equality_matrix=empty_matrix,
                    equality_rhs=empty_vector,
                    inequality_matrix=empty_matrix,
                    inequality_upper=empty_vector,
                    lower_bound=node_lower,
                    upper_bound=node_upper,
                ),
                tolerance=self.qp_tolerance,
                max_iterations=self.qp_max_iterations,
            )
            action = result.solution if result.success else bounded_nominal[node]
            action = action.clone()
            action[12:16] = 0.0
            actions.append(action)
            iterations.append(result.iterations)
            if not result.success:
                feasible = False
                reason = "teacher_qp_infeasible"
            qdd = reduced.qdd_offset + reduced.qdd_from_effort @ action
            contact_force = (
                reduced.contact_offset + reduced.contact_from_effort @ action
            )
            dynamics_residual = (
                sample.dynamics.mass_matrix @ qdd
                + sample.dynamics.bias
                - sample.dynamics.actuation_matrix @ action
                - sample.dynamics.wheel_contact_jacobian.T @ contact_force
            )
            contact_residual = (
                sample.dynamics.wheel_contact_jacobian @ qdd
                + sample.dynamics.wheel_contact_bias
            )
            task_error = task_action_map @ action - desired
            dynamics_residual_max = max(
                dynamics_residual_max,
                float(torch.max(torch.abs(dynamics_residual)).item()),
            )
            contact_residual_max = max(
                contact_residual_max,
                float(torch.max(torch.abs(contact_residual)).item()),
            )
            task_error_max = max(
                task_error_max, float(torch.max(torch.abs(task_error)).item())
            )
        if max(dynamics_residual_max, contact_residual_max) > self.residual_tolerance:
            feasible = False
            reason = "forward_dynamics_residual"
        action_trajectory = torch.stack(actions)
        if not torch.isfinite(action_trajectory).all().item():
            feasible = False
            reason = "nonfinite_teacher_action"
            action_trajectory = bounded_nominal
        return self._solution(
            sample,
            action_trajectory,
            TeacherDiagnostics(
                feasible=feasible,
                fallback_reason=reason,
                dynamics_residual_max=dynamics_residual_max,
                contact_residual_max=contact_residual_max,
                task_acceleration_error_max=task_error_max,
                qp_iterations=tuple(iterations),
            ),
        )


__all__ = [
    "ACTIVE_DOF",
    "TASK_ACCELERATION_DOF",
    "TEACHER_DT",
    "TEACHER_HORIZON",
    "FullActionTeacher",
    "TeacherDiagnostics",
    "TeacherInput",
    "TeacherSolution",
    "build_teacher_input",
]
