"""Synchronized coordinator for two unchanged 50 Hz Panda Arm MPCs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import torch

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import (
    ARM_DOF,
    ARM_MPC_DT,
    ARM_MPC_HORIZON_STEPS,
    ARM_TASK_DOF,
    ArmMpcDiagnostics,
    ArmMpcInput,
    ArmMpcSolution,
    LinearizedArmMpc,
)

from .object_mpc import OBJECT_MPC_DT, ObjectMpcSolution


class _ArmPlanner(Protocol):
    def plan(self, sample: ArmMpcInput) -> ArmMpcSolution: ...


@dataclass(frozen=True)
class DualArmMpcInput:
    left: ArmMpcInput
    right: ArmMpcInput
    object_solution: ObjectMpcSolution

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmMpcInput):
            raise TypeError("left must be ArmMpcInput")
        if not isinstance(self.right, ArmMpcInput):
            raise TypeError("right must be ArmMpcInput")
        if not isinstance(self.object_solution, ObjectMpcSolution):
            raise TypeError("object_solution must be ObjectMpcSolution")


@dataclass(frozen=True)
class DualArmMpcSolution:
    left: ArmMpcSolution
    right: ArmMpcSolution
    both_feasible: bool
    synchronized_fallback: bool

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmMpcSolution):
            raise TypeError("left must be ArmMpcSolution")
        if not isinstance(self.right, ArmMpcSolution):
            raise TypeError("right must be ArmMpcSolution")
        if not isinstance(self.both_feasible, bool):
            raise TypeError("both_feasible must be bool")
        if not isinstance(self.synchronized_fallback, bool):
            raise TypeError("synchronized_fallback must be bool")
        if self.both_feasible and self.synchronized_fallback:
            raise ValueError("a feasible dual-arm solution cannot be a synchronized fallback")


def _resample_object_pose(
    current_pose: torch.Tensor, object_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly resample 25 Hz object targets onto the frozen 50 Hz arm grid."""

    source_times = OBJECT_MPC_DT * torch.arange(
        0, object_pose.shape[0] + 1, dtype=torch.float64
    )
    source_pose = torch.cat((current_pose.unsqueeze(0), object_pose), dim=0)
    arm_times = ARM_MPC_DT * torch.arange(
        1, ARM_MPC_HORIZON_STEPS + 1, dtype=torch.float64
    )
    target = torch.empty(
        (ARM_MPC_HORIZON_STEPS, ARM_TASK_DOF), dtype=torch.float64
    )
    for index, time in enumerate(arm_times):
        upper = int(torch.searchsorted(source_times, time, right=False).item())
        upper = max(1, min(upper, source_times.numel() - 1))
        lower = upper - 1
        fraction = (time - source_times[lower]) / (
            source_times[upper] - source_times[lower]
        )
        # Rotational coordinates are the canonical-base small-angle SO(3)
        # linearization used by the existing ArmMpcInput contract.
        target[index] = source_pose[lower] + fraction * (
            source_pose[upper] - source_pose[lower]
        )
    previous = torch.cat((current_pose.unsqueeze(0), target[:-1]), dim=0)
    target_twist = (target - previous) / ARM_MPC_DT
    return target, target_twist


def _route_target(sample: ArmMpcInput, object_pose: torch.Tensor) -> ArmMpcInput:
    target_pose, target_twist = _resample_object_pose(sample.ee_pose_b, object_pose)
    return replace(
        sample,
        target_pose_b=target_pose,
        target_twist_b=target_twist,
    )


def _clone_arm(solution: ArmMpcSolution, *, fallback_reason: str | None = None) -> ArmMpcSolution:
    diagnostics = solution.diagnostics
    if fallback_reason is not None:
        diagnostics = replace(
            diagnostics,
            feasible=False,
            fallback_used=True,
            fallback_reason=fallback_reason,
            iterations=0,
        )
    return ArmMpcSolution(
        q_ref=solution.q_ref.clone(),
        qd_ref=solution.qd_ref.clone(),
        qdd=solution.qdd.clone(),
        predicted_q=solution.predicted_q.clone(),
        predicted_qd=solution.predicted_qd.clone(),
        predicted_pose_b=solution.predicted_pose_b.clone(),
        predicted_twist_b=solution.predicted_twist_b.clone(),
        predicted_dynamic_mount_wrench_b=solution.predicted_dynamic_mount_wrench_b.clone(),
        diagnostics=diagnostics,
    )


def _hold_arm(sample: ArmMpcInput, reason: str) -> ArmMpcSolution:
    q = sample.q.clone()
    pose = sample.ee_pose_b.clone()
    joint_margin = torch.minimum(q - sample.q_min, sample.q_max - q)
    return ArmMpcSolution(
        q_ref=q,
        qd_ref=torch.zeros(ARM_DOF, dtype=torch.float64),
        qdd=torch.zeros((ARM_MPC_HORIZON_STEPS, ARM_DOF), dtype=torch.float64),
        predicted_q=q.repeat(ARM_MPC_HORIZON_STEPS, 1),
        predicted_qd=torch.zeros(
            (ARM_MPC_HORIZON_STEPS, ARM_DOF), dtype=torch.float64
        ),
        predicted_pose_b=pose.repeat(ARM_MPC_HORIZON_STEPS, 1),
        predicted_twist_b=torch.zeros(
            (ARM_MPC_HORIZON_STEPS, ARM_TASK_DOF), dtype=torch.float64
        ),
        predicted_dynamic_mount_wrench_b=torch.zeros(
            ARM_TASK_DOF, dtype=torch.float64
        ),
        diagnostics=ArmMpcDiagnostics(
            feasible=False,
            fallback_used=True,
            fallback_reason=reason,
            iterations=0,
            saturation_fraction=0.0,
            sigma_min=float(torch.linalg.svdvals(sample.jacobian_b).min().item()),
            min_joint_margin=float(joint_margin.min().item()),
            mean_joint_margin=float(joint_margin.mean().item()),
            ee_position_error=0.0,
            ee_orientation_error=0.0,
        ),
    )


class DualArmMpcCoordinator:
    """Commit both arm plans together or hold both at their last safe solution."""

    def __init__(
        self, planner_factory: Callable[[], _ArmPlanner] = LinearizedArmMpc
    ) -> None:
        if not callable(planner_factory):
            raise TypeError("planner_factory must be callable")
        self.left = planner_factory()
        self.right = planner_factory()
        if not callable(getattr(self.left, "plan", None)) or not callable(
            getattr(self.right, "plan", None)
        ):
            raise TypeError("planner_factory must create arm planners")
        self._last_safe: DualArmMpcSolution | None = None

    def _synchronized_hold(
        self, sample: DualArmMpcInput, reason: str
    ) -> DualArmMpcSolution:
        if self._last_safe is None:
            left = _hold_arm(sample.left, reason)
            right = _hold_arm(sample.right, reason)
        else:
            left = _clone_arm(self._last_safe.left, fallback_reason=reason)
            right = _clone_arm(self._last_safe.right, fallback_reason=reason)
        return DualArmMpcSolution(
            left=left,
            right=right,
            both_feasible=False,
            synchronized_fallback=True,
        )

    def plan(self, sample: DualArmMpcInput) -> DualArmMpcSolution:
        if not isinstance(sample, DualArmMpcInput):
            raise TypeError("sample must be DualArmMpcInput")
        if not sample.object_solution.diagnostics.feasible:
            return self._synchronized_hold(sample, "object_mpc_infeasible")
        left_input = _route_target(sample.left, sample.object_solution.left_palm_pose)
        right_input = _route_target(sample.right, sample.object_solution.right_palm_pose)
        left = self.left.plan(left_input)
        right = self.right.plan(right_input)
        if left.diagnostics.feasible and right.diagnostics.feasible:
            solution = DualArmMpcSolution(
                left=left,
                right=right,
                both_feasible=True,
                synchronized_fallback=False,
            )
            self._last_safe = DualArmMpcSolution(
                left=_clone_arm(left),
                right=_clone_arm(right),
                both_feasible=True,
                synchronized_fallback=False,
            )
            return solution
        reason = "left_arm_mpc_infeasible" if not left.diagnostics.feasible else "right_arm_mpc_infeasible"
        if not left.diagnostics.feasible and not right.diagnostics.feasible:
            reason = "both_arm_mpc_infeasible"
        return self._synchronized_hold(sample, reason)
