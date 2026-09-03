from __future__ import annotations

from dataclasses import replace

import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import FullDynamicsState
from go2_pvcnn.control.m1_bimanual_coordination.full_action_teacher import (
    FullActionTeacher,
    TeacherInput,
    build_teacher_input,
)
from tests.test_m1_bimanual_runtime import _runtime
from tests.test_m1_bimanual_object_mpc import _snapshot


DTYPE = torch.float64
HORIZON = 25


def _dynamics() -> FullDynamicsState:
    mass = 2.0 * torch.eye(59, dtype=DTYPE)
    mass[16, 30] = 0.25
    mass[30, 16] = 0.25
    actuation = torch.zeros((59, 43), dtype=DTYPE)
    actuation[6:49] = torch.eye(43, dtype=DTYPE)
    contact = torch.zeros((12, 59), dtype=DTYPE)
    contact[:, :12] = torch.eye(12, dtype=DTYPE)
    return FullDynamicsState(
        mass_matrix=mass,
        bias=torch.linspace(-0.1, 0.1, 59, dtype=DTYPE),
        actuation_matrix=actuation,
        wheel_contact_jacobian=contact,
        wheel_contact_bias=torch.zeros(12, dtype=DTYPE),
    )


def _input() -> TeacherInput:
    task_jacobian = torch.zeros((18, 59), dtype=DTYPE)
    task_jacobian[:, 12:30] = torch.eye(18, dtype=DTYPE)
    return TeacherInput(
        dynamics=_dynamics(),
        nominal_action_trajectory=torch.zeros(HORIZON, 43, dtype=DTYPE),
        task_jacobian=task_jacobian,
        task_acceleration_target=torch.zeros(HORIZON, 18, dtype=DTYPE),
        task_bias=torch.zeros(HORIZON, 18, dtype=DTYPE),
        task_weights=torch.ones(18, dtype=DTYPE),
        effort_limits=100.0 * torch.ones(43, dtype=DTYPE),
        box_trajectory_b=torch.zeros(HORIZON, 12, dtype=DTYPE),
        left_palm_trajectory_b=torch.zeros(HORIZON, 12, dtype=DTYPE),
        right_palm_trajectory_b=torch.zeros(HORIZON, 12, dtype=DTYPE),
        left_wrench_b=torch.zeros(HORIZON, 6, dtype=DTYPE),
        right_wrench_b=torch.zeros(HORIZON, 6, dtype=DTYPE),
        platform_trajectory=torch.zeros(HORIZON, 2, dtype=DTYPE),
        source_feasible=True,
    )


def test_teacher_returns_complete_one_second_active_trajectory() -> None:
    solution = FullActionTeacher().plan(_input())

    assert solution.action_trajectory.shape == (25, 43)
    assert solution.box_trajectory_b.shape == (25, 12)
    assert solution.left_palm_trajectory_b.shape == (25, 12)
    assert solution.right_palm_trajectory_b.shape == (25, 12)
    assert solution.platform_trajectory.shape == (25, 2)
    assert torch.all(solution.action_trajectory[:, 12:16] == 0.0)
    assert solution.diagnostics.feasible
    assert solution.diagnostics.dynamics_residual_max < 1.0e-8
    assert solution.diagnostics.contact_residual_max < 1.0e-8


def test_teacher_uses_coupled_dynamics_to_track_task_acceleration() -> None:
    sample = _input()
    loaded_target = sample.task_acceleration_target.clone()
    loaded_target[:, 4] = 2.0

    nominal = FullActionTeacher().plan(sample)
    loaded = FullActionTeacher().plan(
        replace(sample, task_acceleration_target=loaded_target)
    )

    assert not torch.allclose(
        nominal.action_trajectory, loaded.action_trajectory
    )
    assert loaded.diagnostics.task_acceleration_error_max < 0.1


def test_teacher_rejects_infeasible_source_without_exposing_59_actions() -> None:
    solution = FullActionTeacher().plan(replace(_input(), source_feasible=False))

    assert not solution.diagnostics.feasible
    assert solution.diagnostics.fallback_reason == "source_plan_infeasible"
    assert solution.action_trajectory.shape[-1] == 43
    assert torch.all(solution.action_trajectory[:, 12:16] == 0.0)


def test_bridge_builds_teacher_input_from_live_hierarchical_plans() -> None:
    snapshot = _snapshot()
    runtime = _runtime()
    command = runtime.compute(snapshot)
    task_jacobian = torch.zeros((18, 59), dtype=DTYPE)
    task_jacobian[:6, :6] = torch.eye(6, dtype=DTYPE)
    task_jacobian[6:12, 19:26] = snapshot.left_arm.jacobian_b
    task_jacobian[12:18, 26:33] = snapshot.right_arm.jacobian_b

    sample = build_teacher_input(
        snapshot=snapshot,
        dynamics=_dynamics(),
        task_jacobian=task_jacobian,
        baseline_command=command,
        latest_solutions=runtime.latest_solutions,
        effort_limits=100.0 * torch.ones(43, dtype=DTYPE),
    )

    assert sample.nominal_action_trajectory.shape == (25, 43)
    assert sample.task_acceleration_target.shape == (25, 18)
    assert sample.box_trajectory_b.shape == (25, 12)
    assert sample.left_palm_trajectory_b.shape == (25, 12)
    assert sample.right_palm_trajectory_b.shape == (25, 12)
    assert sample.platform_trajectory.shape == (25, 2)
    assert sample.source_feasible
