from __future__ import annotations

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import (
    DualArmMpcCoordinator,
    DualArmMpcInput,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_mpc import (
    ObjectMpcDiagnostics,
    ObjectMpcSolution,
)
from go2_pvcnn.control.m1_panda_coordination.arm_mpc import (
    ArmMpcDiagnostics,
    ArmMpcInput,
    ArmMpcSolution,
)


DTYPE = torch.float64


def _arm_input(marker: float = 0.0) -> ArmMpcInput:
    q = torch.zeros(7, dtype=DTYPE)
    q[0] = marker
    return ArmMpcInput(
        q=q,
        qd=torch.zeros(7, dtype=DTYPE),
        ee_pose_b=torch.zeros(6, dtype=DTYPE),
        ee_twist_b=torch.zeros(6, dtype=DTYPE),
        target_pose_b=torch.zeros(20, 6, dtype=DTYPE),
        target_twist_b=torch.zeros(20, 6, dtype=DTYPE),
        jacobian_b=torch.eye(6, 7, dtype=DTYPE),
        arm_mass_matrix=torch.eye(7, dtype=DTYPE),
        arm_bias=torch.zeros(7, dtype=DTYPE),
        base_arm_coupling=torch.zeros(6, 7, dtype=DTYPE),
        q_min=-torch.ones(7, dtype=DTYPE),
        q_max=torch.ones(7, dtype=DTYPE),
        qd_max=torch.ones(7, dtype=DTYPE),
        qdd_max=2.0 * torch.ones(7, dtype=DTYPE),
        effort_max=10.0 * torch.ones(7, dtype=DTYPE),
    )


def _object_solution(feasible: bool = True) -> ObjectMpcSolution:
    time = 0.04 * torch.arange(1, 26, dtype=DTYPE)
    box = torch.zeros(25, 6, dtype=DTYPE)
    left = box.clone()
    right = box.clone()
    left[:, 0] = time
    left[:, 1] = 0.12
    right[:, 0] = 2.0 * time
    right[:, 1] = -0.12
    return ObjectMpcSolution(
        box_pose=box,
        box_twist=torch.zeros_like(box),
        platform_yaw=torch.zeros(25, dtype=DTYPE),
        left_palm_pose=left,
        right_palm_pose=right,
        left_wrench=torch.zeros(25, 6, dtype=DTYPE),
        right_wrench=torch.zeros(25, 6, dtype=DTYPE),
        diagnostics=ObjectMpcDiagnostics(
            feasible=feasible,
            fallback_used=not feasible,
            fallback_reason=None if feasible else "object_failed",
            force_closure_margin=1.0,
            saturation_fraction=0.0,
            iterations=1,
        ),
    )


def _solution(sample: ArmMpcInput, feasible: bool) -> ArmMpcSolution:
    marker = sample.target_pose_b[0, 0].clone()
    q_ref = sample.q.clone()
    q_ref[0] = marker
    return ArmMpcSolution(
        q_ref=q_ref,
        qd_ref=torch.zeros(7, dtype=DTYPE),
        qdd=torch.zeros(20, 7, dtype=DTYPE),
        predicted_q=q_ref.repeat(20, 1),
        predicted_qd=torch.zeros(20, 7, dtype=DTYPE),
        predicted_pose_b=sample.target_pose_b.clone(),
        predicted_twist_b=sample.target_twist_b.clone(),
        predicted_dynamic_mount_wrench_b=torch.zeros(6, dtype=DTYPE),
        diagnostics=ArmMpcDiagnostics(
            feasible=feasible,
            fallback_used=not feasible,
            fallback_reason=None if feasible else "fake_failure",
            iterations=1,
            saturation_fraction=0.0,
            sigma_min=1.0,
            min_joint_margin=0.5,
            mean_joint_margin=0.5,
            ee_position_error=0.0,
            ee_orientation_error=0.0,
        ),
    )


class _FakePlanner:
    def __init__(self):
        self.samples: list[ArmMpcInput] = []

    def plan(self, sample: ArmMpcInput) -> ArmMpcSolution:
        self.samples.append(sample)
        return _solution(sample, feasible=bool(sample.q[0] >= 0.0))


def test_object_palm_targets_are_resampled_to_each_50_hz_arm_input():
    planners: list[_FakePlanner] = []

    def factory():
        planner = _FakePlanner()
        planners.append(planner)
        return planner

    result = DualArmMpcCoordinator(planner_factory=factory).plan(
        DualArmMpcInput(_arm_input(), _arm_input(), _object_solution())
    )

    assert result.both_feasible
    assert not result.synchronized_fallback
    left_target = planners[0].samples[0].target_pose_b
    right_target = planners[1].samples[0].target_pose_b
    assert left_target.shape == right_target.shape == (20, 6)
    assert left_target[0, 0].item() == pytest.approx(0.02)
    assert left_target[-1, 0].item() == pytest.approx(0.4)
    assert right_target[-1, 0].item() == pytest.approx(0.8)
    assert torch.allclose(planners[0].samples[0].target_twist_b[:, 0], torch.ones(20, dtype=DTYPE))


def test_one_side_failure_holds_both_last_safe_solutions():
    coordinator = DualArmMpcCoordinator(planner_factory=_FakePlanner)
    first = coordinator.plan(DualArmMpcInput(_arm_input(), _arm_input(), _object_solution()))
    second = coordinator.plan(
        DualArmMpcInput(_arm_input(0.2), _arm_input(-0.2), _object_solution())
    )

    assert first.both_feasible
    assert not second.both_feasible
    assert second.synchronized_fallback
    assert torch.equal(second.left.q_ref, first.left.q_ref)
    assert torch.equal(second.right.q_ref, first.right.q_ref)
    assert second.left.diagnostics.fallback_used
    assert second.right.diagnostics.fallback_used


def test_object_failure_does_not_advance_either_arm_or_call_planners_again():
    planners: list[_FakePlanner] = []

    def factory():
        planner = _FakePlanner()
        planners.append(planner)
        return planner

    coordinator = DualArmMpcCoordinator(planner_factory=factory)
    safe = coordinator.plan(DualArmMpcInput(_arm_input(), _arm_input(), _object_solution()))
    failed = coordinator.plan(
        DualArmMpcInput(_arm_input(0.3), _arm_input(0.3), _object_solution(False))
    )

    assert [len(planner.samples) for planner in planners] == [1, 1]
    assert failed.synchronized_fallback
    assert torch.equal(failed.left.q_ref, safe.left.q_ref)
    assert torch.equal(failed.right.q_ref, safe.right.q_ref)


def test_first_cycle_failure_holds_current_both_sides():
    result = DualArmMpcCoordinator(planner_factory=_FakePlanner).plan(
        DualArmMpcInput(_arm_input(0.25), _arm_input(-0.25), _object_solution())
    )

    assert result.synchronized_fallback
    assert torch.equal(result.left.q_ref, _arm_input(0.25).q)
    assert torch.equal(result.right.q_ref, _arm_input(-0.25).q)
    assert torch.count_nonzero(result.left.qd_ref) == 0
    assert torch.count_nonzero(result.right.qd_ref) == 0


def test_legacy_arm_input_contract_is_not_extended():
    assert tuple(ArmMpcInput.__dataclass_fields__) == (
        "q",
        "qd",
        "ee_pose_b",
        "ee_twist_b",
        "target_pose_b",
        "target_twist_b",
        "jacobian_b",
        "arm_mass_matrix",
        "arm_bias",
        "base_arm_coupling",
        "q_min",
        "q_max",
        "qd_max",
        "qdd_max",
        "effort_max",
    )
