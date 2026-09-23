from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import DualArmMpcSolution
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FingertipPriorDiagnostics,
    PriorQueryResult,
)
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
from go2_pvcnn.control.m1_bimanual_coordination.grasp_goal import (
    generate_bimanual_grasp_goal,
)
from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime
from go2_pvcnn.control.m1_bimanual_coordination.whole_body_qp import (
    BimanualWbcDiagnostics,
    BimanualWbcSolution,
)
from tests.test_m1_bimanual_dual_arm_mpc import (
    _arm_input,
    _object_solution,
    _solution as _arm_solution,
)
from tests.test_m1_bimanual_hand_mpc import _input as _hand_input
from tests.test_m1_bimanual_object_mpc import _snapshot


class _ObjectController:
    def plan(self, _sample):
        return _object_solution()


class _ArmController:
    def plan(self, _sample):
        return DualArmMpcSolution(
            _arm_solution(_arm_input(), True),
            _arm_solution(_arm_input(), True),
            True,
            False,
        )


class _HandController:
    def plan(self, _sample):
        return O6HandMpc().plan(_hand_input())


class _RecordingUnavailablePrior:
    def __init__(self):
        self.calls = []

    def target(self, sample, baseline_qd):
        self.calls.append((sample, baseline_qd.clone()))
        return PriorQueryResult(
            target=None,
            diagnostics=FingertipPriorDiagnostics(False, "test_unavailable", 0.0),
        )


class _WbcController:
    def __init__(self, fail_at: int | None = None):
        self.calls = 0
        self.fail_at = fail_at

    def solve(self, _request):
        self.calls += 1
        feasible = self.calls != self.fail_at
        return BimanualWbcSolution(
            effort=torch.full((43,), float(self.calls), dtype=torch.float64),
            feasible=feasible,
            fallback_used=not feasible,
            diagnostics=BimanualWbcDiagnostics(
                qp_iterations=1,
                min_collision_distance=0.1,
                force_closure_margin=1.0,
                support_margin=0.1,
                fallback_reason=None if feasible else "fake_wbc_failure",
            ),
        )


def _runtime(wbc=None):
    return BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=_HandController(),
        right_hand_mpc=_HandController(),
        wbc=_WbcController() if wbc is None else wbc,
    )


def test_runtime_cadence_is_25_50_100_200_hz_at_200_hz_physics():
    runtime = _runtime()
    for step in range(16):
        runtime.compute(replace(_snapshot(), timestamp_ns=step + 1))
    assert runtime.counts == {"object": 2, "arm": 4, "hand": 8, "wbc": 16}


def test_rejected_wbc_cycle_returns_last_safe_command_atomically():
    runtime = _runtime(_WbcController(fail_at=2))
    first = runtime.compute(replace(_snapshot(), timestamp_ns=1))
    rejected = runtime.compute(replace(_snapshot(), timestamp_ns=2))
    assert first.feasible
    assert not rejected.feasible
    assert rejected.fallback_reasons == ("fake_wbc_failure",)
    assert torch.equal(rejected.effort, first.effort)


def test_runtime_rejects_nonmonotonic_snapshots():
    runtime = _runtime()
    snapshot = replace(_snapshot(), timestamp_ns=10)
    runtime.compute(snapshot)
    try:
        runtime.compute(snapshot)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("nonmonotonic snapshot was accepted")


def test_runtime_exposes_latest_attempt_even_when_wbc_rejects_it():
    runtime = _runtime(_WbcController(fail_at=1))
    command = runtime.compute(replace(_snapshot(), timestamp_ns=1))
    assert not command.feasible
    latest = runtime.latest_solutions
    assert latest["object"] is not None
    assert latest["arm"] is not None
    assert latest["left_hand"] is not None
    assert latest["right_hand"] is not None
    assert latest["wbc"] is not None


def test_runtime_reset_clears_all_temporal_state_and_accepts_reused_timestamp():
    runtime = _runtime()
    runtime.compute(replace(_snapshot(), timestamp_ns=10))

    runtime.reset()

    assert runtime.counts == {"object": 0, "arm": 0, "hand": 0, "wbc": 0}
    assert all(value is None for value in runtime.latest_solutions.values())
    assert runtime.mission.phase.name == "APPROACH"
    command = runtime.compute(replace(_snapshot(), timestamp_ns=10))
    assert command.feasible


def test_mission_reach_check_uses_final_palm_goal_not_first_waypoint():
    runtime = _runtime()
    object_solution = _object_solution()
    snapshot = _snapshot()
    snapshot = replace(
        snapshot,
        left_arm=replace(
            snapshot.left_arm,
            palm_pose_b=object_solution.left_palm_pose[0].clone(),
        ),
        right_arm=replace(
            snapshot.right_arm,
            palm_pose_b=object_solution.right_palm_pose[0].clone(),
        ),
    )
    wbc_solution = _WbcController().solve(None)

    diagnostics = runtime._mission_diagnostics(
        snapshot, object_solution, wbc_solution
    )

    assert not diagnostics.palms_reached


def test_runtime_uses_bounded_closed_loop_lift_target():
    runtime = _runtime()
    runtime.mission.phase = BimanualPhase.LIFT
    snapshot = _snapshot()
    contact = torch.tensor([True, False, False, False, False])
    snapshot = replace(
        snapshot,
        left_hand=replace(snapshot.left_hand, contact_mask=contact),
        right_hand=replace(snapshot.right_hand, contact_mask=contact),
    )

    sample = runtime._object_input(snapshot)

    assert sample.target_box_pose_b[0, 2] > snapshot.box.pose_b[2]
    assert sample.target_box_pose_b[-1, 2] < snapshot.box.pose_b[2] + 0.10
    assert runtime.latest_motion_target.recovery_side is None


def test_runtime_hand_inputs_keep_real_left_and_right_o6_measurements_isolated():
    runtime = _runtime()
    snapshot = _snapshot()
    left = replace(
        snapshot.left_hand,
        q=torch.arange(6, dtype=torch.float64) / 10.0,
        qd=torch.arange(6, dtype=torch.float64) / 20.0,
        fingertip_positions_b=torch.arange(15, dtype=torch.float64).reshape(5, 3),
        fingertip_jacobian_b=torch.arange(90, dtype=torch.float64).reshape(15, 6),
        contact_mask=torch.tensor([True, False, True, False, True]),
    )
    right = replace(
        snapshot.right_hand,
        q=-torch.arange(6, dtype=torch.float64) / 10.0,
        qd=-torch.arange(6, dtype=torch.float64) / 20.0,
        fingertip_positions_b=-torch.arange(15, dtype=torch.float64).reshape(5, 3),
        fingertip_jacobian_b=-torch.arange(90, dtype=torch.float64).reshape(15, 6),
        contact_mask=torch.tensor([False, True, False, True, False]),
    )
    snapshot = replace(snapshot, left_hand=left, right_hand=right)
    runtime.mission.phase = BimanualPhase.PRELOAD

    left_sample = runtime._hand_input(snapshot, "left", torch.zeros(6, dtype=torch.float64))
    right_sample = runtime._hand_input(snapshot, "right", torch.zeros(6, dtype=torch.float64))

    for sample, state, arm in (
        (left_sample, left, snapshot.left_arm),
        (right_sample, right, snapshot.right_arm),
    ):
        assert sample.phase is BimanualPhase.PRELOAD
        assert sample.q.dtype == sample.qd.dtype == torch.float64
        assert sample.q.device.type == sample.qd.device.type == "cpu"
        assert torch.equal(sample.q, state.q)
        assert torch.equal(sample.qd, state.qd)
        assert torch.equal(sample.fingertip_positions_b, state.fingertip_positions_b)
        assert torch.equal(sample.contact_jacobian, state.fingertip_jacobian_b)
        assert torch.equal(sample.contact_mask, state.contact_mask)
        assert torch.equal(sample.palm_pose_b, arm.palm_pose_b)
    assert not torch.equal(left_sample.fingertip_positions_b, right_sample.fingertip_positions_b)


def test_catalog_diagnostics_project_current_fingertip_forces_on_goal_normals():
    goal = generate_bimanual_grasp_goal(
        (0.0, 0.0, 0.5),
        dimensions=(0.30, 0.21, 0.035),
        grasp_profile="thin_two_hand",
        object_class="book",
    )
    runtime = BimanualRuntime(grasp_goal=goal)
    snapshot = _snapshot()
    left_normals = torch.tensor(
        [target.normal for target in goal.left_contact_targets[:2]],
        dtype=torch.float64,
    )
    right_normals = torch.tensor(
        [target.normal for target in goal.right_contact_targets[:2]],
        dtype=torch.float64,
    )
    left_forces = torch.zeros((5, 3), dtype=torch.float64)
    right_forces = torch.zeros((5, 3), dtype=torch.float64)
    left_forces[:2] = 2.0 * left_normals
    right_forces[:2] = 2.0 * right_normals
    left_forces[:2, 2] += 3.0
    right_forces[:2, 2] += 3.0
    contact_mask = torch.tensor([True, True, False, False, False])
    snapshot = replace(
        snapshot,
        left_hand=replace(
            snapshot.left_hand,
            fingertip_forces_b=left_forces,
            contact_mask=contact_mask,
        ),
        right_hand=replace(
            snapshot.right_hand,
            fingertip_forces_b=right_forces,
            contact_mask=contact_mask,
        ),
    )
    diagnostics = runtime._mission_diagnostics(
        snapshot,
        _object_solution(),
        _WbcController().solve(None),
    )
    assert diagnostics.left_normal_force_n == pytest.approx(4.0)
    assert diagnostics.right_normal_force_n == pytest.approx(4.0)
    assert diagnostics.vertical_force_n == pytest.approx(12.0)
    assert diagnostics.left_normal_alignment == pytest.approx(
        2.0 / (13.0**0.5)
    )
    assert diagnostics.right_normal_alignment == pytest.approx(
        2.0 / (13.0**0.5)
    )


def test_runtime_queries_independent_hand_priors_with_their_own_side_measurements():
    left_prior = _RecordingUnavailablePrior()
    right_prior = _RecordingUnavailablePrior()
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=O6HandMpc(expert_prior=left_prior),
        right_hand_mpc=O6HandMpc(expert_prior=right_prior),
        wbc=_WbcController(),
    )
    snapshot = _snapshot()
    left_positions = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    right_positions = -left_positions - 1.0
    snapshot = replace(
        snapshot,
        left_hand=replace(snapshot.left_hand, fingertip_positions_b=left_positions),
        right_hand=replace(snapshot.right_hand, fingertip_positions_b=right_positions),
    )
    runtime.compute(snapshot)
    assert len(left_prior.calls) == len(right_prior.calls) == 1
    expected_left = left_positions - snapshot.left_arm.palm_pose_b[:3]
    expected_right = right_positions - snapshot.right_arm.palm_pose_b[:3]
    assert torch.equal(left_prior.calls[0][0].fingertip_positions_b, expected_left)
    assert torch.equal(right_prior.calls[0][0].fingertip_positions_b, expected_right)
    assert not torch.equal(
        left_prior.calls[0][0].fingertip_positions_b,
        right_prior.calls[0][0].fingertip_positions_b,
    )
