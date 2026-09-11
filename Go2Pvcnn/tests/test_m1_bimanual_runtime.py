from __future__ import annotations

from dataclasses import replace

import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import DualArmMpcSolution
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
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
