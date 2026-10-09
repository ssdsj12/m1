from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import (
    DualArmMpcCoordinator,
    DualArmMpcSolution,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_mpc import BimanualObjectMpc, ObjectMpcInput
from go2_pvcnn.control.m1_bimanual_coordination.wire_scene_geometry import WireSceneGeometry
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FingertipPriorDiagnostics,
    PriorQueryResult,
)
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
from go2_pvcnn.control.m1_bimanual_coordination.grasp_goal import (
    generate_bimanual_grasp_goal,
)
from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime
from go2_pvcnn.control.m1_bimanual_coordination.deformx_adapter import (
    DloStepResult,
    DualDloStepResult,
)
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
    DloCenterlineState,
    build_wire_station_contact_goal,
)
from go2_pvcnn.control.m1_bimanual_coordination.whole_body_qp import (
    BimanualWbcDiagnostics,
    BimanualWbcSolution,
    BimanualWholeBodyQp,
)
from tests.test_m1_bimanual_dual_arm_mpc import (
    _arm_input,
    _object_solution,
    _solution as _arm_solution,
)
from tests.test_m1_bimanual_hand_mpc import _input as _hand_input
from tests.test_m1_bimanual_object_mpc import _snapshot
from tests.test_m1_wire_coupling import _geometry, _contact_snapshot
from go2_pvcnn.control.m1_bimanual_coordination.wire_coupling import WireCouplingPhase, WireCouplingState
from go2_pvcnn.control.m1_bimanual_coordination.dlo_frame_provider import dual_palm_attached_grasp_frames


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


class _RecordingHandController(_HandController):
    def __init__(self):
        self.samples = []

    def plan(self, sample):
        self.samples.append(sample)
        return super().plan(sample)


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


class _FixedDualDloBackend:
    def __init__(self):
        self.reset_calls = 0

    def step(self, **_frame):
        return DualDloStepResult(
            state=DloCenterlineState(
                np.array(
                    [[0.40, 0.00, 1.20], [0.50, 0.00, 1.20], [0.60, 0.00, 1.20]],
                    dtype=np.float64,
                ),
                radius_m=0.004,
            ),
            left_linear_impulse=np.zeros(3),
            left_angular_impulse=np.zeros(3),
            right_linear_impulse=np.zeros(3),
            right_angular_impulse=np.zeros(3),
            elapsed_time=0.005,
        )

    def reset(self):
        self.reset_calls += 1


class _FixedSingleDloBackend:
    def step(self, **_frame):
        return DloStepResult(
            state=DloCenterlineState(
                np.array([[0.40, 0.00, 1.20], [0.60, 0.00, 1.20]], dtype=np.float64),
                radius_m=0.004,
            ),
            linear_impulse=np.zeros(3),
            angular_impulse=np.zeros(3),
            elapsed_time=0.005,
        )


def _runtime(wbc=None):
    return BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=_HandController(),
        right_hand_mpc=_HandController(),
        wbc=_WbcController() if wbc is None else wbc,
    )


class _StagedBackend(_FixedDualDloBackend):
    def __init__(self):
        super().__init__()
        self.bind_calls = 0
        self.step_calls = 0
        self.fail = None
        self.on_bind = None

    def bind(self, state, **frames):
        self.bind_calls += 1
        if self.on_bind:
            self.on_bind()
        if self.fail == "bind":
            raise ValueError("bad bind geometry")
        self.bound_state = state
        self.bound_frames = frames

    def step(self, **frames):
        self.step_calls += 1
        if self.fail == "step":
            raise RuntimeError("integration failed")
        return super().step(**frames)


def _staged_runtime(wbc=None):
    backend = _StagedBackend()
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(), arm_mpc=_ArmController(),
        left_hand_mpc=_RecordingHandController(), right_hand_mpc=_RecordingHandController(),
        wbc=_WbcController() if wbc is None else wbc,
        dlo_backend=backend, wire_geometry_provider=lambda _snapshot: _geometry(),
        wire_frame_provider=dual_palm_attached_grasp_frames,
    )
    return runtime, backend


def _wire_tick(runtime, tick, *, left=True, right=True):
    return runtime.compute(replace(_contact_snapshot(left=left, right=right), timestamp_ns=tick))


def _clear_wire_inside_box_geometry():
    # The palm is inside the Box policy's +/- (.10, .09, .05) volume,
    # while its 4 cm Y/Z offsets leave the 5 mm wire physically clear.
    snapshot = _snapshot()
    box_pose = torch.tensor([.65, 0., .58, 0., 0., 0.], dtype=torch.float64)
    palm_pose = torch.tensor([.60, -.04, .62, .1, -.2, .3], dtype=torch.float64)
    tips = palm_pose[:3] + torch.tensor([
        [.06, -.01, -.01], [.08, .01, -.01], [.085, .02, -.01],
        [.09, .03, -.01], [.095, .04, -.01],
    ], dtype=torch.float64)
    snapshot = replace(
        snapshot,
        box=replace(snapshot.box, pose_b=box_pose, supported=True),
        right_arm=replace(snapshot.right_arm, palm_pose_b=palm_pose),
        right_hand=replace(snapshot.right_hand, fingertip_positions_b=tips,
                           contact_mask=torch.zeros(5, dtype=torch.bool)),
        left_hand=replace(snapshot.left_hand, contact_mask=torch.zeros(5, dtype=torch.bool)),
    )
    state = DloCenterlineState(np.array([[.626, 0., .58], [.674, 0., .58]]), radius_m=.005)
    geometry = WireSceneGeometry(
        centerline=state, center_b=box_pose[:3],
        tangent_b=torch.tensor([1., 0., 0.], dtype=torch.float64), radius_m=.005,
        left_station_b=torch.tensor(state.positions[0]),
        right_station_b=torch.tensor(state.positions[-1]),
    )
    return snapshot, geometry


@pytest.mark.parametrize("wire_mode", ["staged", "dual"])
def test_wire_real_object_planner_does_not_invoke_discarded_box_orientation(wire_mode):
    snapshot, geometry = _clear_wire_inside_box_geometry()
    runtime, backend = _staged_runtime(wbc=BimanualWholeBodyQp())
    runtime.object_mpc = BimanualObjectMpc()
    runtime.wire_geometry_provider = lambda _snapshot: geometry
    if wire_mode == "dual":
        runtime.wire_geometry_provider = None
        runtime.dlo_frame_provider = lambda _snapshot: {}
        backend.step = lambda **_frames: replace(
            _FixedDualDloBackend().step(), state=geometry.centerline)

    class RecordingArm:
        def __init__(self):
            self.samples = []

        def plan(self, sample):
            self.samples.append(sample)
            return _arm_solution(sample, True)

    runtime.arm_mpc = DualArmMpcCoordinator(planner_factory=RecordingArm)
    command = runtime.compute(snapshot)
    solution = runtime.latest_solutions["object"]
    assert solution.diagnostics.feasible, solution.diagnostics.fallback_reason
    assert runtime.object_mpc.right_orientation_mpc.last_diagnostics is None
    for side in ("left", "right"):
        torch.testing.assert_close(
            getattr(solution, f"{side}_palm_pose")[:, 3:],
            getattr(snapshot, f"{side}_arm").palm_pose_b[3:].repeat(25, 1),
        )
        assert len(getattr(runtime.arm_mpc, side).samples) == 1
    assert runtime.latest_solutions["arm"].both_feasible
    assert command.feasible and command.fallback_reasons == ()


def test_unloaded_wire_contact_between_hand_ticks_remains_accepted():
    snapshot, geometry = _clear_wire_inside_box_geometry()
    runtime, backend = _staged_runtime(wbc=BimanualWholeBodyQp())
    runtime.object_mpc = BimanualObjectMpc()
    runtime.left_hand_mpc = O6HandMpc()
    runtime.right_hand_mpc = O6HandMpc()
    runtime.wire_geometry_provider = lambda _snapshot: geometry
    assert runtime.compute(replace(snapshot, timestamp_ns=1)).feasible
    initial_hand = runtime.latest_solutions["right_hand"]
    assert initial_hand.diagnostics.slip_margin == 0.0
    forces = torch.zeros(5, 3, dtype=torch.float64)
    forces[0] = torch.tensor(
        [0.02256172709014661, -0.34093469384915487, -0.1239094286173951],
        dtype=torch.float64,
    )
    contacted = replace(snapshot, right_hand=replace(
        snapshot.right_hand, fingertip_forces_b=forces,
        contact_mask=torch.tensor([True, False, False, False, False]),
    ))
    assert runtime.compute(replace(contacted, timestamp_ns=2)).feasible
    assert runtime.latest_solutions["right_hand"] is initial_hand
    command = runtime.compute(replace(contacted, timestamp_ns=3))
    hand = runtime.latest_solutions["right_hand"]
    assert hand is not initial_hand
    assert hand.diagnostics.feasible
    assert hand.diagnostics.slip_margin == pytest.approx(-0.44006223674307093)
    assert runtime.mission.phase is BimanualPhase.APPROACH
    object_solution = runtime.latest_solutions["object"]
    assert torch.count_nonzero(object_solution.left_wrench).item() == 0
    assert torch.count_nonzero(object_solution.right_wrench).item() == 0
    assert command.feasible, command.fallback_reasons
    assert command.fallback_reasons == ()
    assert runtime._last_right_hand is hand
    assert runtime.latest_solutions["wbc"].diagnostics.force_closure_margin < 0.0
    assert backend.bind_calls == backend.step_calls == 0


@pytest.mark.parametrize("single_dlo", [False, True])
def test_ordinary_box_keeps_orientation_rejection_inside_box_geometry(single_dlo):
    snapshot, _geometry = _clear_wire_inside_box_geometry()
    runtime, _backend = _staged_runtime(wbc=BimanualWholeBodyQp())
    runtime.wire_geometry_provider = None
    runtime.dlo_backend = _FixedSingleDloBackend() if single_dlo else None
    runtime.dlo_frame_provider = lambda _snapshot: {}
    runtime.object_mpc = BimanualObjectMpc()
    command = runtime.compute(snapshot)
    assert not runtime.latest_solutions["object"].diagnostics.feasible
    assert runtime.latest_solutions["object"].diagnostics.fallback_reason == "right_palm_orientation_infeasible"
    assert "degenerate approach ray" in runtime.object_mpc.right_orientation_mpc.last_diagnostics.fallback_reason
    assert not command.feasible
    assert "object_mpc_infeasible" in command.fallback_reasons


def test_wire_real_object_planner_preserves_translation_rejection():
    snapshot, geometry = _clear_wire_inside_box_geometry()
    runtime, _backend = _staged_runtime(wbc=BimanualWholeBodyQp())
    runtime.object_mpc = BimanualObjectMpc()
    runtime.arm_mpc = DualArmMpcCoordinator()
    runtime.wire_geometry_provider = lambda _snapshot: geometry
    unreachable = snapshot.box.pose_b.repeat(25, 1)
    unreachable[:, 0] += 10.
    runtime._object_input_provider = lambda snapshot, _phase, _previous: ObjectMpcInput(
        snapshot=snapshot, target_box_pose_b=unreachable, phase=BimanualPhase.HOLD,
    )
    command = runtime.compute(snapshot)
    solution = runtime.latest_solutions["object"]
    assert not solution.diagnostics.feasible
    assert solution.diagnostics.fallback_reason == "target_unreachable"
    assert runtime.object_mpc.right_orientation_mpc.last_diagnostics is None
    assert not runtime.latest_solutions["arm"].both_feasible
    assert runtime.latest_solutions["arm"].right.diagnostics.fallback_reason == "object_mpc_infeasible"
    assert not command.feasible
    assert "object_mpc_infeasible" in command.fallback_reasons


def test_staged_wire_plans_scene_geometry_without_precontact_backend_calls():
    runtime, backend = _staged_runtime()
    _wire_tick(runtime, 1, left=False, right=False)
    # Already centered measured pinches must not be moved away by the old
    # hard-coded palm standoff, even before contact has been registered.
    snapshot = _contact_snapshot(left=False, right=False)
    for side in ("left", "right"):
        torch.testing.assert_close(
            getattr(runtime.latest_solutions["object"], f"{side}_palm_pose")[-1, :3],
            getattr(snapshot, f"{side}_arm").palm_pose_b[:3],
        )
    assert backend.bind_calls == backend.step_calls == 0
    assert runtime.latest_dlo_step is None
    np.testing.assert_allclose(runtime.latest_wire_goal.center, _geometry().center_b)
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.APPROACH_ENDPOINTS
    assert runtime.latest_wire_diagnostics is not None


def test_fifth_wire_contact_binds_only_after_feasible_wbc_then_steps_cached_geometry():
    runtime, backend = _staged_runtime()
    for tick in range(1, 5):
        _wire_tick(runtime, tick)
    assert backend.bind_calls == backend.step_calls == 0
    def check_boundary():
        assert runtime.wbc.calls == 5
        assert runtime.wire_coupling_state.phase is WireCouplingPhase.CONTACT_LATCHING
    backend.on_bind = check_boundary
    _wire_tick(runtime, 5)
    assert backend.bind_calls == 1 and backend.step_calls == 0
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND
    np.testing.assert_allclose(runtime.latest_dlo_step.state.positions, _geometry().centerline.positions)
    _wire_tick(runtime, 6)
    assert backend.step_calls == 1
    for tick in range(7, 10):
        _wire_tick(runtime, tick)
    np.testing.assert_allclose(runtime.latest_wire_goal.center, [0.5, 0., 1.2])
    assert backend.bind_calls == 1


@pytest.mark.parametrize("reject_tick", [1, 5, 9])
def test_rejected_wire_wbc_preserves_every_accepted_wire_field(reject_tick):
    runtime, backend = _staged_runtime(_WbcController(fail_at=reject_tick))
    for tick in range(1, reject_tick):
        _wire_tick(runtime, tick)
    before = (runtime.wire_coupling_state, runtime.latest_dlo_step, runtime._wire_attachment,
              runtime._last_command, runtime.latest_wire_goal,
              runtime._active_wire_goal, runtime._last_object, runtime._last_arm,
              runtime._last_left_hand, runtime._last_right_hand)
    calls = backend.bind_calls, backend.step_calls
    command = _wire_tick(runtime, reject_tick)
    assert not command.feasible
    after = (runtime.wire_coupling_state, runtime.latest_dlo_step, runtime._wire_attachment,
             runtime._last_command, runtime.latest_wire_goal,
             runtime._active_wire_goal, runtime._last_object, runtime._last_arm,
             runtime._last_left_hand, runtime._last_right_hand)
    assert all(a is b for a, b in zip(before, after))
    assert (backend.bind_calls, backend.step_calls) == calls
    assert runtime.latest_wire_diagnostics is not None
    assert not runtime.latest_wire_diagnostics.wbc_feasible
    assert not runtime.latest_wire_diagnostics.wire_goal_committed
    _wire_tick(runtime, reject_tick + 1)
    if reject_tick == 5:
        assert backend.bind_calls == 1
        assert runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND


@pytest.mark.parametrize("failure", ["contact", "bind", "step", "geometry"])
def test_wire_failure_safe_hold_keeps_last_command_goal_and_centerline(failure):
    runtime, backend = _staged_runtime()
    count = 4 if failure == "bind" else 6
    for tick in range(1, count + 1):
        _wire_tick(runtime, tick)
    old_command, old_goal, old_step = runtime._last_command, runtime.latest_wire_goal, runtime.latest_dlo_step
    backend.fail = failure
    if failure == "geometry":
        def invalid(_snapshot):
            raise ValueError("nonfinite scene geometry")
        runtime.wire_geometry_provider = invalid
    result = _wire_tick(runtime, count + 1, right=failure != "contact")
    assert not result.feasible
    torch.testing.assert_close(result.effort, old_command.effort)
    assert runtime._last_command is old_command
    assert runtime.latest_wire_goal is old_goal
    assert runtime.latest_dlo_step is old_step
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.SAFE_HOLD
    assert runtime.wire_coupling_state.failure_reason
    calls = backend.bind_calls, backend.step_calls
    _wire_tick(runtime, count + 2)
    assert (backend.bind_calls, backend.step_calls) == calls


def test_staged_wire_reset_clears_all_wire_state_and_resets_backend_once():
    runtime, backend = _staged_runtime()
    for tick in range(1, 7):
        _wire_tick(runtime, tick)
    runtime.reset()
    assert runtime.wire_coupling_state == WireCouplingState.initial()
    assert runtime._wire_attachment is None
    assert runtime.latest_dlo_step is None
    assert runtime.latest_wire_goal is None
    assert runtime._active_wire_goal is None
    assert runtime.latest_wire_diagnostics is None
    assert backend.reset_calls == 1


def test_latched_wire_side_holds_accepted_targets_while_other_side_approaches():
    runtime, backend = _staged_runtime()
    for tick in range(1, 6):
        _wire_tick(runtime, tick, right=False)
    previous = runtime._last_object
    goal = runtime.latest_wire_goal
    moved = _geometry()
    shift = np.array([0.05, 0., 0.])
    moved = replace(moved, centerline=DloCenterlineState(moved.centerline.positions + shift, moved.radius_m))
    runtime.wire_geometry_provider = lambda _snapshot: moved
    for tick in range(6, 10):
        _wire_tick(runtime, tick, right=False)
    torch.testing.assert_close(runtime._last_object.left_palm_pose, previous.left_palm_pose)
    assert runtime.latest_wire_goal.left_contact_targets == goal.left_contact_targets
    assert runtime.latest_wire_goal.right_contact_targets != goal.right_contact_targets
    assert backend.bind_calls == 0


def test_nonfinite_backend_impulses_enter_safe_hold_without_publishing_step():
    runtime, backend = _staged_runtime()
    for tick in range(1, 6):
        _wire_tick(runtime, tick)
    before = runtime.latest_dlo_step
    backend.step = lambda **_frames: replace(before, left_linear_impulse=np.full(3, np.nan))
    result = _wire_tick(runtime, 6)
    assert not result.feasible
    assert runtime.latest_dlo_step is before
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.SAFE_HOLD


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        pytest.param("left_linear_impulse", np.zeros(2), id="wrong-impulse-shape"),
        pytest.param("elapsed_time", np.zeros(2), id="nonscalar-elapsed-time"),
        pytest.param("elapsed_time", -1.0, id="negative-elapsed-time"),
    ],
)
def test_malformed_backend_result_preserves_accepted_wire_state_and_sticks_safe_hold(
    field, invalid_value,
):
    runtime, backend = _staged_runtime()
    for tick in range(1, 6):
        _wire_tick(runtime, tick)
    accepted = (
        runtime._last_command,
        runtime.latest_wire_goal,
        runtime.latest_dlo_step,
        runtime._wire_attachment,
    )

    def malformed_step(**_frames):
        backend.step_calls += 1
        return replace(runtime.latest_dlo_step, **{field: invalid_value})

    backend.step = malformed_step
    result = _wire_tick(runtime, 6)

    assert not result.feasible
    assert runtime._last_command is accepted[0]
    assert runtime.latest_wire_goal is accepted[1]
    assert runtime.latest_dlo_step is accepted[2]
    assert runtime._wire_attachment is accepted[3]
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.SAFE_HOLD
    assert runtime.wire_coupling_state.failure_reason.startswith("wire_backend_failure:")
    assert (backend.bind_calls, backend.step_calls) == (1, 1)

    held_failure_reason = runtime.wire_coupling_state.failure_reason
    held_result = _wire_tick(runtime, 7)

    assert not held_result.feasible
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.SAFE_HOLD
    assert runtime.wire_coupling_state.failure_reason == held_failure_reason
    assert all(
        actual is expected
        for actual, expected in zip(
            (
                runtime._last_command,
                runtime.latest_wire_goal,
                runtime.latest_dlo_step,
                runtime._wire_attachment,
            ),
            accepted,
        )
    )
    assert (backend.bind_calls, backend.step_calls) == (1, 1)


def test_invalid_backend_centerline_enters_safe_hold_without_publishing_step():
    runtime, backend = _staged_runtime()
    for tick in range(1, 6):
        _wire_tick(runtime, tick)
    before = runtime.latest_dlo_step
    old_command = runtime._last_command
    backend.step = lambda **_frames: replace(before, state=None)

    result = _wire_tick(runtime, 6)

    assert not result.feasible
    assert runtime.latest_dlo_step is before
    assert runtime._last_command is old_command
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.SAFE_HOLD
    assert "DloCenterlineState" in runtime.wire_coupling_state.failure_reason


def test_rejected_contact_dropout_does_not_commit_safe_hold_or_step_backend():
    runtime, backend = _staged_runtime(_WbcController(fail_at=6))
    for tick in range(1, 6):
        _wire_tick(runtime, tick)
    before = runtime.wire_coupling_state
    result = _wire_tick(runtime, 6, right=False)
    assert not result.feasible
    assert runtime.wire_coupling_state is before
    assert backend.step_calls == 0
    _wire_tick(runtime, 7)
    assert runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND
    assert backend.bind_calls == backend.step_calls == 1


def test_runtime_cadence_is_25_50_100_200_hz_at_200_hz_physics():
    runtime = _runtime()
    for step in range(16):
        runtime.compute(replace(_snapshot(), timestamp_ns=step + 1))
    assert runtime.counts == {"object": 2, "arm": 4, "hand": 8, "wbc": 16}


def test_runtime_routes_dual_dlo_centerline_into_arm_palm_trajectory():
    left_hand = _RecordingHandController()
    right_hand = _RecordingHandController()
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=left_hand,
        right_hand_mpc=right_hand,
        wbc=_WbcController(),
        dlo_backend=_FixedDualDloBackend(),
        dlo_frame_provider=lambda _snapshot: {
            "left_frame": {},
            "right_frame": {},
        },
    )
    rigid_solution = _object_solution()

    runtime.compute(replace(_snapshot(), timestamp_ns=1))

    routed = runtime.latest_solutions["object"]
    assert runtime.latest_wire_goal is not None
    assert runtime.latest_wire_goal.object_class == "wire"
    assert not torch.equal(routed.left_palm_pose, rigid_solution.left_palm_pose)
    assert not torch.equal(routed.right_palm_pose, rigid_solution.right_palm_pose)
    for controller in (left_hand, right_hand):
        sample = controller.samples[-1]
        assert sample.fingertip_target_positions_b is not None
        assert torch.equal(
            sample.fingertip_target_mask,
            torch.tensor([True, True, False, False, False]),
        )


def test_runtime_keeps_single_endpoint_dlo_backend_backward_compatible():
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=_HandController(),
        right_hand_mpc=_HandController(),
        wbc=_WbcController(),
        dlo_backend=_FixedSingleDloBackend(),
        dlo_frame_provider=lambda _snapshot: {"position": np.zeros(3)},
    )

    runtime.compute(replace(_snapshot(), timestamp_ns=1))

    assert isinstance(runtime.latest_dlo_step, DloStepResult)
    assert runtime.latest_wire_goal is None
    torch.testing.assert_close(
        runtime.latest_solutions["object"].left_palm_pose,
        _object_solution().left_palm_pose,
    )


def test_runtime_reset_clears_the_stateful_dlo_backend_and_public_goal():
    backend = _FixedDualDloBackend()
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=_HandController(),
        right_hand_mpc=_HandController(),
        wbc=_WbcController(),
        dlo_backend=backend,
        dlo_frame_provider=lambda _snapshot: {"left_frame": {}, "right_frame": {}},
    )
    runtime.compute(replace(_snapshot(), timestamp_ns=1))
    assert runtime.latest_wire_goal is not None

    runtime.reset()

    assert backend.reset_calls == 1
    assert runtime.latest_dlo_step is None
    assert runtime.latest_wire_goal is None


def test_rejected_wbc_cycle_does_not_publish_unaccepted_wire_goal():
    left_hand = _RecordingHandController()
    runtime = BimanualRuntime(
        object_mpc=_ObjectController(),
        arm_mpc=_ArmController(),
        left_hand_mpc=left_hand,
        right_hand_mpc=_HandController(),
        wbc=_WbcController(fail_at=1),
        dlo_backend=_FixedDualDloBackend(),
        dlo_frame_provider=lambda _snapshot: {"left_frame": {}, "right_frame": {}},
    )

    command = runtime.compute(replace(_snapshot(), timestamp_ns=1))

    assert not command.feasible
    assert left_hand.samples[-1].fingertip_target_positions_b is not None
    assert runtime.latest_wire_goal is None


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


@pytest.mark.parametrize("near_sides", ((False, False), (True, False), (False, True), (True, True)))
def test_wire_mission_reach_requires_both_true_stations_not_receding_waypoints(near_sides):
    runtime, _backend = _staged_runtime()
    planner = runtime.dlo_trajectory_planner
    original_plan = planner.plan_state

    def nearby_waypoints(snapshot, object_solution, state):
        plan = original_plan(snapshot, object_solution, state)
        return replace(plan, object_solution=replace(plan.object_solution,
            left_palm_pose=snapshot.left_arm.palm_pose_b.repeat(25, 1),
            right_palm_pose=snapshot.right_arm.palm_pose_b.repeat(25, 1)))

    planner.plan_state = nearby_waypoints
    snapshot = _contact_snapshot(left=False, right=False)
    for side, near in zip(("left", "right"), near_sides):
        hand = getattr(snapshot, f"{side}_hand")
        shifted = hand.fingertip_positions_b.clone()
        if not near:
            shifted[:2, 0] += .20
        snapshot = replace(snapshot, **{f"{side}_hand": replace(hand, fingertip_positions_b=shifted)})
    for tick in range(runtime.mission.cfg.approach_dwell_steps + 1):
        runtime.compute(replace(snapshot, timestamp_ns=tick + 1))
    expected = BimanualPhase.PRELOAD if all(near_sides) else BimanualPhase.APPROACH
    assert runtime.mission.phase is expected


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


def test_staged_wire_diagnostics_use_outward_station_normals_not_catalog_obb_normals():
    generic_goal = generate_bimanual_grasp_goal(
        (0.5, 0.0, 1.2),
        dimensions=(0.08, 0.01, 0.01),
        grasp_profile="cylindrical",
        object_class="wire",
    )
    wire_goal = build_wire_station_contact_goal(_geometry().centerline)
    runtime = BimanualRuntime(grasp_goal=generic_goal)
    snapshot = _snapshot()
    contact_mask = torch.tensor([True, True, False, False, False])
    hands = {}
    for side in ("left", "right"):
        normals = torch.tensor(
            [target.normal for target in getattr(wire_goal, f"{side}_contact_targets")],
            dtype=torch.float64,
        )
        forces = torch.zeros((5, 3), dtype=torch.float64)
        forces[:2] = 2.0 * normals
        hands[f"{side}_hand"] = replace(
            getattr(snapshot, f"{side}_hand"),
            fingertip_forces_b=forces,
            contact_mask=contact_mask,
        )
    snapshot = replace(snapshot, **hands)

    diagnostics = runtime._mission_diagnostics(
        snapshot,
        _object_solution(),
        _WbcController().solve(None),
        wire_goal=wire_goal,
    )

    assert diagnostics.left_normal_force_n == pytest.approx(4.0)
    assert diagnostics.right_normal_force_n == pytest.approx(4.0)
    assert diagnostics.left_normal_alignment == pytest.approx(1.0)
    assert diagnostics.right_normal_alignment == pytest.approx(1.0)


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
