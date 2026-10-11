from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from go2_pvcnn.control.m1_bimanual_coordination.wire_coupling import (
    WireCouplingDiagnostics, WireCouplingPhase,
)
from scripts.m1_dual_panda_o6_bimanual_play import (
    _WireReportRecorder,
    _write_wire_report_atomic,
)
from tests.test_m1_bimanual_runtime import (
    _HandController, _WbcController, _staged_runtime, _wire_tick,
)
from tests.test_m1_wire_coupling import _contact_snapshot
from tests.test_m1_wire_coupling import _geometry
from tests.test_m1_bimanual_prior_startup_binding import _load_wrapper
from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualCommand, BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import (
    LatentNormalizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.latent_runtime import LatentRuntime
from go2_pvcnn.control.m1_bimanual_coordination.wire_contact_acceptance import (
    WireAcceptanceConfig, WireAcceptanceTracker, WireAcceptancePhase,
)
from tests.test_m1_bimanual_full_action_teacher import _input as _teacher_input
from tests.test_m1_bimanual_latent_runtime import _Model, _Safety, _Teacher


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_m1_wire_contact_report.py"


def _acceptance_wrapper(monkeypatch):
    from tests.test_m1_bimanual_safety_projection import _input

    module = _load_wrapper(monkeypatch)
    wrapper = object.__new__(module.M1DualPandaO6BimanualWrapper)
    geometry = _geometry()
    snapshot = _contact_snapshot(left=True, right=True)
    for side, endpoint in (("left", geometry.left_station_b), ("right", geometry.right_station_b)):
        arm = getattr(snapshot, f"{side}_arm")
        hand = getattr(snapshot, f"{side}_hand")
        pose = arm.palm_pose_b.clone()
        pose[:3] = endpoint
        forces = torch.zeros((5, 3), dtype=torch.float64)
        forces[:2, 2] = 0.4
        snapshot = replace(snapshot, **{
            f"{side}_arm": replace(arm, palm_pose_b=pose),
            f"{side}_hand": replace(hand, fingertip_forces_b=forces),
        })
    good = SimpleNamespace(feasible=True)
    hand_solution = SimpleNamespace(q_ref=torch.full((6,), 0.7, dtype=torch.float64), diagnostics=good)
    runtime = SimpleNamespace(
        mission=SimpleNamespace(phase=BimanualPhase.LIFT),
        latest_solutions={"object": SimpleNamespace(diagnostics=good,
            left_palm_pose=snapshot.left_arm.palm_pose_b[None],
            right_palm_pose=snapshot.right_arm.palm_pose_b[None]),
            "arm": SimpleNamespace(both_feasible=True), "left_hand": hand_solution,
            "right_hand": hand_solution, "wbc": SimpleNamespace(feasible=True)},
        reset=lambda: None,
    )
    lane = module.BimanualLaneController(0, SimpleNamespace(), runtime, None, None)
    lane.wire_acceptance_tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    lane.wire_acceptance_tracker.reset(wire_center_height_m=1.2)
    lane.last_safety_result = SimpleNamespace(feasible=True)
    lane.baseline_command = BimanualCommand(1, torch.ones(43, dtype=torch.float64), True, ())
    wrapper.lanes = [lane]
    wrapper._wire_geometry_provider = lambda _snapshot: geometry
    wrapper.env = SimpleNamespace(unwrapped=SimpleNamespace(physics_dt=0.1))
    # The only substitute here is the Isaac-dependent safety-input builder;
    # gate logic, tracker, contracts, and contact evaluation run unchanged.
    wrapper._safety_input = lambda lane, snap, dynamics, candidate: replace(
        _input(), candidate_effort=candidate, safe_effort=lane.baseline_command.effort)
    lane.safety = SimpleNamespace(project=lambda sample: SimpleNamespace(
        effort=sample.candidate_effort.clone(), feasible=True, fallback_reason=None))
    return module, wrapper, lane, snapshot


def _wire_command(wrapper, lane, snapshot):
    return wrapper._apply_wire_acceptance(lane, snapshot, None, lane.baseline_command)


def test_wire_acceptance_preserves_the_final_safety_effort_vector(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    projected = []
    def project(sample):
        effort = sample.candidate_effort.clone()
        projected.append(effort)
        return SimpleNamespace(effort=effort, feasible=True, fallback_reason=None)
    lane.safety = SimpleNamespace(project=project)
    command = _wire_command(wrapper, lane, snapshot)
    assert projected
    assert torch.equal(command.effort, projected[-1])


def test_wrapper_requires_continuous_real_contact_before_lift(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    for _ in range(4):
        command = _wire_command(wrapper, lane, snapshot)
        assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
        assert not wrapper.wire_acceptance_state.lift_permitted
    _wire_command(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.lift_permitted
    command = _wire_command(wrapper, lane, snapshot)
    assert torch.equal(command.effort[17:31], torch.ones(14, dtype=torch.float64))
    assert command.effort.shape == (43,)


def test_contact_loss_holds_arms_and_reclamps_only_uncontacted_digits(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    wrapper._lane_hand_targets(lane, snapshot)
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.tensor([True, False, False, False, False])))
    command = _wire_command(wrapper, lane, lost)
    left, right = wrapper._lane_hand_targets(lane, lost)
    assert wrapper.wire_acceptance_state.request_reclamp
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert torch.equal(left[:3], snapshot.left_hand.q[:3])
    assert torch.equal(right[:2], snapshot.right_hand.q[:2])
    assert torch.equal(right[2:], torch.full((4,), 0.7, dtype=torch.float64))


@pytest.mark.parametrize("failure", ["object", "arm", "left_hand", "right_hand", "wbc", "safety"])
def test_contact_loss_with_rejection_preserves_hold_and_current_safety(monkeypatch, failure):
    from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import SafetyProjection
    from tests.test_m1_bimanual_safety_projection import _input

    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    previous = lane.wire_last_safe_command.effort.clone()
    assert torch.equal(previous[17:31], torch.ones(14, dtype=torch.float64))
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.tensor([True, False, False, False, False])))
    limits = torch.full((43,), 0.25, dtype=torch.float64)
    lane.safety = SafetyProjection()
    wrapper._safety_input = lambda lane, snap, dynamics, candidate: replace(
        _input(), candidate_effort=candidate, safe_effort=lane.baseline_command.effort,
        effort_limits=limits, phase=BimanualPhase.LIFT,
        force_closure_margin=-1.0 if failure == "safety" else 1.0)
    if failure != "safety":
        solution = lane.runtime.latest_solutions[failure]
        if failure == "arm":
            solution.both_feasible = False
        else:
            # Do not mutate the shared feasible diagnostic on the other layers.
            lane.runtime.latest_solutions[failure] = SimpleNamespace(feasible=False)

    command = _wire_command(wrapper, lane, lost)

    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "safety_rejected"
    torch.testing.assert_close(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert torch.all(command.effort.abs() <= limits)
    assert torch.equal(command.effort[12:16], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(lane.wire_last_safe_command.effort, previous)


def test_proximity_without_filtered_force_cannot_advance_acceptance(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    snapshot = replace(snapshot, left_hand=replace(snapshot.left_hand,
        fingertip_forces_b=torch.zeros((5, 3), dtype=torch.float64)))
    for _ in range(7):
        _wire_command(wrapper, lane, snapshot)
    assert not wrapper.wire_acceptance_state.lift_permitted
    assert wrapper.wire_acceptance_diagnostics["left_valid_contacts"] == 0


def test_acceptance_diagnostics_include_base_frame_measurements_and_timers(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    _wire_command(wrapper, lane, snapshot)
    result = wrapper.wire_acceptance_diagnostics
    assert result["frame"] == "m1_base"
    assert result["wire_center_height_m"] == pytest.approx(1.2)
    assert result["bilateral_contact_elapsed_s"] == pytest.approx(0.1)
    assert result["left_slip_m"] == 0
    assert result["left_palm_error_m"] == 0
    assert len(result["left_fingertip_distance_m"]) == 5
    assert result["left_fingertip_forces_b"] == snapshot.left_hand.fingertip_forces_b.tolist()
    assert result["left_contact_normals_b"][:2] == [[0., 0., 1.]] * 2
    assert result["feasibility"]["safety"]
    assert result["rejection_reason"] is None
    json.dumps(result, allow_nan=False)


def test_approach_outside_slip_tolerance_does_not_terminate_before_grasp(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission.phase = BimanualPhase.APPROACH
    pose = snapshot.left_arm.palm_pose_b.clone()
    pose[0] += 0.2
    snapshot = replace(snapshot, left_arm=replace(snapshot.left_arm, palm_pose_b=pose))
    command = _wire_command(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.phase is WireAcceptancePhase.CONTACT
    assert wrapper.wire_acceptance_state.terminal_reason is None
    assert command.feasible


@pytest.mark.parametrize("phase", [BimanualPhase.PRELOAD, BimanualPhase.GRASP, BimanualPhase.LIFT])
def test_nonzero_palm_offset_tracks_endpoint_displacement_from_grasp(monkeypatch, phase):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission.phase = phase
    for side in ("left", "right"):
        getattr(snapshot, f"{side}_arm").palm_pose_b[:3] += torch.tensor([0.08, 0.0, -0.04])
    for _ in range(6):
        command = _wire_command(wrapper, lane, snapshot)
    assert command.feasible
    assert wrapper.wire_acceptance_state.lift_permitted
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0.0)
    snapshot.left_arm.palm_pose_b[0] += 0.01
    assert _wire_command(wrapper, lane, snapshot).feasible
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0.01)
    snapshot.left_arm.palm_pose_b[0] += 0.011
    assert not _wire_command(wrapper, lane, snapshot).feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "excessive_endpoint_slip"


def test_palm_rotation_preserving_endpoint_attachment_is_not_slip(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    geometry = _geometry()
    local_offset = torch.tensor([0.08, 0., 0.04], dtype=torch.float64)
    for side in ("left", "right"):
        pose = getattr(snapshot, f"{side}_arm").palm_pose_b
        pose[:3] = getattr(geometry, f"{side}_station_b") - local_offset
        pose[3:] = 0.0
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    # A quarter turn about each stationary grasp point preserves its palm-local
    # displacement even though that displacement changes in the base frame.
    for side in ("left", "right"):
        pose = getattr(snapshot, f"{side}_arm").palm_pose_b
        pose[:3] = getattr(geometry, f"{side}_station_b") - torch.tensor([0.04, 0., -0.08])
        pose[4] = torch.pi / 2
    command = _wire_command(wrapper, lane, snapshot)
    assert command.feasible
    assert wrapper.wire_acceptance_state.lift_permitted
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0., abs=1e-8)
    assert wrapper.wire_acceptance_diagnostics["right_slip_m"] == pytest.approx(0., abs=1e-8)


def test_inconsistent_contact_does_not_capture_grasp_slip_reference(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission.phase = BimanualPhase.PRELOAD
    inconsistent = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_consistent=False))
    assert _wire_command(wrapper, lane, inconsistent).feasible
    assert lane.wire_palm_endpoint_offsets_local is None
    assert not lane.wire_acceptance_started
    snapshot.left_arm.palm_pose_b[0] += 0.08
    assert _wire_command(wrapper, lane, snapshot).feasible
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0.0)


def test_slip_reference_survives_reclamp_and_reset_captures_new_grasp(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    snapshot.left_arm.palm_pose_b[0] += 0.08
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.lift_permitted
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.zeros(5, dtype=torch.bool)))
    _wire_command(wrapper, lane, lost)
    snapshot.left_arm.palm_pose_b[0] += 0.015
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.lift_permitted
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0.015)
    wrapper._reset_wire_acceptance(lane, snapshot)
    assert not lane.wire_acceptance_started
    snapshot.left_arm.palm_pose_b[0] += 0.1
    _wire_command(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.terminal_reason is None
    assert wrapper.wire_acceptance_diagnostics["left_slip_m"] == pytest.approx(0.0)


def test_final_hold_safety_rejection_is_atomic_and_clears_acceptance(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    _wire_command(wrapper, lane, snapshot)
    safe = lane.wire_last_safe_command.effort.clone()
    lane.safety.project = lambda sample: SimpleNamespace(
        effort=sample.safe_effort.clone(), feasible=False,
        fallback_reason="hold_rejected")
    command = _wire_command(wrapper, lane, snapshot)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.phase is WireAcceptancePhase.TERMINATED
    assert torch.equal(command.effort, safe)
    assert torch.equal(lane.wire_last_safe_command.effort, safe)
    assert wrapper.wire_acceptance_diagnostics["rejection_reason"] == "safety_rejected"


def test_lane_reset_clears_acceptance_and_contact_latches(monkeypatch):
    module, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    wrapper._lane_hand_targets(lane, snapshot)
    wrapper.target_object_id, wrapper.obstacle_object_ids = "wire_000", ()
    robot = SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros((1, 13))))
    wrapper.env.unwrapped.scene = {"robot": robot}
    monkeypatch.setattr(module, "M1DualPandaO6SnapshotAdapter", lambda *_a, **_k: SimpleNamespace(snapshot=lambda: snapshot))
    wrapper.reset_lanes(torch.tensor([True]))
    assert wrapper.wire_acceptance_state.phase is WireAcceptancePhase.CONTACT
    assert wrapper.wire_acceptance_state.bilateral_contact_elapsed_s == 0
    assert torch.isnan(lane.left_hand_contact_q).all()
    assert torch.isnan(lane.right_hand_contact_q).all()
    assert lane.wire_last_safe_command is None
    assert lane.wire_hold_arm_q is None
    assert wrapper.wire_acceptance_diagnostics is None


def test_wire_hand_mpc_releases_historical_contact_and_returns_closing_target(monkeypatch):
    from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime = BimanualRuntime()
    wrapper._configure_wire_hand_reclamp(lane)
    lane.runtime.mission.phase = BimanualPhase.PRELOAD
    first = lane.runtime._hand_input(snapshot, "right", torch.zeros(6, dtype=torch.float64))
    lane.runtime.right_hand_mpc.plan(first)
    assert torch.isfinite(lane.runtime.right_hand_mpc.precontact._latched_contact_q[:3]).all()
    lane.runtime.mission.phase = BimanualPhase.LIFT
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.tensor([True, False, False, False, False])))
    second = lane.runtime._hand_input(lost, "right", torch.zeros(6, dtype=torch.float64))
    assert second.phase is BimanualPhase.PRELOAD
    solution = lane.runtime.right_hand_mpc.plan(second)
    assert solution.diagnostics.feasible
    assert torch.equal(solution.q_ref[:2], snapshot.right_hand.q[:2])
    assert solution.q_ref[2] > snapshot.right_hand.q[2]


def test_wire_gate_is_applied_before_environment_submission(monkeypatch):
    module, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    submitted = []
    targets = []
    robot = SimpleNamespace(set_joint_position_target=lambda target, **kwargs: targets.append(target.clone()))
    lane.adapter = SimpleNamespace(snapshot=lambda: snapshot, dynamics=lambda: None,
        teacher_task_jacobian=lambda: None, robot=robot, left_hand_ids=tuple(range(6)),
        right_hand_ids=tuple(range(6, 12)))
    lane.runtime.compute = lambda snap: lane.baseline_command
    wrapper._compute_lane_command = lambda *args: lane.baseline_command
    monkeypatch.setattr(module, "build_teacher_input", lambda **kwargs: None)
    wrapper.env.unwrapped.device = "cpu"
    wrapper.env.step = lambda action: submitted.append(action.clone())
    wrapper.startup_complete, wrapper._step = True, 0
    wrapper._effort_limits = torch.ones(43, dtype=torch.float64)
    wrapper._step_impl()
    assert len(submitted) == 1
    assert submitted[0].shape == (1, 43)
    assert torch.equal(submitted[0][0, 17:31], torch.zeros(14))
    assert targets[0].shape == targets[1].shape == (1, 6)
    assert wrapper.wire_acceptance_state.bilateral_contact_elapsed_s == pytest.approx(0.1)


def test_non_wire_gate_leaves_command_and_hand_behavior_unchanged(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.wire_acceptance_tracker = None
    assert _wire_command(wrapper, lane, snapshot) is lane.baseline_command
    assert wrapper.wire_acceptance_state is None
    left, right = wrapper._lane_hand_targets(lane, snapshot)
    assert torch.equal(left, torch.full((6,), 0.7, dtype=torch.float64))
    assert torch.equal(right, left)


def test_lift_loss_returns_to_preload_and_keeps_arm_hold_until_contact_dwell(monkeypatch):
    from go2_pvcnn.control.m1_bimanual_coordination.state_machine import BimanualMission
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission = BimanualMission()
    lane.runtime.mission.phase = BimanualPhase.LIFT
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.tensor([True, False, False, False, False])))
    wrapper._prepare_wire_reclamp(lane, lost)
    assert lane.runtime.mission.phase is BimanualPhase.PRELOAD
    command = _wire_command(wrapper, lane, lost)
    assert wrapper.wire_acceptance_state.request_reclamp
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    command = _wire_command(wrapper, lane, lost)
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    for _ in range(4):
        command = _wire_command(wrapper, lane, snapshot)
        assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    _wire_command(wrapper, lane, snapshot)
    command = _wire_command(wrapper, lane, snapshot)
    assert lane.wire_hold_arm_q is None


def _bound_acceptance_wrapper(monkeypatch, *, loss_budget_s=0.5, dt_s=0.1,
                              initial_phase=BimanualPhase.PRELOAD):
    from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
    from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import build_wire_station_contact_goal

    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    wrapper.env.unwrapped.physics_dt = dt_s
    goal = build_wire_station_contact_goal(_geometry().centerline)
    for side in ("left", "right"):
        getattr(snapshot, f"{side}_hand").fingertip_forces_b[:2] = 0.4 * torch.tensor(
            [target.normal for target in getattr(goal, f"{side}_contact_targets")], dtype=torch.float64)
    lane.runtime, backend = _staged_runtime()
    lane.runtime.left_hand_mpc = O6HandMpc()
    lane.runtime.right_hand_mpc = O6HandMpc()
    wrapper._configure_wire_hand_reclamp(lane)
    lane.runtime.mission.phase = initial_phase
    lane.wire_acceptance_tracker = WireAcceptanceTracker(WireAcceptanceConfig(
        required_bilateral_contact_s=loss_budget_s))
    lane.wire_acceptance_tracker.reset(wire_center_height_m=1.2)
    for _ in range(6):
        command = _bound_acceptance_tick(wrapper, lane, snapshot)
        assert command.feasible, wrapper.wire_acceptance_diagnostics
    assert lane.runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND
    assert wrapper.wire_acceptance_state.lift_permitted == (6 * dt_s >= loss_budget_s)
    assert backend.bind_calls == 1
    lane.runtime.mission.phase = BimanualPhase.LIFT
    return wrapper, lane, snapshot, backend


def _bound_acceptance_tick(wrapper, lane, snapshot):
    snapshot = replace(snapshot, timestamp_ns=round(
        (lane.runtime._step + 1) * wrapper.env.unwrapped.physics_dt * 1e9))
    wrapper._prepare_wire_reclamp(lane, snapshot)
    lane.baseline_command = lane.runtime.compute(snapshot)
    lane.last_safety_result = SimpleNamespace(feasible=True)
    return _wire_command(wrapper, lane, snapshot)


def test_backend_rejection_survives_rebuilt_teacher_command_and_suppresses_lift(monkeypatch):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(monkeypatch)
    lane.runtime.left_hand_mpc = _HandController()
    lane.runtime.right_hand_mpc = _HandController()
    lane.runtime.left_hand_mpc._fingertip_target_contact_latch = torch.zeros(5, dtype=torch.bool)
    lane.runtime.right_hand_mpc._fingertip_target_contact_latch = torch.zeros(5, dtype=torch.bool)
    lane.runtime._last_left_hand = lane.runtime._last_right_hand = None
    backend.fail = "step"
    snapshot = replace(snapshot, timestamp_ns=round(
        (lane.runtime._step + 1) * wrapper.env.unwrapped.physics_dt * 1e9))

    lane.baseline_command = lane.runtime.compute(snapshot)
    assert not lane.baseline_command.feasible
    runtime_reason = lane.baseline_command.fallback_reasons[0]
    assert lane.runtime.latest_solutions["object"].diagnostics.feasible
    assert lane.runtime.latest_solutions["arm"].both_feasible
    assert lane.runtime.latest_solutions["left_hand"].diagnostics.feasible
    assert lane.runtime.latest_solutions["right_hand"].diagnostics.feasible
    assert lane.runtime.latest_solutions["wbc"].feasible
    lane.teacher = SimpleNamespace(
        fixed_base=True,
        plan=lambda _input: SimpleNamespace(diagnostics=SimpleNamespace(feasible=True)),
    )
    wrapper.mode = "teacher"
    wrapper._step = 0

    rebuilt = wrapper._compute_lane_command(lane, snapshot, None, object())
    command = wrapper._apply_wire_acceptance(lane, snapshot, None, rebuilt)

    assert not rebuilt.feasible
    assert rebuilt.fallback_reasons == (runtime_reason,)
    assert not command.feasible
    assert command.fallback_reasons == (runtime_reason, "safety_rejected")
    assert wrapper.wire_acceptance_state.terminal_reason == "safety_rejected"
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert command.effort.shape == (43,)


def test_backend_rejection_survives_real_latent_command_and_suppresses_lift(monkeypatch):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(monkeypatch)
    lane.runtime.left_hand_mpc = _HandController()
    lane.runtime.right_hand_mpc = _HandController()
    lane.runtime.left_hand_mpc._fingertip_target_contact_latch = torch.zeros(5, dtype=torch.bool)
    lane.runtime.right_hand_mpc._fingertip_target_contact_latch = torch.zeros(5, dtype=torch.bool)
    lane.runtime._last_left_hand = lane.runtime._last_right_hand = None
    backend.fail = "step"
    snapshot = replace(snapshot, timestamp_ns=round(
        (lane.runtime._step + 1) * wrapper.env.unwrapped.physics_dt * 1e9))

    lane.baseline_command = lane.runtime.compute(snapshot)
    assert not lane.baseline_command.feasible
    runtime_reason = lane.baseline_command.fallback_reasons[0]
    assert lane.runtime.latest_solutions["object"].diagnostics.feasible
    assert lane.runtime.latest_solutions["arm"].both_feasible
    assert lane.runtime.latest_solutions["left_hand"].diagnostics.feasible
    assert lane.runtime.latest_solutions["right_hand"].diagnostics.feasible
    assert lane.runtime.latest_solutions["wbc"].feasible

    lane.latent_runtime = LatentRuntime(
        model=_Model(),
        normalizer=LatentNormalizer(
            mean=torch.zeros(111), scale=torch.ones(111)
        ),
        teacher=_Teacher(),
        safety=_Safety(),
        effort_limits=100.0 * torch.ones(43, dtype=torch.float64),
        safety_input_provider=lambda _snapshot, _dynamics, candidate: {
            "candidate": candidate
        },
    )
    wrapper.mode = "latent"

    rebuilt = wrapper._compute_lane_command(
        lane, snapshot, _teacher_input().dynamics, _teacher_input()
    )
    command = wrapper._apply_wire_acceptance(lane, snapshot, None, rebuilt)

    assert not rebuilt.feasible
    assert rebuilt.fallback_reasons == (runtime_reason,)
    assert not command.feasible
    assert command.fallback_reasons == (runtime_reason, "safety_rejected")
    assert wrapper.wire_acceptance_state.terminal_reason == "safety_rejected"
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert command.effort.shape == (43,)


@pytest.mark.parametrize("loss", ["mask", "force", "inconsistent"])
def test_bound_backend_contact_loss_pauses_and_resumes_after_fresh_dwell(monkeypatch, loss):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(monkeypatch)
    attachment, step = lane.runtime._wire_attachment, lane.runtime.latest_dlo_step
    steps = backend.step_calls
    changes = ({"contact_mask": torch.tensor([True, False, False, False, False])}
               if loss == "mask" else {"fingertip_forces_b": torch.zeros((5, 3), dtype=torch.float64)})
    if loss == "inconsistent":
        changes = {"contact_consistent": False}
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand, **changes))

    command = _bound_acceptance_tick(wrapper, lane, lost)

    assert command.feasible
    assert wrapper.wire_acceptance_state.request_reclamp
    assert wrapper.wire_acceptance_state.terminal_reason is None
    assert lane.runtime.mission.phase is BimanualPhase.PRELOAD
    assert lane.runtime.wire_coupling_state.phase.name == "RECLAMPING"
    assert backend.step_calls == steps
    assert lane.runtime._wire_attachment is attachment
    assert lane.runtime.latest_dlo_step is step
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    for _ in range(4):
        command = _bound_acceptance_tick(wrapper, lane, snapshot)
        assert command.feasible
        assert not wrapper.wire_acceptance_state.lift_permitted
        assert backend.step_calls == steps
        assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    _bound_acceptance_tick(wrapper, lane, snapshot)
    assert wrapper.wire_acceptance_state.lift_permitted
    assert backend.step_calls == steps
    command = _bound_acceptance_tick(wrapper, lane, snapshot)
    assert command.feasible
    assert lane.runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND
    assert backend.step_calls == steps + 1
    assert backend.bind_calls == 1
    assert lane.runtime._wire_attachment is attachment
    assert lane.wire_hold_arm_q is None


@pytest.mark.parametrize("loss_budget_s", [0.3, 0.5])
def test_bound_backend_persistent_loss_terminates_at_configured_boundary(monkeypatch, loss_budget_s):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(
        monkeypatch, loss_budget_s=loss_budget_s)
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.zeros(5, dtype=torch.bool)))
    steps = backend.step_calls
    for _ in range(round(loss_budget_s / 0.1) - 1):
        command = _bound_acceptance_tick(wrapper, lane, lost)
        assert command.feasible
        assert wrapper.wire_acceptance_state.terminal_reason is None
    command = _bound_acceptance_tick(wrapper, lane, lost)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "persistent_contact_loss"
    assert not wrapper.wire_acceptance_state.success
    for _ in range(6):
        assert not _bound_acceptance_tick(wrapper, lane, snapshot).feasible
    assert backend.step_calls == steps
    assert backend.bind_calls == 1


@pytest.mark.parametrize("initial_phase", [BimanualPhase.PRELOAD, BimanualPhase.APPROACH])
def test_bound_loss_before_initial_dwell_still_has_recovery_timeout(monkeypatch, initial_phase):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(
        monkeypatch, dt_s=0.01, initial_phase=initial_phase)
    assert not wrapper.wire_acceptance_state.lift_permitted
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.zeros(5, dtype=torch.bool)))
    steps = backend.step_calls
    for _ in range(49):
        command = _bound_acceptance_tick(wrapper, lane, lost)
        assert command.feasible
        assert wrapper.wire_acceptance_state.request_reclamp
        assert wrapper.wire_acceptance_state.terminal_reason is None
    command = _bound_acceptance_tick(wrapper, lane, lost)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "persistent_contact_loss"
    assert backend.step_calls == steps
    assert backend.bind_calls == 1


@pytest.mark.parametrize("failure", ["wbc", "slip", "backend"])
def test_bound_backend_recovery_keeps_other_failures_terminal(monkeypatch, failure):
    wrapper, lane, snapshot, backend = _bound_acceptance_wrapper(monkeypatch)
    steps = backend.step_calls
    lost = replace(snapshot, right_hand=replace(snapshot.right_hand,
        contact_mask=torch.zeros(5, dtype=torch.bool)))
    if failure == "wbc":
        lane.runtime.wbc.fail_at = lane.runtime.wbc.calls + 1
    elif failure == "slip":
        lost.left_arm.palm_pose_b[0] += 0.021
    else:
        backend.fail = "step"
        lost = snapshot
    command = _bound_acceptance_tick(wrapper, lane, lost)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == (
        "excessive_endpoint_slip" if failure == "slip" else "safety_rejected")
    assert not wrapper.wire_acceptance_state.success
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    for _ in range(6):
        assert not _bound_acceptance_tick(wrapper, lane, snapshot).feasible
    assert backend.step_calls == steps + (failure == "backend")
    assert backend.bind_calls == 1


def test_terminal_slip_does_not_publish_previous_upward_command(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    for _ in range(6):
        _wire_command(wrapper, lane, snapshot)
    previous = lane.wire_last_safe_command.effort.clone()
    pose = snapshot.left_arm.palm_pose_b.clone()
    pose[0] += 0.03
    slipped = replace(snapshot, left_arm=replace(snapshot.left_arm, palm_pose_b=pose))
    command = _wire_command(wrapper, lane, slipped)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "excessive_endpoint_slip"
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert torch.equal(lane.wire_last_safe_command.effort, previous)


@pytest.mark.parametrize("phase", [BimanualPhase.APPROACH, BimanualPhase.PRELOAD, BimanualPhase.GRASP])
def test_non_finite_contact_holds_arms_on_first_rejected_tick(monkeypatch, phase):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission.phase = phase
    snapshot.left_hand.fingertip_forces_b[0, 0] = float("nan")
    command = _wire_command(wrapper, lane, snapshot)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "non_finite_contact_observation"
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))
    assert lane.wire_last_safe_command is None


def test_slip_during_grasp_holds_arms_on_first_rejected_tick(monkeypatch):
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    lane.runtime.mission.phase = BimanualPhase.GRASP
    _wire_command(wrapper, lane, snapshot)
    snapshot.left_arm.palm_pose_b[0] += 0.03
    command = _wire_command(wrapper, lane, snapshot)
    assert not command.feasible
    assert wrapper.wire_acceptance_state.terminal_reason == "excessive_endpoint_slip"
    assert torch.equal(command.effort[17:31], torch.zeros(14, dtype=torch.float64))


def _load_verifier():
    spec = importlib.util.spec_from_file_location("wire_contact_report_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostics(
    state: str,
    *,
    transition_reason: str | None = None,
    left_contact_ticks: int = 0,
    right_contact_ticks: int = 0,
    bind_count: int = 0,
    deformx_elapsed_time: float = 0.0,
) -> WireCouplingDiagnostics:
    return WireCouplingDiagnostics(
        object_id="wire_000",
        target_prim_path="{ENV_REGEX_NS}/wire_000",
        state=state,
        transition_reason=transition_reason or f"entered_{state.lower()}",
        centerline_midpoint_b=(0.5, 0.0, 1.2),
        tangent_b=(1.0, 0.0, 0.0),
        radius_m=0.004,
        left_station_b=(0.476, 0.0, 1.2),
        right_station_b=(0.524, 0.0, 1.2),
        left_pinch_error_m=0.001,
        right_pinch_error_m=0.002,
        left_contact_mask=(True, True, False, False, False),
        right_contact_mask=(True, True, False, False, False),
        left_contact_ticks=left_contact_ticks,
        right_contact_ticks=right_contact_ticks,
        bind_count=bind_count,
        deformx_elapsed_time=deformx_elapsed_time,
        wbc_feasible=True,
        wire_goal_committed=True,
    )


def _valid_report() -> dict[str, object]:
    transitions = [
        _diagnostics("SCENE_TRACKED").to_json(),
        _diagnostics("APPROACH_ENDPOINTS").to_json(),
        _diagnostics(
            "CONTACT_LATCHING",
            left_contact_ticks=1,
            right_contact_ticks=1,
        ).to_json(),
        _diagnostics(
            "COSSERAT_BOUND",
            left_contact_ticks=5,
            right_contact_ticks=5,
            bind_count=1,
        ).to_json(),
    ]
    return {"schema_version": 1, "transitions": transitions, "final": transitions[-1]}


def test_diagnostics_are_frozen_and_json_safe():
    diagnostics = _diagnostics(
        "COSSERAT_BOUND",
        left_contact_ticks=5,
        right_contact_ticks=5,
        bind_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        diagnostics.bind_count = 2

    payload = diagnostics.to_json()
    required = {
        "object_id",
        "target_prim_path",
        "state",
        "transition_reason",
        "centerline_midpoint_b",
        "tangent_b",
        "radius_m",
        "left_station_b",
        "right_station_b",
        "left_pinch_error_m",
        "right_pinch_error_m",
        "left_contact_mask",
        "right_contact_mask",
        "left_contact_ticks",
        "right_contact_ticks",
        "bind_count",
        "deformx_elapsed_time",
        "wbc_feasible",
        "wire_goal_committed",
    }
    assert required <= payload.keys()
    assert isinstance(payload["centerline_midpoint_b"], list)
    assert isinstance(payload["left_contact_mask"], list)
    json.dumps(payload, allow_nan=False)


def test_diagnostics_reject_nonfinite_values_before_json_publication():
    with pytest.raises(ValueError, match="finite"):
        replace(_diagnostics("SCENE_TRACKED"), radius_m=float("nan"))


def test_runtime_publishes_immutable_complete_diagnostics_without_prebind_elapsed_time():
    runtime, backend = _staged_runtime()

    _wire_tick(runtime, 1, left=False, right=False)

    diagnostics = runtime.latest_wire_diagnostics
    assert isinstance(diagnostics, WireCouplingDiagnostics)
    assert diagnostics.state == "APPROACH_ENDPOINTS"
    assert diagnostics.deformx_elapsed_time == 0.0
    assert backend.bind_calls == backend.step_calls == 0
    json.dumps(diagnostics.to_json(), allow_nan=False)


def test_rejected_tick_refreshes_contact_masks_and_candidate_errors():
    runtime, backend = _staged_runtime(_WbcController(fail_at=2))
    _wire_tick(runtime, 1, left=False, right=False)
    previous = runtime.latest_wire_diagnostics
    snapshot = replace(_contact_snapshot(left=True, right=False), timestamp_ns=2)
    snapshot.left_hand.fingertip_positions_b[:, 0] += 0.02
    snapshot.right_hand.fingertip_positions_b[:, 0] += 0.03

    command = runtime.compute(snapshot)

    assert not command.feasible
    diagnostics = runtime.latest_wire_diagnostics
    assert diagnostics is not previous
    assert not diagnostics.wbc_feasible
    assert not diagnostics.wire_goal_committed
    assert diagnostics.state == runtime.wire_coupling_state.phase.name
    assert diagnostics.transition_reason == "fake_wbc_failure"
    assert diagnostics.left_contact_mask == (True, True, False, False, False)
    assert diagnostics.right_contact_mask == (False,) * 5
    assert diagnostics.left_pinch_error_m == pytest.approx(0.02)
    assert diagnostics.right_pinch_error_m == pytest.approx(0.03)
    assert diagnostics.left_contact_ticks == diagnostics.right_contact_ticks == 0
    assert backend.bind_calls == backend.step_calls == 0
    json.dumps(diagnostics.to_json(), allow_nan=False)


@pytest.mark.parametrize("failure_tick", [1, 5, 6, 9])
def test_diagnostics_constructor_failure_cannot_interrupt_control_commit(monkeypatch, failure_tick):
    runtime, backend = _staged_runtime()
    for tick in range(1, failure_tick):
        _wire_tick(runtime, tick)
    previous = runtime.latest_wire_diagnostics
    observed_command = []

    def fail_diagnostics(**_fields):
        observed_command.append(runtime._last_command)
        raise ArithmeticError("forced diagnostics normalization failure")

    monkeypatch.setattr(
        "go2_pvcnn.control.m1_bimanual_coordination.runtime.WireCouplingDiagnostics",
        fail_diagnostics,
    )
    command = _wire_tick(runtime, failure_tick)

    assert command.feasible
    assert runtime._last_command is command
    assert len(observed_command) == 1 and observed_command[0] is command
    assert runtime._last_object is runtime._latest_solutions["object"]
    assert runtime._last_arm is runtime._latest_solutions["arm"]
    assert runtime._last_left_hand is runtime._latest_solutions["left_hand"]
    assert runtime._last_right_hand is runtime._latest_solutions["right_hand"]
    assert runtime.latest_wire_goal is runtime._active_wire_goal
    assert runtime.latest_wire_goal is not None
    assert runtime._last_snapshot.timestamp_ns == failure_tick
    assert runtime._step == failure_tick
    assert runtime.wire_coupling_state.left_contact_ticks == min(failure_tick, 5)
    assert backend.bind_calls == int(failure_tick >= 5)
    assert backend.step_calls == max(0, failure_tick - 5)
    if failure_tick >= 5:
        assert runtime._wire_attachment is not None
        assert runtime.latest_dlo_step is not None
        assert runtime.wire_coupling_state.phase.name == "COSSERAT_BOUND"
    if previous is not None:
        assert runtime.latest_wire_diagnostics is previous
    else:
        assert not runtime.latest_wire_diagnostics.wire_goal_committed
    json.dumps(runtime.latest_wire_diagnostics.to_json(), allow_nan=False)


@pytest.mark.parametrize("first_tick", [True, False])
def test_geometry_failure_still_publishes_current_tick_diagnostics(first_tick):
    runtime, backend = _staged_runtime()
    if not first_tick:
        _wire_tick(runtime, 1)
    previous = runtime.latest_wire_diagnostics

    def invalid_geometry(_snapshot):
        raise ValueError("invalid geometry")

    runtime.wire_geometry_provider = invalid_geometry
    command = _wire_tick(runtime, 2, left=False, right=False)

    assert not command.feasible
    diagnostics = runtime.latest_wire_diagnostics
    assert diagnostics is not None and diagnostics is not previous
    assert diagnostics.state == "SAFE_HOLD"
    assert diagnostics.wbc_feasible
    assert not diagnostics.wire_goal_committed
    assert "wire_geometry_failure" in diagnostics.transition_reason
    assert diagnostics.left_contact_mask == diagnostics.right_contact_mask == (False,) * 5
    assert backend.bind_calls == backend.step_calls == 0
    json.dumps(diagnostics.to_json(), allow_nan=False)


def test_report_recorder_keeps_only_state_transitions_and_latest_final():
    recorder = _WireReportRecorder()
    recorder.record(_diagnostics("SCENE_TRACKED"))
    recorder.record(_diagnostics("SCENE_TRACKED", transition_reason="state_unchanged"))
    recorder.record(_diagnostics("APPROACH_ENDPOINTS"))
    latest = _diagnostics("APPROACH_ENDPOINTS", transition_reason="state_unchanged")
    recorder.record(latest)

    report = recorder.to_json()

    assert [item["state"] for item in report["transitions"]] == [
        "SCENE_TRACKED",
        "APPROACH_ENDPOINTS",
    ]
    assert report["final"] == latest.to_json()
    json.dumps(report, allow_nan=False)


def test_wire_report_is_replaced_atomically(tmp_path: Path):
    destination = tmp_path / "nested" / "wire.json"
    destination.parent.mkdir()
    destination.write_text('{"old": true}\n', encoding="utf-8")

    report = _valid_report()
    _write_wire_report_atomic(destination, report)

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert list(destination.parent.iterdir()) == [destination]


def test_validator_accepts_complete_ordered_wire_report():
    verifier = _load_verifier()

    assert verifier.validate_report(_valid_report()) is None


@pytest.mark.parametrize("bound_index", [-1, -2])
def test_validator_rejects_unlatched_bound_record_even_when_final_has_five_ticks(bound_index):
    verifier = _load_verifier()
    report = _valid_report()
    report["final"] = dict(report["final"])
    report["transitions"].append(dict(report["final"]))
    report["transitions"][bound_index].update(left_contact_ticks=0, right_contact_ticks=0)

    with pytest.raises(ValueError, match="contact_ticks"):
        verifier.validate_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["final"].__setitem__("object_id", "box_000"),
            "object_id",
        ),
        (
            lambda report: report["final"].__setitem__(
                "target_prim_path", "{ENV_REGEX_NS}/Box"
            ),
            "target_prim_path",
        ),
        (
            lambda report: report["transitions"].pop(1),
            "ordered",
        ),
        (
            lambda report: report["final"].__setitem__("radius_m", float("nan")),
            "finite",
        ),
        (
            lambda report: report["final"].__setitem__("bind_count", 2),
            "bind_count",
        ),
        (
            lambda report: (
                report.__setitem__("final", dict(report["final"])),
                report["transitions"][-1].__setitem__("bind_count", 0),
            ),
            "bind_count",
        ),
        (
            lambda report: report["final"].__setitem__(
                "left_pinch_error_m", 0.0050001
            ),
            "pinch",
        ),
        (
            lambda report: report["final"].__setitem__(
                "right_pinch_error_m", 0.0050001
            ),
            "pinch",
        ),
        (
            lambda report: report["final"].__setitem__("left_contact_ticks", 4),
            "contact_ticks",
        ),
        (
            lambda report: report["final"].__setitem__("right_contact_ticks", 4),
            "contact_ticks",
        ),
        (
            lambda report: report["transitions"][1].__setitem__(
                "deformx_elapsed_time", 1.0e-6
            ),
            "pre-bind",
        ),
    ],
)
def test_validator_rejects_invalid_wire_reports(mutate, message):
    verifier = _load_verifier()
    report = _valid_report()
    mutate(report)

    with pytest.raises(ValueError, match=message):
        verifier.validate_report(report)


def test_validator_rejects_wrong_object_or_prim_in_transition_history():
    verifier = _load_verifier()
    report = _valid_report()
    report["transitions"][0]["object_id"] = "wire_001"

    with pytest.raises(ValueError, match="object_id"):
        verifier.validate_report(report)
