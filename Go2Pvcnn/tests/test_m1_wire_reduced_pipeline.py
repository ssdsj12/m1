from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.contracts import (
    BimanualCommand,
    BimanualPhase,
)
from go2_pvcnn.control.m1_bimanual_coordination.diagnostics import (
    WirePipelineDiagnostics,
)
from go2_pvcnn.control.m1_bimanual_coordination.grasp_lift_pipeline import (
    BimanualGraspLiftPipeline,
    WirePipelineConfig,
)
from tests.test_m1_bimanual_object_mpc import _snapshot
from tests.test_m1_bimanual_grasp_lift_pipeline import _goal
from tests.test_m1_bimanual_whole_body_qp import _request
from go2_pvcnn.control.m1_bimanual_coordination.constraints import WireGateConfig, subsolution_failure_reason
from go2_pvcnn.control.m1_bimanual_coordination.state_machine import BimanualMission


class _Runtime:
    def __init__(self, snapshot):
        self.phase = BimanualPhase.APPROACH
        self.latest_solutions = {}
        self._snapshot = snapshot
        self._effort = torch.zeros(43, dtype=torch.float64)
        self.wire_diagnostics = None

    def compute(self, snapshot):
        self._snapshot = snapshot
        return BimanualCommand(snapshot.timestamp_ns, self._effort, True, ())

    def reset(self):
        self.phase = BimanualPhase.APPROACH
        self.wire_diagnostics = None


def _diag(**changes):
    values = dict(
        phase="APPROACH",
        bilateral_contact=False,
        left_contact_mask=(False,) * 5,
        right_contact_mask=(False,) * 5,
        left_clamp_force_n=0.0,
        right_clamp_force_n=0.0,
        force_closure_margin=0.0,
        slip_m=0.0,
        wire_feasible=True,
        arm_recoverability=True,
        command_accepted=True,
        terminal_reason=None,
    )
    values.update(changes)
    return WirePipelineDiagnostics(**values)


def test_pipeline_latches_each_digit_and_requires_bilateral_contacts():
    snapshot = _snapshot()
    runtime = _Runtime(snapshot)
    pipeline = BimanualGraspLiftPipeline(
        runtime=runtime, wire_config=WirePipelineConfig(contact_dwell_steps=1)
    )
    runtime.wire_diagnostics = _diag(
        phase="PRELOAD", bilateral_contact=False,
        left_contact_mask=(True, True, False, False, False),
        right_contact_mask=(False,) * 5,
    )
    first = pipeline.step(snapshot)
    assert first.phase is BimanualPhase.PRELOAD
    assert first.fallback and first.fallback_reason == "bilateral_contact_required"
    runtime.wire_diagnostics = _diag(
        phase="GRASP", bilateral_contact=True,
        left_contact_mask=(True, True, False, False, False),
        right_contact_mask=(True, True, False, False, False),
        left_clamp_force_n=1.0, right_clamp_force_n=1.0,
        force_closure_margin=1.0,
    )
    second = pipeline.step(replace(snapshot, timestamp_ns=snapshot.timestamp_ns + 1))
    assert second.phase is BimanualPhase.GRASP
    assert not second.fallback
    assert second.diagnostics.left_latched and second.diagnostics.right_latched


def test_pipeline_pauses_lift_on_contact_loss_or_slip_atomically():
    snapshot = _snapshot()
    runtime = _Runtime(snapshot)
    pipeline = BimanualGraspLiftPipeline(runtime=runtime)
    runtime.phase = BimanualPhase.LIFT
    runtime.wire_diagnostics = _diag(
        phase="LIFT", bilateral_contact=False, wire_feasible=True,
        left_clamp_force_n=1.0, right_clamp_force_n=1.0,
    )
    result = pipeline.step(snapshot)
    assert result.fallback_reason == "contact_lost_during_lift"
    assert not result.command.feasible
    runtime.wire_diagnostics = _diag(
        phase="LIFT", bilateral_contact=True, slip_m=0.03,
        left_contact_mask=(True, True, False, False, False),
        right_contact_mask=(True, True, False, False, False),
        left_clamp_force_n=1.0, right_clamp_force_n=1.0,
    )
    result = pipeline.step(replace(snapshot, timestamp_ns=snapshot.timestamp_ns + 1))
    assert result.fallback_reason == "wire_slip_limit"


def test_diagnostics_json_is_deterministic_and_machine_readable():
    value = _diag(phase="HOLD", bilateral_contact=True,
                  left_contact_mask=(True, True, False, False, False),
                  right_contact_mask=(True, True, False, False, False),
                  left_clamp_force_n=1.0, right_clamp_force_n=1.0,
                  force_closure_margin=1.0)
    assert value.to_json() == value.to_json()
    assert value.to_json()["terminal_reason"] is None


def _clamped(**changes):
    values = dict(phase="GRASP", bilateral_contact=True,
                  left_contact_mask=(True, True, False, False, False),
                  right_contact_mask=(True, True, False, False, False),
                  left_clamp_force_n=1.0, right_clamp_force_n=1.0,
                  force_closure_margin=1.0)
    values.update(changes)
    return _diag(**values)


@pytest.mark.parametrize("phase", list(BimanualPhase))
def test_rigid_object_diagnostics_do_not_activate_wire_gate(phase):
    snapshot = _snapshot()
    runtime = _Runtime(snapshot)
    runtime.phase = phase
    runtime.latest_solutions = {"object": _request().object_solution}
    result = BimanualGraspLiftPipeline(runtime=runtime).step(snapshot)
    assert result.diagnostics is None
    assert result.command.feasible
    assert result.phase is phase


def test_first_wire_rejection_returns_zero_effort():
    runtime = _Runtime(_snapshot())
    runtime._effort.fill_(7.0)
    runtime.wire_diagnostics = _diag(phase="LIFT")
    result = BimanualGraspLiftPipeline(runtime=runtime).step(_snapshot())
    assert not result.command.feasible
    assert torch.count_nonzero(result.command.effort) == 0


@pytest.mark.parametrize("phase", ["LIFT", "HOLD"])
@pytest.mark.parametrize("failure", ["contact", "slip", "dynamics", "arm"])
def test_lift_rejection_requests_real_preload_and_clears_hold(phase, failure):
    snapshot = _snapshot()
    runtime = _Runtime(snapshot)
    runtime.mission = BimanualMission()
    runtime.mission.phase = BimanualPhase[phase]
    runtime.mission._hold_elapsed_ns = 700_000_000
    changes = {"phase": phase}
    if failure == "contact":
        changes.update(bilateral_contact=False, right_contact_mask=(False,) * 5)
    elif failure == "slip":
        changes["slip_m"] = 0.03
    elif failure == "dynamics":
        changes["wire_feasible"] = False
    else:
        changes["arm_recoverability"] = False
    runtime.wire_diagnostics = _clamped(**changes)
    result = BimanualGraspLiftPipeline(runtime=runtime).step(snapshot)
    assert result.phase is BimanualPhase.PRELOAD
    assert result.request_reclamp and not result.lift_permitted
    assert runtime.mission.phase is BimanualPhase.PRELOAD
    assert runtime.mission._hold_elapsed_ns == 0
    assert not result.success


def test_preload_latches_digits_only_after_configured_consecutive_dwell():
    runtime = _Runtime(_snapshot())
    pipeline = BimanualGraspLiftPipeline(runtime=runtime,
        wire_config=WirePipelineConfig(contact_dwell_steps=3))
    runtime.wire_diagnostics = _clamped(phase="PRELOAD")
    for index in range(3):
        result = pipeline.step(replace(_snapshot(), timestamp_ns=index + 1))
        assert result.diagnostics.left_latched is (index == 2)
        assert result.diagnostics.right_latched is (index == 2)
    # A latched digit survives a transient loss during closing, but current
    # contacts are still required for GRASP/LIFT.
    runtime.wire_diagnostics = _diag(phase="PRELOAD")
    result = pipeline.step(replace(_snapshot(), timestamp_ns=4))
    assert result.diagnostics.left_latched and result.diagnostics.right_latched
    pipeline.reset()
    runtime.wire_diagnostics = _clamped(phase="GRASP")
    result = pipeline.step(replace(_snapshot(), timestamp_ns=5))
    assert result.fallback_reason == "per_digit_contact_latch_required"


@pytest.mark.parametrize("bad", [1, "false", None, float("nan")])
def test_wire_diagnostics_reject_non_boolean_contact_masks(bad):
    with pytest.raises(TypeError):
        _diag(left_contact_mask=(bad, False, False, False, False))


def test_wire_diagnostics_reject_unknown_phase_and_contradictory_contact():
    with pytest.raises(ValueError):
        _diag(phase="TYPO")
    with pytest.raises(ValueError):
        _diag(bilateral_contact=True)


def _native_runtime(snapshot):
    runtime = _Runtime(snapshot)
    runtime.phase = BimanualPhase.GRASP
    runtime.latest_solutions = {"object": SimpleNamespace(diagnostics=SimpleNamespace(
        force_closure_margin=1.0, wire_lift_feasible=True,
        wire_friction_feasible=True, wire_slip_max_m=0.0))}
    return runtime


def _force_snapshot(left, right, mask=(True, True, False, False, False)):
    snapshot = _snapshot()
    return replace(snapshot,
        left_hand=replace(snapshot.left_hand, contact_mask=torch.tensor(mask),
            fingertip_forces_b=torch.tensor(left, dtype=torch.float64)),
        right_hand=replace(snapshot.right_hand, contact_mask=torch.tensor(mask),
            fingertip_forces_b=torch.tensor(right, dtype=torch.float64)))


def test_native_wire_contact_uses_selected_force_magnitudes_without_goal():
    forces = [[1., 0., 0.], [1., 0., 0.], [0., 0., 100.], [0., 0., 0.], [0., 0., 0.]]
    snapshot = _force_snapshot(forces, forces)
    result = BimanualGraspLiftPipeline(runtime=_native_runtime(snapshot)).step(snapshot)
    assert not result.fallback
    assert result.diagnostics.left_clamp_force_n == 2.0


def test_zero_force_contact_digit_cannot_be_hidden_by_aggregate_force():
    forces = [[0., 0., 2.], [0., 0., 0.], [0., 0., 100.], [0., 0., 0.], [0., 0., 0.]]
    snapshot = _force_snapshot(forces, forces)
    result = BimanualGraspLiftPipeline(runtime=_native_runtime(snapshot)).step(snapshot)
    assert result.fallback
    assert result.diagnostics.left_contact_count == 1


@pytest.mark.parametrize("force,tangent,accepted", [(1., 0., False), (2., 0., True), (6., 0., False), (2., 5., False)])
def test_goal_normal_force_range_and_alignment_gate_wire(force, tangent, accepted):
    left = [[force, 0., tangent]] * 2 + [[0., 0., 100.]] * 3
    right = [[-force, 0., tangent]] * 2 + [[0., 0., 100.]] * 3
    snapshot = _force_snapshot(left, right)
    result = BimanualGraspLiftPipeline(_goal(), runtime=_native_runtime(snapshot)).step(snapshot)
    assert result.command.feasible is accepted
    assert result.diagnostics.left_clamp_force_n == pytest.approx(2 * force)


@pytest.mark.parametrize("contacts", [1, 2])
def test_lower_wire_gate_requires_two_valid_contacts_per_hand(contacts):
    request = _request()
    mask = torch.tensor([i < contacts for i in range(5)])
    forces = torch.zeros(5, 3, dtype=torch.float64)
    forces[:contacts, 0] = 1.0
    snapshot = replace(request.snapshot,
        left_hand=replace(request.snapshot.left_hand, contact_mask=mask, fingertip_forces_b=forces),
        right_hand=replace(request.snapshot.right_hand, contact_mask=mask, fingertip_forces_b=forces))
    request = replace(request, snapshot=snapshot, object_solution=replace(request.object_solution,
        diagnostics=replace(request.object_solution.diagnostics, wire_lift_feasible=True)))
    assert subsolution_failure_reason(request) == ("per_digit_contact_latch_required" if contacts == 1 else None)


@pytest.mark.parametrize("force,contacts,slip,reason", [
    (0.5, 3, 0., "per_digit_contact_latch_required"),
    (2., 2, 0., "per_digit_contact_latch_required"),
    (2., 3, .015, "wire_slip_limit"),
    (2., 3, 0., None),
])
def test_lower_wire_gate_consumes_configured_thresholds(force, contacts, slip, reason):
    request = _request()
    forces = torch.zeros(5, 3, dtype=torch.float64)
    forces[:contacts, 0] = force
    mask = torch.tensor([i < contacts for i in range(5)])
    request = replace(request,
        wire_gate_config=WireGateConfig(min_contact_force_n=1., min_contacts_per_hand=3, max_slip_m=.01),
        snapshot=replace(request.snapshot,
            left_hand=replace(request.snapshot.left_hand, contact_mask=mask, fingertip_forces_b=forces),
            right_hand=replace(request.snapshot.right_hand, contact_mask=mask, fingertip_forces_b=forces)),
        object_solution=replace(request.object_solution, diagnostics=replace(
            request.object_solution.diagnostics, wire_lift_feasible=True, wire_slip_max_m=slip)))
    assert subsolution_failure_reason(request) == reason


def test_interrupted_preload_dwell_is_per_digit():
    runtime = _Runtime(_snapshot())
    pipeline = BimanualGraspLiftPipeline(runtime=runtime,
        wire_config=WirePipelineConfig(contact_dwell_steps=2))
    for index, mask in enumerate(((True, False, False, False, False),
                                  (False, True, False, False, False),
                                  (True, True, False, False, False))):
        runtime.wire_diagnostics = _clamped(phase="PRELOAD", left_contact_mask=mask)
        result = pipeline.step(replace(_snapshot(), timestamp_ns=index + 1))
    assert not result.diagnostics.left_latched
    assert result.diagnostics.right_latched
    result = pipeline.step(replace(_snapshot(), timestamp_ns=4))
    assert result.diagnostics.left_latched


def test_rejected_command_preserves_both_last_safe_effort_and_hand_targets():
    runtime = _Runtime(_snapshot())
    pipeline = BimanualGraspLiftPipeline(runtime=runtime)
    runtime.wire_diagnostics = _clamped()
    safe_target = torch.ones(20, 6, dtype=torch.float64)
    runtime.latest_solutions = {"left_hand": SimpleNamespace(q_ref=safe_target),
                                "right_hand": SimpleNamespace(q_ref=safe_target * 2)}
    runtime._effort.fill_(3.)
    safe = pipeline.step(_snapshot())
    runtime._effort.fill_(9.)
    runtime.latest_solutions = {"left_hand": SimpleNamespace(q_ref=safe_target * 8),
                                "right_hand": SimpleNamespace(q_ref=safe_target * 9)}
    runtime.wire_diagnostics = _clamped(phase="LIFT", wire_feasible=False)
    rejected = pipeline.step(replace(_snapshot(), timestamp_ns=2))
    assert torch.equal(rejected.command.effort, safe.command.effort)
    assert torch.equal(rejected.left_hand_target, safe.left_hand_target)
    assert torch.equal(rejected.right_hand_target, safe.right_hand_target)
    assert rejected.command.timestamp_ns == 2
    assert not rejected.diagnostics.command_accepted


def test_native_runtime_safe_transition_does_not_hide_lift_reclamp_request():
    class Runtime(_Runtime):
        def compute(self, snapshot):
            self.phase = BimanualPhase.HOLD_SAFE
            self.mission._transition(self.phase)
            return BimanualCommand(snapshot.timestamp_ns, self._effort, False,
                                   ("wire_dynamics_infeasible",))

    snapshot = _snapshot()
    runtime = Runtime(snapshot)
    runtime.phase = BimanualPhase.LIFT
    runtime.mission = BimanualMission()
    runtime.mission.phase = runtime.phase
    runtime.latest_solutions = _native_runtime(snapshot).latest_solutions
    result = BimanualGraspLiftPipeline(runtime=runtime).step(snapshot)
    assert result.request_reclamp
    assert result.phase is BimanualPhase.PRELOAD
    assert runtime.mission.phase is BimanualPhase.PRELOAD
    assert result.fallback_reason == "wire_dynamics_infeasible"
