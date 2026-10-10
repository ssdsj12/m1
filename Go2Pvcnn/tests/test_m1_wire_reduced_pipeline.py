from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

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
