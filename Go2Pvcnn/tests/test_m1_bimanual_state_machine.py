from __future__ import annotations

from dataclasses import replace

from go2_pvcnn.control.m1_bimanual_coordination import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.state_machine import (
    BimanualMission,
    BimanualMissionCfg,
    BimanualMissionDiagnostics,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _diagnostics(**overrides) -> BimanualMissionDiagnostics:
    values = {
        "command_accepted": True,
        "palms_reached": False,
        "bilateral_contact": False,
        "force_closure_margin": 0.0,
        "relative_palm_slip_m": 0.0,
        "box_twist_norm": 0.0,
        "box_supported": False,
        "hands_open": False,
        "collision_margin_m": 0.1,
        "support_margin_m": 0.1,
        "subsystem_failure": None,
    }
    values.update(overrides)
    return BimanualMissionDiagnostics(**values)


def _lifted(height: float, supported: bool = False):
    snapshot = _snapshot()
    pose = snapshot.box.pose_b.clone()
    pose[2] += height
    return replace(snapshot, box=replace(snapshot.box, pose_b=pose, supported=supported))


def test_normal_phase_sequence_requires_contact_lift_hold_and_support():
    mission = BimanualMission(
        BimanualMissionCfg(
            approach_dwell_steps=1,
            preload_dwell_steps=1,
            grasp_dwell_steps=1,
            hold_steps=2,
        )
    )
    assert mission.update(_snapshot(), _diagnostics(palms_reached=True)).phase is BimanualPhase.PRELOAD
    assert mission.update(
        _snapshot(),
        _diagnostics(bilateral_contact=True, force_closure_margin=1.0),
    ).phase is BimanualPhase.GRASP
    assert mission.update(
        _snapshot(),
        _diagnostics(bilateral_contact=True, force_closure_margin=1.0),
    ).phase is BimanualPhase.LIFT
    assert mission.update(_lifted(0.10), _diagnostics()).phase is BimanualPhase.HOLD
    assert mission.update(_lifted(0.10), _diagnostics()).phase is BimanualPhase.HOLD
    assert mission.update(_lifted(0.10), _diagnostics()).phase is BimanualPhase.LOWER
    supported = _lifted(0.0, supported=True)
    assert mission.update(
        supported, _diagnostics(box_supported=True, box_twist_norm=0.001)
    ).phase is BimanualPhase.RELEASE
    assert mission.update(
        supported,
        _diagnostics(box_supported=True, box_twist_norm=0.001, hands_open=True),
    ).phase is BimanualPhase.DONE


def test_default_hold_is_exactly_three_seconds_at_200_hz():
    cfg = BimanualMissionCfg()
    assert cfg.physics_dt == 0.005
    assert cfg.hold_steps == 600
    assert cfg.lift_height_m == 0.10


def test_normal_transition_is_not_committed_when_command_is_rejected():
    mission = BimanualMission(BimanualMissionCfg(approach_dwell_steps=1))
    state = mission.update(
        _snapshot(),
        _diagnostics(command_accepted=False, palms_reached=True),
    )
    assert state.phase is BimanualPhase.APPROACH


def test_preload_requires_contact_force_closure_and_slip_margin_together():
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1)
    )
    mission.update(_snapshot(), _diagnostics(palms_reached=True))

    no_closure = mission.update(
        _snapshot(),
        _diagnostics(bilateral_contact=True, force_closure_margin=0.0),
    )
    assert no_closure.phase is BimanualPhase.PRELOAD
    excessive_slip = mission.update(
        _snapshot(),
        _diagnostics(
            bilateral_contact=True,
            force_closure_margin=1.0,
            relative_palm_slip_m=0.006,
            box_supported=True,
        ),
    )
    assert excessive_slip.phase is BimanualPhase.PRELOAD
    ready = mission.update(
        _snapshot(),
        _diagnostics(
            bilateral_contact=True,
            force_closure_margin=1.0,
            relative_palm_slip_m=0.001,
        ),
    )
    assert ready.phase is BimanualPhase.GRASP


def test_slip_while_airborne_enters_hold_lower_release_safe_path():
    mission = BimanualMission(BimanualMissionCfg(safe_hold_steps=1))
    assert mission.update(
        _lifted(0.05), _diagnostics(relative_palm_slip_m=0.01)
    ).phase is BimanualPhase.HOLD_SAFE
    assert mission.update(_lifted(0.05), _diagnostics()).phase is BimanualPhase.LOWER_SAFE
    supported = _lifted(0.0, supported=True)
    assert mission.update(
        supported, _diagnostics(box_supported=True, box_twist_norm=0.0)
    ).phase is BimanualPhase.SAFE_RELEASE
    assert mission.update(
        supported,
        _diagnostics(box_supported=True, box_twist_norm=0.0, hands_open=True),
    ).phase is BimanualPhase.TERMINATED


def test_repeated_subsystem_failure_enters_safe_path():
    mission = BimanualMission(BimanualMissionCfg(max_consecutive_failures=2))
    failed = _diagnostics(command_accepted=False, subsystem_failure="arm")
    supported = _lifted(0.0, supported=True)
    assert mission.update(supported, failed).phase is BimanualPhase.APPROACH
    assert mission.update(
        supported, replace(failed, box_supported=True)
    ).phase is BimanualPhase.SAFE_RELEASE
