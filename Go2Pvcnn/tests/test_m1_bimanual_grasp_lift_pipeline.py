from __future__ import annotations

from dataclasses import replace

import pytest

from go2_pvcnn.control.m1_bimanual_coordination import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.grasp_goal import (
    generate_bimanual_grasp_goal,
)
from go2_pvcnn.control.m1_bimanual_coordination.grasp_lift_pipeline import (
    BimanualGraspLiftPipeline,
)
from go2_pvcnn.control.m1_bimanual_coordination.state_machine import (
    BimanualMission,
    BimanualMissionCfg,
    BimanualMissionDiagnostics,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _goal():
    return generate_bimanual_grasp_goal(
        (0.0, 0.0, 0.5),
        dimensions=(0.30, 0.21, 0.035),
        grasp_profile="thin_two_hand",
        object_class="book",
    )


def _diagnostics(**overrides):
    values = dict(
        command_accepted=True,
        palms_reached=False,
        left_palm_reached=False,
        right_palm_reached=False,
        left_contact=False,
        right_contact=False,
        contact_consistent=True,
        bilateral_contact=False,
        force_closure_margin=0.0,
        relative_palm_slip_m=0.0,
        box_twist_norm=0.0,
        box_supported=False,
        hands_open=False,
        collision_margin_m=0.1,
        support_margin_m=0.1,
        subsystem_failure=None,
        left_contact_count=0,
        right_contact_count=0,
        left_normal_force_n=0.0,
        right_normal_force_n=0.0,
        vertical_force_n=0.0,
        object_tilt_rad=0.0,
    )
    if overrides.get("palms_reached"):
        values["left_palm_reached"] = True
        values["right_palm_reached"] = True
    values.update(overrides)
    return BimanualMissionDiagnostics(**values)


def _contact(**overrides):
    values = dict(
        left_contact=True,
        right_contact=True,
        bilateral_contact=True,
        force_closure_margin=1.0,
        left_contact_count=2,
        right_contact_count=2,
        left_normal_force_n=5.0,
        right_normal_force_n=5.0,
        vertical_force_n=10.0,
    )
    values.update(overrides)
    return _diagnostics(**values)


def test_catalog_phase_aliases_describe_close_contact_and_clamp_hold():
    assert BimanualPhase.CLOSE_CONTACT is BimanualPhase.PRELOAD
    assert BimanualPhase.CLAMP_HOLD is BimanualPhase.GRASP


def test_pipeline_exposes_o6_hand_targets_and_closed_loop_status():
    class Runtime:
        phase = BimanualPhase.APPROACH
        latest_grasp_goal = _goal()
        latest_solutions = {"left_hand": None, "right_hand": None}

        def compute(self, snapshot):
            self.phase = BimanualPhase.CLOSE_CONTACT
            return type("Command", (), {"feasible": True, "fallback_reasons": (), "effort": snapshot.base_state.new_zeros(43)})()

        def reset(self):
            self.phase = BimanualPhase.APPROACH

    pipeline = BimanualGraspLiftPipeline(_goal(), runtime=Runtime())
    result = pipeline.step(_snapshot())
    assert result.phase is BimanualPhase.PRELOAD
    assert not result.success
    assert not result.fallback


def test_catalog_goal_requires_close_contact_before_lift():
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1, grasp_dwell_steps=1),
        grasp_goal=_goal(),
    )
    snapshot = _snapshot()
    assert mission.update(snapshot, _diagnostics(palms_reached=True)).phase is BimanualPhase.PRELOAD
    # Contact is present, but below the catalog clamp force criterion.
    assert mission.update(
        replace(snapshot, timestamp_ns=2),
        _contact(left_normal_force_n=1.0, right_normal_force_n=1.0),
    ).phase is BimanualPhase.PRELOAD
    assert mission.update(replace(snapshot, timestamp_ns=3), _contact()).phase is BimanualPhase.GRASP
    assert mission.update(replace(snapshot, timestamp_ns=4), _contact()).phase is BimanualPhase.LIFT


def test_catalog_lift_requires_height_force_and_tilt_criteria():
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1, grasp_dwell_steps=1),
        grasp_goal=_goal(),
    )
    snapshot = _snapshot()
    mission.update(snapshot, _diagnostics(palms_reached=True))
    mission.update(replace(snapshot, timestamp_ns=2), _contact())
    mission.update(replace(snapshot, timestamp_ns=3), _contact())
    assert mission.phase is BimanualPhase.LIFT
    low = mission.update(replace(snapshot, timestamp_ns=4), _contact(vertical_force_n=1.0))
    assert low.phase is BimanualPhase.LIFT
    pose = snapshot.box.pose_b.clone()
    pose[2] += 0.10
    lifted = replace(snapshot, timestamp_ns=5, box=replace(snapshot.box, pose_b=pose))
    tilted = mission.update(lifted, _contact(object_tilt_rad=0.30))
    assert tilted.phase is BimanualPhase.LIFT
    assert mission.update(replace(lifted, timestamp_ns=6), _contact()).phase is BimanualPhase.HOLD


def test_catalog_lift_contact_loss_enters_deterministic_safe_fallback():
    mission = BimanualMission(
        BimanualMissionCfg(
            approach_dwell_steps=1,
            preload_dwell_steps=1,
            grasp_dwell_steps=1,
            safe_hold_steps=1,
        ),
        grasp_goal=_goal(),
    )
    snapshot = _snapshot()
    mission.update(snapshot, _diagnostics(palms_reached=True))
    mission.update(replace(snapshot, timestamp_ns=2), _contact())
    mission.update(replace(snapshot, timestamp_ns=3), _contact())
    pose = snapshot.box.pose_b.clone()
    pose[2] += 0.05
    lifted = replace(snapshot, timestamp_ns=4, box=replace(snapshot.box, pose_b=pose))
    assert mission.update(lifted, _contact(left_contact=False, right_contact=True, bilateral_contact=False)).phase is BimanualPhase.HOLD_SAFE
    assert mission.update(replace(lifted, timestamp_ns=5), _diagnostics()).phase is BimanualPhase.LOWER_SAFE


@pytest.mark.parametrize("bad", [float("nan"), -1.0])
def test_catalog_diagnostics_reject_invalid_measured_force(bad):
    with pytest.raises((TypeError, ValueError)):
        _diagnostics(left_normal_force_n=bad)
