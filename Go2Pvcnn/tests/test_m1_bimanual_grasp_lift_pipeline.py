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
        left_normal_alignment=1.0,
        right_normal_alignment=1.0,
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


def test_pipeline_reports_only_done_as_success():
    class Runtime:
        latest_grasp_goal = None
        latest_solutions = {"left_hand": None, "right_hand": None}

        def __init__(self, phase):
            self.phase = phase

        def compute(self, snapshot):
            return type(
                "Command",
                (),
                {
                    "feasible": True,
                    "fallback_reasons": (),
                    "effort": snapshot.base_state.new_zeros(43),
                },
            )()

        def reset(self):
            pass

    done = BimanualGraspLiftPipeline(runtime=Runtime(BimanualPhase.DONE)).step(_snapshot())
    terminated = BimanualGraspLiftPipeline(
        runtime=Runtime(BimanualPhase.TERMINATED)
    ).step(_snapshot())
    assert done.success
    assert not done.fallback
    assert not terminated.success
    assert terminated.fallback


def test_pipeline_propagates_mission_fallback_reason():
    class Runtime:
        phase = BimanualPhase.TERMINATED
        latest_grasp_goal = None
        latest_solutions = {"left_hand": None, "right_hand": None}
        latest_mission_state = type(
            "MissionState", (), {"fallback_reason": "clamp_criteria_failed"}
        )()

        def compute(self, snapshot):
            return type(
                "Command",
                (),
                {
                    "feasible": True,
                    "fallback_reasons": (),
                    "effort": snapshot.base_state.new_zeros(43),
                },
            )()

        def reset(self):
            pass

    result = BimanualGraspLiftPipeline(runtime=Runtime()).step(_snapshot())
    assert result.fallback
    assert result.fallback_reason == "clamp_criteria_failed"


def test_pipeline_rejects_conflicting_injected_runtime_goal_without_mutating_it():
    first = _goal()
    second = generate_bimanual_grasp_goal(
        (0.1, 0.0, 0.5),
        dimensions=(0.30, 0.21, 0.035),
        grasp_profile="thin_two_hand",
        object_class="book",
    )

    class Mission:
        def __init__(self):
            self.grasp_goal = first
            self.setter_calls = 0

        def set_grasp_goal(self, goal):
            self.setter_calls += 1
            self.grasp_goal = goal

    class Runtime:
        phase = BimanualPhase.APPROACH
        latest_solutions = {"left_hand": None, "right_hand": None}

        def __init__(self):
            self.mission = Mission()
            self._grasp_goal = first

        def compute(self, snapshot):
            raise AssertionError("compute must not run")

    runtime = Runtime()
    with pytest.raises(ValueError, match="conflicting grasp_goal"):
        BimanualGraspLiftPipeline(second, runtime=runtime)
    assert runtime.mission.grasp_goal is first
    assert runtime.mission.setter_calls == 0


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


def test_catalog_clamp_enforces_goal_force_cap_and_alignment():
    base = _goal()
    goal = replace(
        base,
        clamp=replace(
            base.clamp,
            max_normal_force_n=4.0,
            min_normal_alignment=0.9,
        ),
    )
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1),
        grasp_goal=goal,
    )
    snapshot = _snapshot()
    mission.update(snapshot, _diagnostics(palms_reached=True))
    assert mission.update(
        replace(snapshot, timestamp_ns=2),
        _contact(left_normal_force_n=5.0, right_normal_force_n=5.0),
    ).phase is BimanualPhase.PRELOAD
    assert mission.update(
        replace(snapshot, timestamp_ns=3),
        _contact(left_normal_alignment=0.8, right_normal_alignment=1.0),
    ).phase is BimanualPhase.PRELOAD
    assert mission.update(
        replace(snapshot, timestamp_ns=4),
        _contact(
            left_normal_force_n=4.0,
            right_normal_force_n=4.0,
            left_normal_alignment=0.9,
            right_normal_alignment=0.9,
        ),
    ).phase is BimanualPhase.GRASP


def test_catalog_clamp_uses_goal_slip_speed_threshold():
    base = _goal()
    goal = replace(
        base,
        clamp=replace(base.clamp, max_slip_speed_m_s=0.1),
    )
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1),
        grasp_goal=goal,
    )
    snapshot = _snapshot()
    mission.update(snapshot, _diagnostics(palms_reached=True))
    assert mission.update(
        replace(snapshot, timestamp_ns=2),
        _contact(relative_palm_slip_speed_m_s=0.2),
    ).phase is BimanualPhase.HOLD_SAFE
    mission = BimanualMission(
        BimanualMissionCfg(approach_dwell_steps=1, preload_dwell_steps=1),
        grasp_goal=goal,
    )
    mission.update(snapshot, _diagnostics(palms_reached=True))
    assert mission.update(
        replace(snapshot, timestamp_ns=3),
        _contact(relative_palm_slip_speed_m_s=0.1),
    ).phase is BimanualPhase.GRASP


def test_catalog_hold_uses_goal_hold_time_not_legacy_cfg():
    base = _goal()
    goal = replace(base, lift=replace(base.lift, hold_time_s=0.01))
    mission = BimanualMission(
        BimanualMissionCfg(
            approach_dwell_steps=1,
            preload_dwell_steps=1,
            grasp_dwell_steps=1,
            hold_duration_s=3.0,
        ),
        grasp_goal=goal,
    )
    snapshot = _snapshot()
    mission.update(snapshot, _diagnostics(palms_reached=True))
    mission.update(replace(snapshot, timestamp_ns=2), _contact())
    mission.update(replace(snapshot, timestamp_ns=3), _contact())
    pose = snapshot.box.pose_b.clone()
    pose[2] += goal.lift.height_m
    lifted = replace(snapshot, box=replace(snapshot.box, pose_b=pose))
    assert mission.update(replace(lifted, timestamp_ns=4), _contact()).phase is BimanualPhase.HOLD
    assert mission.update(replace(lifted, timestamp_ns=9_000_004), _contact()).phase is BimanualPhase.HOLD
    assert mission.update(replace(lifted, timestamp_ns=19_000_004), _contact()).phase is BimanualPhase.LOWER


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
