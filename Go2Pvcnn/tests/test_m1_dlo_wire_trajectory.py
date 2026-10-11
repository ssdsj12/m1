from dataclasses import replace

import numpy as np
import torch

from go2_pvcnn.control.m1_bimanual_coordination.deformx_adapter import (
    DualDloStepResult,
)
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
    DloCenterlineState,
)
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_trajectory import (
    DloWireTrajectoryPlanner,
)
from tests.test_m1_bimanual_object_mpc import _snapshot
from tests.test_m1_bimanual_dual_arm_mpc import _object_solution


def _step() -> DualDloStepResult:
    return DualDloStepResult(
        state=DloCenterlineState(
            np.array(
                [
                    [0.40, 0.00, 1.20],
                    [0.50, 0.00, 1.20],
                    [0.60, 0.00, 1.20],
                ],
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


def test_wire_trajectory_places_palms_outside_opposed_line_contacts():
    snapshot = _snapshot()
    snapshot = replace(
        snapshot,
        left_arm=replace(
            snapshot.left_arm,
            palm_pose_b=torch.tensor([0.50, 0.30, 1.20, 0.1, 0.2, 0.3], dtype=torch.float64),
        ),
        right_arm=replace(
            snapshot.right_arm,
            palm_pose_b=torch.tensor([0.50, -0.30, 1.20, -0.1, -0.2, -0.3], dtype=torch.float64),
        ),
    )
    planner = DloWireTrajectoryPlanner(palm_standoff_m=0.08, palm_speed_m_s=1.0)

    planned = planner.plan(snapshot, _object_solution(), _step())

    left_contact_y = np.mean(
        [target.position[1] for target in planned.contact_goal.left_contact_targets]
    )
    right_contact_y = np.mean(
        [target.position[1] for target in planned.contact_goal.right_contact_targets]
    )
    assert planned.object_solution.left_palm_pose[-1, 1] > left_contact_y
    assert planned.object_solution.right_palm_pose[-1, 1] < right_contact_y
    torch.testing.assert_close(
        planned.object_solution.left_palm_pose[:, 3:],
        snapshot.left_arm.palm_pose_b[3:].repeat(25, 1),
    )
    torch.testing.assert_close(
        planned.object_solution.right_palm_pose[:, 3:],
        snapshot.right_arm.palm_pose_b[3:].repeat(25, 1),
    )


def test_wire_trajectory_respects_per_node_palm_speed_limit():
    snapshot = _snapshot()
    speed = 0.04
    dt = 0.04
    planner = DloWireTrajectoryPlanner(
        dt=dt,
        palm_speed_m_s=speed,
        palm_standoff_m=0.08,
    )

    planned = planner.plan(snapshot, _object_solution(), _step()).object_solution

    for current, path in (
        (snapshot.left_arm.palm_pose_b[:3], planned.left_palm_pose[:, :3]),
        (snapshot.right_arm.palm_pose_b[:3], planned.right_palm_pose[:, :3]),
    ):
        points = torch.cat((current[None], path), dim=0)
        increments = torch.linalg.vector_norm(torch.diff(points, dim=0), dim=1)
        assert bool((increments <= speed * dt + 1.0e-12).all())


def test_wire_trajectory_preserves_object_mpc_solution_and_diagnostics():
    snapshot = _snapshot()
    original = _object_solution()

    planned = DloWireTrajectoryPlanner().plan(snapshot, original, _step()).object_solution

    torch.testing.assert_close(planned.box_pose, original.box_pose)
    torch.testing.assert_close(planned.box_twist, original.box_twist)
    torch.testing.assert_close(planned.left_wrench, original.left_wrench)
    torch.testing.assert_close(planned.right_wrench, original.right_wrench)
    assert planned.diagnostics is original.diagnostics
    assert not torch.equal(planned.left_palm_pose, original.left_palm_pose)


def test_plan_state_uses_distinct_stations_and_explicit_palm_standoff():
    planner = DloWireTrajectoryPlanner(palm_speed_m_s=10.)
    state = _step().state
    plan = planner.plan_state(_snapshot(), _object_solution(), state)
    compatible = planner.plan(_snapshot(), _object_solution(), _step())
    for side, station in (("left", state.positions[0]), ("right", state.positions[-1])):
        targets = getattr(plan.contact_goal, f"{side}_contact_targets")
        normal = np.array(getattr(plan.contact_goal, f"{side}_palm_approach_normal"))
        np.testing.assert_allclose(np.mean([t.position for t in targets], axis=0), station)
        pose = getattr(plan.object_solution, f"{side}_palm_pose")
        np.testing.assert_allclose(pose[-1, :3], station - 0.08 * normal)
        torch.testing.assert_close(pose, getattr(compatible.object_solution, f"{side}_palm_pose"))


def test_staged_wire_palm_translation_aligns_measured_pinch_centers():
    snapshot = _snapshot()
    # Physical O6 palms are held at their measured orientations: their pinch
    # centers point mostly forward, not along the wire's +/-Y approach normal.
    for side, offset in (("left", (.090, .014, -.010)), ("right", (.070, .045, -.040))):
        palm = getattr(snapshot, f"{side}_arm").palm_pose_b[:3]
        tips = (palm + torch.tensor(offset, dtype=torch.float64)).repeat(5, 1)
        tips[0, 2] += .03
        tips[1, 2] -= .03
        snapshot = replace(snapshot, **{f"{side}_hand": replace(
            getattr(snapshot, f"{side}_hand"), fingertip_positions_b=tips,
        )})
    planner = DloWireTrajectoryPlanner(palm_speed_m_s=10., track_pinch_center=True)
    plan = planner.plan_state(snapshot, _object_solution(), _step().state)
    for side, station in (("left", _step().state.positions[0]), ("right", _step().state.positions[-1])):
        arm = getattr(snapshot, f"{side}_arm")
        pinch = getattr(snapshot, f"{side}_hand").fingertip_positions_b[:2].mean(0)
        pose = getattr(plan.object_solution, f"{side}_palm_pose")
        predicted_pinch = pinch + pose[-1, :3] - arm.palm_pose_b[:3]
        np.testing.assert_allclose(predicted_pinch, station, atol=1e-12)
        torch.testing.assert_close(pose[:, 3:], arm.palm_pose_b[3:].repeat(25, 1))


def test_staged_wire_palm_target_uses_actual_o6_thumb_index_offsets():
    """The palm reference is derived from the measured O6 fingertip layout."""
    snapshot = _snapshot()
    # Measured initial O6 palm-local offsets from the identity-geometry
    # artifact (thumb, index, then the remaining fingertips).
    offsets = {
        "left": ((0.088278055, 0.011744365, 0.022792578),
                  (0.091163635, 0.016176015, -0.042143941)),
        "right": ((0.044979000, -0.070304650, -0.038530200),
                   (0.072326870, -0.019518450, -0.068821910)),
    }
    for side, (thumb_offset, index_offset) in offsets.items():
        palm = getattr(snapshot, f"{side}_arm").palm_pose_b[:3]
        tips = palm + torch.tensor(
            [thumb_offset, index_offset, thumb_offset, index_offset, thumb_offset],
            dtype=torch.float64,
        )
        snapshot = replace(snapshot, **{f"{side}_hand": replace(
            getattr(snapshot, f"{side}_hand"), fingertip_positions_b=tips,
        )})

    planner = DloWireTrajectoryPlanner(
        palm_speed_m_s=10.0, track_pinch_center=True
    )
    state = _step().state
    plan = planner.plan_state(snapshot, _object_solution(), state)
    for side, station in (("left", state.positions[0]), ("right", state.positions[-1])):
        arm = getattr(snapshot, f"{side}_arm")
        pinch = getattr(snapshot, f"{side}_hand").fingertip_positions_b[:2].mean(0)
        final_palm = getattr(plan.object_solution, f"{side}_palm_pose")[-1, :3]
        expected = arm.palm_pose_b[:3] + torch.tensor(station) - pinch
        torch.testing.assert_close(final_palm, expected, atol=1e-12, rtol=0.0)
