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
