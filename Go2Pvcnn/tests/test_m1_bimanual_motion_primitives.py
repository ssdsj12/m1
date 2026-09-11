from __future__ import annotations

from dataclasses import replace

import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.motion_primitives import (
    BimanualMotionPrimitive,
    MotionPrimitiveCfg,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _contact_snapshot(left: bool = True, right: bool = True):
    snapshot = _snapshot()
    left_mask = torch.tensor([left, False, False, False, False])
    right_mask = torch.tensor([right, False, False, False, False])
    return replace(
        snapshot,
        left_hand=replace(snapshot.left_hand, contact_mask=left_mask),
        right_hand=replace(snapshot.right_hand, contact_mask=right_mask),
    )


def test_lift_target_climbs_without_overshooting_ten_centimetres() -> None:
    primitive = BimanualMotionPrimitive(
        MotionPrimitiveCfg(lift_height_m=0.10, lift_speed_m_s=0.025)
    )
    snapshot = _contact_snapshot()
    primitive.target(BimanualPhase.APPROACH, snapshot)

    target = primitive.target(BimanualPhase.LIFT, snapshot)

    assert target.box_pose_b.shape == (25, 6)
    assert target.box_pose_b[0, 2] > snapshot.box.pose_b[2]
    assert target.box_pose_b[-1, 2] <= snapshot.box.pose_b[2] + 0.10
    assert target.recovery_side is None


def test_lost_contact_pauses_lift_and_requests_only_missing_side() -> None:
    primitive = BimanualMotionPrimitive()
    snapshot = _contact_snapshot(left=True, right=False)
    primitive.target(BimanualPhase.APPROACH, snapshot)

    target = primitive.target(BimanualPhase.LIFT, snapshot)

    assert target.recovery_side == "right"
    assert torch.allclose(
        target.box_pose_b,
        snapshot.box.pose_b.repeat(target.box_pose_b.shape[0], 1),
    )


def test_stable_grasp_captures_both_palm_offsets_in_box() -> None:
    primitive = BimanualMotionPrimitive()
    snapshot = _contact_snapshot()

    target = primitive.target(BimanualPhase.GRASP, snapshot)

    assert torch.allclose(
        target.left_palm_in_box,
        snapshot.left_arm.palm_pose_b - snapshot.box.pose_b,
    )
    assert torch.allclose(
        target.right_palm_in_box,
        snapshot.right_arm.palm_pose_b - snapshot.box.pose_b,
    )
