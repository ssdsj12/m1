from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from go2_pvcnn.control.m1_bimanual_coordination.dlo_frame_provider import (
    capture_palm_grasp_attachment,
    dual_palm_attached_grasp_frames,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _pose(position, rotvec) -> torch.Tensor:
    return torch.tensor((*position, *rotvec), dtype=torch.float64)


def _snapshot_with_pinches():
    snapshot = _snapshot()
    left_positions = snapshot.left_hand.fingertip_positions_b.clone()
    right_positions = snapshot.right_hand.fingertip_positions_b.clone()
    left_positions[:2] = torch.tensor(
        [[0.99, 3.0, 3.0], [1.01, 3.0, 3.0]], dtype=torch.float64
    )
    right_positions[:2] = torch.tensor(
        [[-1.0, -1.98, 0.5], [-1.0, -2.02, 0.5]], dtype=torch.float64
    )
    return replace(
        snapshot,
        left_arm=replace(
            snapshot.left_arm,
            palm_pose_b=_pose((1.0, 2.0, 3.0), (0.0, 0.0, np.pi / 2.0)),
        ),
        right_arm=replace(
            snapshot.right_arm,
            palm_pose_b=_pose((-2.0, -2.0, 0.5), (0.0, np.pi / 2.0, 0.0)),
        ),
        left_hand=replace(snapshot.left_hand, fingertip_positions_b=left_positions),
        right_hand=replace(snapshot.right_hand, fingertip_positions_b=right_positions),
    )


def test_captured_offsets_reconstruct_bind_time_pinch_centers():
    snapshot = _snapshot_with_pinches()

    attachment = capture_palm_grasp_attachment(snapshot)
    frames = dual_palm_attached_grasp_frames(snapshot, attachment)

    np.testing.assert_allclose(
        frames["left_frame"]["position"],
        snapshot.left_hand.fingertip_positions_b[:2].mean(dim=0).numpy(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        frames["right_frame"]["position"],
        snapshot.right_hand.fingertip_positions_b[:2].mean(dim=0).numpy(),
        atol=1.0e-12,
    )
    assert frames["left_frame"]["director"].shape == (3, 3)


def test_attached_frames_follow_new_palm_pose_and_include_rigid_point_velocity():
    snapshot = _snapshot_with_pinches()
    attachment = capture_palm_grasp_attachment(snapshot)
    left_pose = _pose((2.0, 0.0, 1.0), (0.0, 0.0, np.pi))
    right_pose = _pose((0.0, 1.0, 2.0), (0.0, 0.0, -np.pi / 2.0))
    left_twist = torch.tensor(
        [0.5, 0.25, -0.1, 0.0, 0.0, 2.0], dtype=torch.float64
    )
    right_twist = torch.tensor(
        [-0.2, 0.1, 0.4, 1.5, 0.0, 0.0], dtype=torch.float64
    )
    moved = replace(
        snapshot,
        left_arm=replace(
            snapshot.left_arm,
            palm_pose_b=left_pose,
            palm_twist_b=left_twist,
        ),
        right_arm=replace(
            snapshot.right_arm,
            palm_pose_b=right_pose,
            palm_twist_b=right_twist,
        ),
    )

    frames = dual_palm_attached_grasp_frames(moved, attachment)

    for side, pose, twist, offset_local in (
        ("left", left_pose, left_twist, attachment.left_offset_local),
        ("right", right_pose, right_twist, attachment.right_offset_local),
    ):
        rotation = Rotation.from_rotvec(pose[3:].numpy()).as_matrix()
        offset_b = rotation @ offset_local
        expected_position = pose[:3].numpy() + offset_b
        expected_velocity = twist[:3].numpy() + np.cross(
            twist[3:].numpy(), offset_b
        )
        frame = frames[f"{side}_frame"]
        np.testing.assert_allclose(frame["position"], expected_position, atol=1.0e-12)
        np.testing.assert_allclose(frame["director"], rotation, atol=1.0e-12)
        np.testing.assert_allclose(frame["velocity"], expected_velocity, atol=1.0e-12)
        np.testing.assert_allclose(frame["omega"], twist[3:].numpy(), atol=1.0e-12)
        assert all(value.dtype == np.float64 for value in frame.values())
        assert all(np.isfinite(value).all() for value in frame.values())


def test_attachment_capture_has_no_hidden_state_across_episode_reset():
    first = _snapshot_with_pinches()
    first_attachment = capture_palm_grasp_attachment(first)
    reset = _snapshot()
    second_attachment = capture_palm_grasp_attachment(reset)

    assert not np.array_equal(
        first_attachment.left_offset_local,
        second_attachment.left_offset_local,
    )
    np.testing.assert_array_equal(
        second_attachment.left_offset_local,
        np.array([0.0, -0.12, 0.0]),
    )
    np.testing.assert_array_equal(
        second_attachment.right_offset_local,
        np.array([0.0, 0.12, 0.0]),
    )
