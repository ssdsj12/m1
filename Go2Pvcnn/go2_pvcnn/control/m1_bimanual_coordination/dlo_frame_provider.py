"""Isaac snapshot to DeformX frame conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _offset(name: str, value: np.ndarray) -> np.ndarray:
    offset = np.array(value, dtype=np.float64, copy=True)
    if offset.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(offset).all():
        raise ValueError(f"{name} must be finite")
    offset.setflags(write=False)
    return offset


@dataclass(frozen=True)
class PalmGraspAttachment:
    """Immutable palm-local offsets captured at a bilateral pinch."""

    left_offset_local: np.ndarray
    right_offset_local: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_offset_local",
            _offset("left_offset_local", self.left_offset_local),
        )
        object.__setattr__(
            self,
            "right_offset_local",
            _offset("right_offset_local", self.right_offset_local),
        )


def _palm_frame(arm: object) -> dict[str, np.ndarray]:
    pose = arm.palm_pose_b.detach().cpu().numpy()
    twist = arm.palm_twist_b.detach().cpu().numpy()
    return {
        "position": np.asarray(pose[:3], dtype=np.float64),
        "director": Rotation.from_rotvec(pose[3:6]).as_matrix(),
        "velocity": np.asarray(twist[:3], dtype=np.float64),
        "omega": np.asarray(twist[3:6], dtype=np.float64),
    }


def _capture_offset(arm: object, hand: object) -> np.ndarray:
    pinch_b = hand.fingertip_positions_b[:2].mean(dim=0).detach().cpu().numpy()
    pose = arm.palm_pose_b.detach().cpu().numpy()
    palm_rotation_b = Rotation.from_rotvec(pose[3:]).as_matrix()
    return palm_rotation_b.T @ (pinch_b - pose[:3])


def capture_palm_grasp_attachment(snapshot: object) -> PalmGraspAttachment:
    """Capture thumb/index pinch centers as palm-local rigid offsets."""

    return PalmGraspAttachment(
        left_offset_local=_capture_offset(snapshot.left_arm, snapshot.left_hand),
        right_offset_local=_capture_offset(snapshot.right_arm, snapshot.right_hand),
    )


def _attached_palm_frame(
    arm: object,
    offset_local: np.ndarray,
) -> dict[str, np.ndarray]:
    pose = arm.palm_pose_b.detach().cpu().numpy()
    twist = arm.palm_twist_b.detach().cpu().numpy()
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    offset_b = rotation @ offset_local
    return {
        "position": np.asarray(pose[:3] + offset_b, dtype=np.float64),
        "director": np.asarray(rotation, dtype=np.float64),
        "velocity": np.asarray(
            twist[:3] + np.cross(twist[3:], offset_b),
            dtype=np.float64,
        ),
        "omega": np.array(twist[3:], dtype=np.float64, copy=True),
    }


def right_palm_frame(snapshot: object) -> dict[str, np.ndarray]:
    """Use the right O6 palm as a single DeformX rod frame."""

    return _palm_frame(snapshot.right_arm)


def dual_palm_frames(snapshot: object) -> dict[str, dict[str, np.ndarray]]:
    """Bind the two O6 palms to the two endpoints of one Cosserat rod."""

    return {
        "left_frame": _palm_frame(snapshot.left_arm),
        "right_frame": _palm_frame(snapshot.right_arm),
    }


def dual_palm_attached_grasp_frames(
    snapshot: object,
    attachment: PalmGraspAttachment,
) -> dict[str, dict[str, np.ndarray]]:
    """Reconstruct two rigid pinch frames from current palm kinematics."""

    if not isinstance(attachment, PalmGraspAttachment):
        raise TypeError("attachment must be PalmGraspAttachment")
    return {
        "left_frame": _attached_palm_frame(
            snapshot.left_arm,
            attachment.left_offset_local,
        ),
        "right_frame": _attached_palm_frame(
            snapshot.right_arm,
            attachment.right_offset_local,
        ),
    }


__all__ = [
    "PalmGraspAttachment",
    "capture_palm_grasp_attachment",
    "dual_palm_attached_grasp_frames",
    "dual_palm_frames",
    "right_palm_frame",
]
