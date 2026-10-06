"""Isaac snapshot to DeformX frame conversion."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _palm_frame(arm: object) -> dict[str, np.ndarray]:
    pose = arm.palm_pose_b.detach().cpu().numpy()
    twist = arm.palm_twist_b.detach().cpu().numpy()
    return {
        "position": np.asarray(pose[:3], dtype=np.float64),
        "director": Rotation.from_rotvec(pose[3:6]).as_matrix(),
        "velocity": np.asarray(twist[:3], dtype=np.float64),
        "omega": np.asarray(twist[3:6], dtype=np.float64),
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


__all__ = ["dual_palm_frames", "right_palm_frame"]
