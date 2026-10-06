"""Isaac snapshot to DeformX frame conversion."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def right_palm_frame(snapshot: object) -> dict[str, np.ndarray]:
    """Use the right O6 palm as the provisional DeformX rod frame."""

    pose = snapshot.right_arm.palm_pose_b.detach().cpu().numpy()
    twist = snapshot.right_arm.palm_twist_b.detach().cpu().numpy()
    return {
        "position": np.asarray(pose[:3], dtype=np.float64),
        "director": Rotation.from_rotvec(pose[3:6]).as_matrix(),
        "velocity": np.asarray(twist[:3], dtype=np.float64),
        "omega": np.asarray(twist[3:6], dtype=np.float64),
    }


__all__ = ["right_palm_frame"]
