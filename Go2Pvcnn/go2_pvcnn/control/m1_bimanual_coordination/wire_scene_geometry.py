"""Pure scene-pose geometry for the catalog wire target."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .dlo_wire_goal import DloCenterlineState
from .object_catalog import ObjectClassRecord


@dataclass(frozen=True)
class WireSceneGeometry:
    centerline: DloCenterlineState
    center_b: torch.Tensor
    tangent_b: torch.Tensor
    radius_m: float
    left_station_b: torch.Tensor
    right_station_b: torch.Tensor


def build_wire_scene_geometry(
    pose_b: torch.Tensor,
    record: ObjectClassRecord,
    *,
    station_fraction: float = 0.30,
    sample_count: int = 17,
) -> WireSceneGeometry:
    """Derive the held wire span and grasp stations from its live scene pose."""

    if not isinstance(pose_b, torch.Tensor):
        raise TypeError("pose_b must be a torch.Tensor")
    if (
        pose_b.dtype != torch.float64
        or pose_b.device.type != "cpu"
        or pose_b.shape != (6,)
    ):
        raise ValueError("pose_b must be CPU float64 with shape [6]")
    if not bool(torch.isfinite(pose_b).all()):
        raise ValueError("pose_b must be finite")
    if record.dimensions is None or record.longitudinal_axis is None:
        raise ValueError("wire dimensions and longitudinal_axis are required")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 3
        or sample_count % 2 == 0
    ):
        raise ValueError("sample_count must be an odd integer >= 3")
    if not math.isfinite(station_fraction) or not 0.0 < station_fraction < 0.5:
        raise ValueError("station_fraction must be finite and in (0, 0.5)")

    center = pose_b[:3].clone()
    rotation = Rotation.from_rotvec(pose_b[3:].numpy())
    tangent = torch.from_numpy(
        rotation.apply(np.asarray(record.longitudinal_axis))
    ).to(torch.float64)
    tangent = tangent / torch.linalg.vector_norm(tangent)
    length = float(record.dimensions[0])
    radius = 0.5 * max(float(record.dimensions[1]), float(record.dimensions[2]))
    left = center - station_fraction * length * tangent
    right = center + station_fraction * length * tangent
    alpha = torch.linspace(0.0, 1.0, sample_count, dtype=torch.float64)
    positions = left[None] + alpha[:, None] * (right - left)[None]
    return WireSceneGeometry(
        centerline=DloCenterlineState(positions.numpy(), radius_m=radius),
        center_b=center,
        tangent_b=tangent.clone(),
        radius_m=radius,
        left_station_b=left.clone(),
        right_station_b=right.clone(),
    )


__all__ = ["WireSceneGeometry", "build_wire_scene_geometry"]
