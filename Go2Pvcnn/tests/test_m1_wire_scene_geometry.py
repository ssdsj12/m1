from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectClassRecord,
)
from go2_pvcnn.control.m1_bimanual_coordination.wire_scene_geometry import (
    build_wire_scene_geometry,
)


def _wire_record(
    *,
    dimensions: tuple[float, float, float] | None = (0.08, 0.01, 0.01),
    axis: tuple[float, float, float] | None = (1.0, 0.0, 0.0),
) -> ObjectClassRecord:
    return ObjectClassRecord(
        object_class="wire",
        usd_path=Path("wire.usda"),
        source_sha256="0" * 64,
        resolved_sha256="1" * 64,
        mass_kg=0.05,
        scale=1.0,
        collision_profile="convex_decomposition",
        grasp_profile="cylindrical",
        dimensions=dimensions,
        longitudinal_axis=axis,
    )


def test_scene_wire_geometry_rotates_axis_and_places_distinct_stations() -> None:
    record = _wire_record(
        dimensions=(0.08, 0.01, 0.01), axis=(1.0, 0.0, 0.0)
    )
    pose_b = torch.tensor(
        [0.65, 0.0, 1.20, 0.0, 0.0, math.pi / 2], dtype=torch.float64
    )

    geometry = build_wire_scene_geometry(pose_b, record, sample_count=5)

    torch.testing.assert_close(geometry.center_b, pose_b[:3])
    torch.testing.assert_close(
        geometry.tangent_b,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        geometry.left_station_b,
        torch.tensor([0.65, -0.024, 1.20], dtype=torch.float64),
    )
    torch.testing.assert_close(
        geometry.right_station_b,
        torch.tensor([0.65, 0.024, 1.20], dtype=torch.float64),
    )
    assert geometry.radius_m == pytest.approx(0.005)
    assert geometry.centerline.positions.shape == (5, 3)
    assert geometry.centerline.positions.dtype == np.float64


@pytest.mark.parametrize(
    ("dimensions", "axis"),
    [
        (None, (1.0, 0.0, 0.0)),
        ((0.08, 0.01, 0.01), None),
    ],
)
def test_scene_wire_geometry_requires_dimensions_and_axis(
    dimensions: tuple[float, float, float] | None,
    axis: tuple[float, float, float] | None,
) -> None:
    pose_b = torch.zeros(6, dtype=torch.float64)

    with pytest.raises(ValueError, match="dimensions and longitudinal_axis"):
        build_wire_scene_geometry(
            pose_b, _wire_record(dimensions=dimensions, axis=axis)
        )


def test_scene_wire_geometry_rejects_nonfinite_pose() -> None:
    pose_b = torch.zeros(6, dtype=torch.float64)
    pose_b[0] = torch.nan

    with pytest.raises(ValueError, match="pose_b must be finite"):
        build_wire_scene_geometry(pose_b, _wire_record())


@pytest.mark.parametrize("sample_count", [1, 2, 4, True])
def test_scene_wire_geometry_requires_odd_sample_count_at_least_three(
    sample_count: int,
) -> None:
    with pytest.raises(ValueError, match="odd integer >= 3"):
        build_wire_scene_geometry(
            torch.zeros(6, dtype=torch.float64),
            _wire_record(),
            sample_count=sample_count,
        )
