from __future__ import annotations

from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualObjectMpc,
    BimanualRuntime,
    build_catalog_grasp_goal,
    build_catalog_grasp_goal_provider,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectClassRecord,
    ObjectInstance,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _catalog() -> ObjectCatalog:
    return ObjectCatalog(
        {
            "book": ObjectClassRecord(
                "book", Path("book.usd"), "1" * 64, "2" * 64, 0.3, 1.0, "box", "thin_two_hand"
            ),
            "bottle": ObjectClassRecord(
                "bottle", Path("bottle.usd"), "3" * 64, "4" * 64, 0.5, 1.0, "convex", "cylindrical"
            ),
        },
        {},
    )


@pytest.mark.parametrize(
    ("object_class", "dimensions", "profile"),
    (
        ("book", (0.30, 0.21, 0.035), "thin_two_hand"),
        ("bottle", (0.075, 0.075, 0.24), "cylindrical"),
    ),
)
def test_catalog_goal_reaches_object_mpc_boundary(
    object_class: str, dimensions: tuple[float, float, float], profile: str
) -> None:
    snapshot = _snapshot()
    provider = build_catalog_grasp_goal_provider(
        _catalog(),
        (ObjectInstance("target", object_class, (0.0, 0.0, 0.0)),),
        "target",
        geometry_provider=lambda _snapshot, _instance: dimensions,
    )
    runtime = BimanualRuntime(
        grasp_goal_provider=provider
    )

    sample = runtime._object_input(snapshot)
    assert sample.grasp_goal is not None
    assert sample.grasp_goal.object_class == object_class
    assert sample.grasp_goal.grasp_profile == profile
    assert sample.grasp_goal.obb_dimensions == pytest.approx(dimensions)
    assert sample.grasp_goal.left_contact_targets
    assert sample.grasp_goal.right_contact_targets
    assert sample.grasp_goal.clamp_criteria.min_contact_count_per_hand == 2
    assert sample.grasp_goal.lift_criteria.height_m == pytest.approx(0.10)
    assert runtime.latest_motion_target is not None
    assert runtime.latest_motion_target.left_fingertip_targets == sample.grasp_goal.left_fingertip_targets
    assert runtime.latest_motion_target.right_contact_targets == sample.grasp_goal.right_contact_targets
    assert runtime.latest_motion_target.clamp_criteria == sample.grasp_goal.clamp_criteria
    assert runtime.latest_motion_target.lift_criteria == sample.grasp_goal.lift_criteria

    solution = BimanualObjectMpc().plan(sample)
    goal = sample.grasp_goal
    assert torch.allclose(
        solution.left_palm_pose[-1, :3], torch.tensor(goal.left_palm_target, dtype=torch.float64)
    )
    assert torch.allclose(
        solution.right_palm_pose[-1, :3], torch.tensor(goal.right_palm_target, dtype=torch.float64)
    )


def test_custom_dimensions_are_forwarded_without_catalog_geometry() -> None:
    snapshot = _snapshot()
    goal = build_catalog_grasp_goal(
        snapshot.box.pose_b.tolist(),
        object_class="book",
        catalog=_catalog(),
        dimensions=(0.50, 0.12, 0.08),
    )
    runtime = BimanualRuntime(grasp_goal=goal)

    sample = runtime._object_input(snapshot)

    assert sample.grasp_goal is goal
    assert sample.grasp_goal.obb_dimensions == pytest.approx((0.50, 0.12, 0.08))
    result = BimanualObjectMpc().plan(sample)
    assert result.left_palm_pose[-1, 1].item() == pytest.approx(
        goal.left_palm_target[1]
    )


def test_catalog_default_dimensions_require_explicit_opt_in() -> None:
    snapshot = _snapshot()
    provider = build_catalog_grasp_goal_provider(
        _catalog(),
        (ObjectInstance("target", "book", (0.0, 0.0, 0.0)),),
        "target",
    )
    with pytest.raises(ValueError, match="explicitly enable default dimensions"):
        provider(snapshot)

    fallback_provider = build_catalog_grasp_goal_provider(
        _catalog(),
        (ObjectInstance("target", "book", (0.0, 0.0, 0.0)),),
        "target",
        allow_default_dimensions=True,
    )
    assert fallback_provider(snapshot).obb_dimensions == pytest.approx(
        (0.30, 0.21, 0.035)
    )


def test_legacy_box_runtime_has_no_grasp_goal_and_keeps_existing_palm_target() -> None:
    snapshot = _snapshot()
    runtime = BimanualRuntime()

    sample = runtime._object_input(snapshot)
    result = BimanualObjectMpc().plan(sample)

    assert sample.grasp_goal is None
    assert result.left_palm_pose[-1, 1].item() == pytest.approx(0.20)
    assert result.right_palm_pose[-1, 1].item() == pytest.approx(-0.20)
