from __future__ import annotations

import math

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.grasp_goal import (
    BimanualGraspGoal,
    generate_bimanual_grasp_goal,
)


PROFILES = (
    ("book", "thin_two_hand", (0.30, 0.21, 0.035)),
    ("bottle", "cylindrical", (0.075, 0.075, 0.24)),
    ("cup", "rim_or_body", (0.10, 0.10, 0.11)),
    ("cube", "symmetric_two_hand", (0.12, 0.12, 0.12)),
    ("cylinder", "polygonal_two_hand", (0.10, 0.10, 0.20)),
)


def _assert_mirror_symmetric(goal: BimanualGraspGoal) -> None:
    left = np.asarray(goal.left.palm_position)
    right = np.asarray(goal.right.palm_position)
    center = np.asarray(goal.object_center)
    assert np.allclose((left + right) / 2.0, center)
    assert np.allclose(left + right, 2.0 * center)
    assert np.allclose(
        np.asarray(goal.left.contact.position) + np.asarray(goal.right.contact.position),
        2.0 * center,
    )
    assert np.allclose(
        np.asarray(goal.left.contact.normal), -np.asarray(goal.right.contact.normal)
    )
    assert len(goal.left.fingertip_positions) == len(goal.right.fingertip_positions)
    for left_tip, right_tip in zip(
        goal.left.fingertip_positions, goal.right.fingertip_positions
    ):
        assert np.allclose(np.asarray(left_tip) + np.asarray(right_tip), 2.0 * center)


@pytest.mark.parametrize("name,profile,dimensions", PROFILES)
def test_catalog_profiles_produce_complete_symmetric_goal(
    name: str, profile: str, dimensions: tuple[float, float, float]
) -> None:
    goal = generate_bimanual_grasp_goal(
        object_pose=(0.4, -0.2, 0.7, 1.0, 0.0, 0.0, 0.0),
        dimensions=dimensions,
        grasp_profile=profile,
    )

    assert goal.object_class == name
    assert goal.grasp_profile == profile
    assert goal.left.palm_position != goal.right.palm_position
    assert len(goal.left.fingertip_positions) == 5
    assert len(goal.left.contact.targets) == 5
    assert goal.clamp.min_contact_count_per_hand == 2
    assert goal.lift.height_m > 0.0
    assert goal.lift.hold_time_s > 0.0
    _assert_mirror_symmetric(goal)


def test_object_pose_and_obb_axes_are_respected() -> None:
    # The first OBB axis is rotated into world Y.  Contacts must therefore be
    # mirrored about world Y, while the goal remains centered at the pose.
    goal = generate_bimanual_grasp_goal(
        object_pose=(1.0, 2.0, 3.0),
        dimensions=(0.4, 0.2, 0.1),
        obb_axes=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        grasp_profile="symmetric_two_hand",
    )
    assert np.allclose(goal.object_center, (1.0, 2.0, 3.0))
    assert goal.left.palm_position[0] == pytest.approx(goal.right.palm_position[0])
    assert abs(goal.left.palm_position[1] - 2.0) == pytest.approx(
        abs(goal.right.palm_position[1] - 2.0)
    )
    assert np.allclose(goal.left.contact.normal, -np.asarray(goal.right.contact.normal))
    assert np.allclose(np.abs(goal.left.contact.normal), (0.0, 1.0, 0.0))


def test_point_cloud_derives_deterministic_obb_and_unseen_dimensions_scale_targets() -> None:
    points = np.asarray(
        [
            (-0.3, -0.1, -0.05),
            (-0.3, -0.1, 0.05),
            (-0.3, 0.1, -0.05),
            (-0.3, 0.1, 0.05),
            (0.3, -0.1, -0.05),
            (0.3, -0.1, 0.05),
            (0.3, 0.1, -0.05),
            (0.3, 0.1, 0.05),
        ],
        dtype=float,
    )
    first = generate_bimanual_grasp_goal(
        object_pose=(0.0, 0.0, 0.0),
        point_cloud=points,
        grasp_profile="thin_two_hand",
    )
    second = generate_bimanual_grasp_goal(
        object_pose=(0.0, 0.0, 0.0),
        point_cloud=points[::-1],
        grasp_profile="thin_two_hand",
    )
    assert first.obb_dimensions == pytest.approx((0.6, 0.2, 0.1), abs=1.0e-8)
    assert first == second

    small = generate_bimanual_grasp_goal(
        object_pose=(0.0, 0.0, 0.0),
        dimensions=(0.04, 0.02, 0.01),
        grasp_profile="symmetric_two_hand",
    )
    large = generate_bimanual_grasp_goal(
        object_pose=(0.0, 0.0, 0.0),
        dimensions=(0.40, 0.20, 0.10),
        grasp_profile="symmetric_two_hand",
    )
    assert large.clamp.closure_width_m > small.clamp.closure_width_m
    assert np.linalg.norm(np.asarray(large.left.contact.position)) > np.linalg.norm(
        np.asarray(small.left.contact.position)
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dimensions": (0.0, 0.1, 0.1)},
        {"dimensions": (0.1, math.inf, 0.1)},
        {"dimensions": (0.1, 0.1)},
        {"dimensions": (0.1, 0.1, 0.1), "grasp_profile": "unknown"},
        {"dimensions": (0.1, 0.1, 0.1), "obb_axes": ((1.0, 0.0, 0.0),) * 3},
        {
            "point_cloud": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        },
    ),
)
def test_invalid_geometry_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="geometry|dimension|profile|point|orthonormal"
    ):
        generate_bimanual_grasp_goal(
            object_pose=(0.0, 0.0, 0.0),
            grasp_profile=kwargs.pop("grasp_profile", "symmetric_two_hand"),
            **kwargs,
        )


def test_pose_quaternion_is_normalized_and_profile_defaults_are_explicit() -> None:
    goal = generate_bimanual_grasp_goal(
        object_pose=(0.0, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0),
        dimensions=(0.2, 0.1, 0.1),
        grasp_profile="generic",
    )
    assert goal.object_pose[3:] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert goal.clamp.max_slip_speed_m_s > 0.0
    assert goal.lift.max_tilt_rad < math.pi / 2.0
