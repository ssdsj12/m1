from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.frame_kinematics import (
    damped_cartesian_joint_delta,
    embed_fixed_base_jacobian,
    embed_fixed_base_mass_matrix,
    embed_fixed_base_vector,
    physx_jacobian_body_row,
    pose_in_base,
    spatial_jacobian_in_base,
    twist_in_base,
    vectors_in_base,
)


def test_damped_cartesian_joint_delta_tracks_translation_and_limits_step():
    jacobian = torch.zeros((6, 7), dtype=torch.float64)
    jacobian[:3, :3] = torch.eye(3, dtype=torch.float64)
    delta = damped_cartesian_joint_delta(
        jacobian,
        torch.tensor([0.0, -0.04, 0.0], dtype=torch.float64),
        damping=1.0e-3,
        max_abs_joint_delta=0.02,
    )
    assert delta.shape == (7,)
    assert delta[1].item() == pytest.approx(-0.02)
    assert torch.count_nonzero(delta).item() == 1


def test_damped_cartesian_joint_delta_can_lock_spatial_orientation():
    jacobian = torch.zeros((6, 7), dtype=torch.float64)
    jacobian[:, :6] = torch.eye(6, dtype=torch.float64)
    delta = damped_cartesian_joint_delta(
        jacobian,
        torch.tensor([0.0, 0.01, 0.0, 0.0, 0.0, -0.02], dtype=torch.float64),
        damping=1.0e-3,
        max_abs_joint_delta=0.05,
    )
    assert torch.allclose(
        delta[:6],
        torch.tensor([0.0, 0.01, 0.0, 0.0, 0.0, -0.02], dtype=torch.float64),
        atol=1.0e-5,
    )


DTYPE = torch.float64


def _identity_quat() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)


def _z_quat(angle: float) -> torch.Tensor:
    return torch.tensor(
        [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)],
        dtype=DTYPE,
    )


def test_pose_uses_base_translation_and_rotation() -> None:
    base_position = torch.tensor([1.0, 2.0, 0.5], dtype=DTYPE)
    base_quaternion = _z_quat(math.pi / 2.0)
    target_position = torch.tensor([1.0, 3.0, 0.5], dtype=DTYPE)

    pose = pose_in_base(
        base_position,
        base_quaternion,
        target_position,
        base_quaternion,
    )

    assert torch.allclose(
        pose,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE),
        atol=1.0e-12,
    )


def test_pose_is_invariant_to_common_world_translation() -> None:
    offset = torch.tensor([4.0, -7.0, 2.0], dtype=DTYPE)
    base_position = torch.tensor([0.2, -0.4, 0.6], dtype=DTYPE)
    target_position = torch.tensor([0.8, 0.1, 0.9], dtype=DTYPE)
    base_quaternion = _z_quat(0.3)
    target_quaternion = _z_quat(-0.2)

    original = pose_in_base(
        base_position,
        base_quaternion,
        target_position,
        target_quaternion,
    )
    translated = pose_in_base(
        base_position + offset,
        base_quaternion,
        target_position + offset,
        target_quaternion,
    )

    assert torch.allclose(original, translated, atol=1.0e-12)


def test_twist_removes_base_transport_velocity() -> None:
    base_position = torch.tensor([0.5, 0.0, 0.0], dtype=DTYPE)
    target_position = torch.tensor([1.5, 0.0, 0.0], dtype=DTYPE)
    base_angular_velocity = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE)
    base_linear_velocity = torch.tensor([0.1, 0.2, 0.0], dtype=DTYPE)
    target_linear_velocity = base_linear_velocity + torch.linalg.cross(
        base_angular_velocity, target_position - base_position
    )

    result = twist_in_base(
        base_position,
        _identity_quat(),
        base_linear_velocity,
        base_angular_velocity,
        target_position,
        target_linear_velocity,
        base_angular_velocity,
    )

    assert torch.allclose(result, torch.zeros(6, dtype=DTYPE), atol=1.0e-12)


def test_vectors_and_both_spatial_jacobian_blocks_rotate_into_base() -> None:
    base_quaternion = _z_quat(math.pi / 2.0)
    world_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    expected = torch.tensor([[0.0, -1.0, 0.0]], dtype=DTYPE)
    jacobian_world = torch.zeros((6, 2), dtype=DTYPE)
    jacobian_world[0, 0] = 1.0
    jacobian_world[3, 1] = 1.0

    assert torch.allclose(
        vectors_in_base(base_quaternion, world_x), expected, atol=1.0e-12
    )
    jacobian_base = spatial_jacobian_in_base(base_quaternion, jacobian_world)
    assert torch.allclose(jacobian_base[:3, 0], expected[0], atol=1.0e-12)
    assert torch.allclose(jacobian_base[3:, 1], expected[0], atol=1.0e-12)


@pytest.mark.parametrize(
    ("jacobian_body_count", "expected"),
    ((60, 12), (59, 11)),
)
def test_physx_body_row_supports_current_and_legacy_layouts(
    jacobian_body_count: int, expected: int
) -> None:
    assert physx_jacobian_body_row(12, 60, jacobian_body_count) == expected


def test_physx_body_row_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        physx_jacobian_body_row(12, 60, 58)


def test_fixed_base_physx_dynamics_are_embedded_in_floating_base_contract() -> None:
    mass = torch.diag(torch.arange(1, 54, dtype=DTYPE))
    bias = torch.arange(53, dtype=DTYPE)
    jacobian = torch.arange(6 * 53, dtype=DTYPE).reshape(6, 53)

    embedded_mass = embed_fixed_base_mass_matrix(mass)
    embedded_bias = embed_fixed_base_vector(bias)
    embedded_jacobian = embed_fixed_base_jacobian(jacobian)

    assert embedded_mass.shape == (59, 59)
    assert torch.equal(embedded_mass[:6, :6], torch.eye(6, dtype=DTYPE))
    assert torch.equal(embedded_mass[6:, 6:], mass)
    assert torch.equal(embedded_bias[:6], torch.zeros(6, dtype=DTYPE))
    assert torch.equal(embedded_bias[6:], bias)
    assert torch.equal(embedded_jacobian[:, :6], torch.zeros(6, 6, dtype=DTYPE))
    assert torch.equal(embedded_jacobian[:, 6:], jacobian)


def test_snapshot_adapter_routes_all_base_frame_quantities_through_pure_transforms() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text(encoding="utf-8")

    assert "pose_in_base(" in wrapper
    assert "twist_in_base(" in wrapper
    assert "vectors_in_base(" in wrapper
    assert "spatial_jacobian_in_base(" in wrapper
    assert "physx_jacobian_body_row(" in wrapper
    assert "palm_position - base_position" not in wrapper
    assert "box_position - data.root_pos_w" not in wrapper
