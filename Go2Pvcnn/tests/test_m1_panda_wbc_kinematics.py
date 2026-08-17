import math

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.kinematics import (
    coordinated_jacobian,
    damped_pseudoinverse,
    planar_base_spatial_jacobian,
    singularity_metrics,
)


def _transform_point(planar_pose: torch.Tensor, point_base: torch.Tensor) -> torch.Tensor:
    x, y, yaw = planar_pose
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, yaw.new_zeros(()))),
            torch.stack((sine, cosine, yaw.new_zeros(()))),
            torch.stack((yaw.new_zeros(()), yaw.new_zeros(()), yaw.new_ones(()))),
        )
    )
    return rotation @ point_base + torch.stack((x, y, yaw.new_zeros(())))


def test_planar_base_spatial_jacobian_has_linear_then_angular_rows():
    point = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)

    jacobian = planar_base_spatial_jacobian(point)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert jacobian.shape == (6, 3)
    assert torch.equal(jacobian, expected)


def test_planar_base_linear_columns_match_central_finite_difference():
    point = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    analytic = planar_base_spatial_jacobian(point)[:3]
    epsilon = 1.0e-6
    numerical_columns = []

    for coordinate in range(3):
        delta = torch.zeros(3, dtype=torch.float64)
        delta[coordinate] = epsilon
        positive = _transform_point(delta, point)
        negative = _transform_point(-delta, point)
        numerical_columns.append((positive - negative) / (2.0 * epsilon))

    numerical = torch.stack(numerical_columns, dim=-1)
    assert torch.allclose(analytic, numerical, atol=1.0e-10, rtol=1.0e-10)


def test_planar_base_spatial_jacobian_preserves_batch_dtype_and_device():
    points = torch.tensor(
        [[[0.1, 0.2, 0.3]], [[-0.4, 0.5, 0.6]]], dtype=torch.float32
    )

    jacobian = planar_base_spatial_jacobian(points)

    assert jacobian.shape == (2, 1, 6, 3)
    assert jacobian.dtype == points.dtype
    assert jacobian.device == points.device


def test_coordinated_jacobian_concatenates_base_then_panda_columns():
    point = torch.tensor([0.3, 0.1, 0.8], dtype=torch.float64)
    panda = torch.arange(42, dtype=torch.float64).reshape(6, 7)

    jacobian = coordinated_jacobian(point, panda)

    assert jacobian.shape == (6, 10)
    assert torch.equal(jacobian[:, :3], planar_base_spatial_jacobian(point))
    assert torch.equal(jacobian[:, 3:], panda)


def test_coordinated_jacobian_supports_batches():
    point = torch.tensor([[0.3, 0.1, 0.8], [0.2, -0.4, 0.7]])
    panda = torch.randn(2, 6, 7)

    jacobian = coordinated_jacobian(point, panda)

    assert jacobian.shape == (2, 6, 10)
    assert torch.equal(jacobian[..., 3:], panda)


def test_coordinated_jacobian_rejects_mismatched_batch_shape():
    with pytest.raises(ValueError, match="batch dimensions must match"):
        coordinated_jacobian(torch.zeros(2, 3), torch.zeros(3, 6, 7))


def test_damped_pseudoinverse_reconstructs_well_conditioned_matrix():
    jacobian = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ],
        dtype=torch.float64,
    )

    pseudoinverse = damped_pseudoinverse(jacobian, damping=1.0e-8)

    assert pseudoinverse.shape == (4, 3)
    assert torch.allclose(
        jacobian @ pseudoinverse @ jacobian,
        jacobian,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_damped_pseudoinverse_handles_batched_rank_deficiency_finitely():
    jacobian = torch.zeros(2, 6, 10, dtype=torch.float64)
    jacobian[0, :6, :6] = torch.eye(6, dtype=torch.float64)

    pseudoinverse = damped_pseudoinverse(jacobian, damping=1.0e-3)

    assert pseudoinverse.shape == (2, 10, 6)
    assert torch.isfinite(pseudoinverse).all()
    assert torch.count_nonzero(pseudoinverse[1]) == 0


def test_zero_damping_uses_finite_moore_penrose_inverse_for_zero_singular_values():
    jacobian = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))

    pseudoinverse = damped_pseudoinverse(jacobian, damping=0.0)

    assert torch.equal(
        pseudoinverse,
        torch.diag(torch.tensor([0.5, 0.0], dtype=torch.float64)),
    )
    assert torch.isfinite(pseudoinverse).all()


@pytest.mark.parametrize("damping", [-1.0, float("nan"), float("inf")])
def test_damped_pseudoinverse_rejects_invalid_damping(damping):
    with pytest.raises(ValueError, match="damping must be finite and non-negative"):
        damped_pseudoinverse(torch.eye(3), damping=damping)


def test_singularity_metrics_return_minimum_and_product_of_singular_values():
    diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    panda = torch.zeros(6, 7, dtype=torch.float64)
    panda[:, :6] = torch.diag(diagonal)

    sigma_min, manipulability = singularity_metrics(panda)

    assert sigma_min.item() == pytest.approx(1.0)
    assert manipulability.item() == pytest.approx(math.prod(diagonal.tolist()))


def test_singularity_metrics_preserve_batch_dimensions_and_rank_zero():
    panda = torch.zeros(2, 4, 6, 7, dtype=torch.float32)
    panda[0, 0, :, :6] = torch.eye(6)

    sigma_min, manipulability = singularity_metrics(panda)

    assert sigma_min.shape == (2, 4)
    assert manipulability.shape == (2, 4)
    assert sigma_min[0, 0].item() == pytest.approx(1.0)
    assert manipulability[0, 0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(sigma_min[1]) == 0
    assert torch.count_nonzero(manipulability[1]) == 0


@pytest.mark.parametrize(
    ("function_name", "arguments", "message"),
    [
        (
            "planar",
            (torch.tensor([0.0, float("nan"), 0.0]),),
            "ee_position_base must contain only finite values",
        ),
        (
            "coordinated",
            (torch.zeros(3), torch.full((6, 7), float("inf"))),
            "panda_spatial_jacobian must contain only finite values",
        ),
        (
            "pseudoinverse",
            (torch.tensor([[float("nan")]]), 0.1),
            "jacobian must contain only finite values",
        ),
        (
            "metrics",
            (torch.full((6, 7), float("nan")),),
            "panda_spatial_jacobian must contain only finite values",
        ),
    ],
)
def test_kinematics_reject_non_finite_inputs(function_name, arguments, message):
    functions = {
        "planar": planar_base_spatial_jacobian,
        "coordinated": coordinated_jacobian,
        "pseudoinverse": lambda value, damping: damped_pseudoinverse(
            value, damping=damping
        ),
        "metrics": singularity_metrics,
    }

    with pytest.raises(ValueError, match=message):
        functions[function_name](*arguments)


def test_kinematics_reject_wrong_trailing_shapes():
    with pytest.raises(ValueError, match=r"ee_position_base must end with shape \(3,\)"):
        planar_base_spatial_jacobian(torch.zeros(4))
    with pytest.raises(
        ValueError, match=r"panda_spatial_jacobian must end with shape \(6, 7\)"
    ):
        singularity_metrics(torch.zeros(7, 6))
