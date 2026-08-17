"""Kinematics for planar M1 base and seven-joint Panda coordination."""

from __future__ import annotations

import math

import torch

from .contracts import require_tensor


def _require_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    return require_tensor(
        name,
        value,
        trailing_shape=tuple(value.shape[-2:]),
    )


def planar_base_spatial_jacobian(ee_position_base: torch.Tensor) -> torch.Tensor:
    """Return the spatial Jacobian for base x, y, and yaw coordinates.

    Spatial rows use ``[linear xyz, angular xyz]`` ordering.
    """

    require_tensor(
        "ee_position_base",
        ee_position_base,
        trailing_shape=(3,),
    )
    if not ee_position_base.is_floating_point():
        raise TypeError("ee_position_base must have a floating dtype")

    jacobian = ee_position_base.new_zeros(ee_position_base.shape[:-1] + (6, 3))
    jacobian[..., 0, 0] = 1.0
    jacobian[..., 1, 1] = 1.0
    jacobian[..., 0, 2] = -ee_position_base[..., 1]
    jacobian[..., 1, 2] = ee_position_base[..., 0]
    jacobian[..., 5, 2] = 1.0
    return jacobian


def coordinated_jacobian(
    ee_position_base: torch.Tensor,
    panda_spatial_jacobian: torch.Tensor,
) -> torch.Tensor:
    """Concatenate planar-base and Panda columns into a 6 x 10 Jacobian."""

    require_tensor(
        "panda_spatial_jacobian",
        panda_spatial_jacobian,
        trailing_shape=(6, 7),
    )
    base_jacobian = planar_base_spatial_jacobian(ee_position_base)
    if base_jacobian.shape[:-2] != panda_spatial_jacobian.shape[:-2]:
        raise ValueError(
            "ee_position_base and panda_spatial_jacobian batch dimensions must match"
        )
    if base_jacobian.dtype != panda_spatial_jacobian.dtype:
        raise TypeError("ee_position_base and panda_spatial_jacobian dtype must match")
    if base_jacobian.device != panda_spatial_jacobian.device:
        raise ValueError("ee_position_base and panda_spatial_jacobian device must match")
    return torch.cat((base_jacobian, panda_spatial_jacobian), dim=-1)


def damped_pseudoinverse(
    jacobian: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Compute a batched SVD damped least-squares pseudoinverse."""

    _require_matrix("jacobian", jacobian)
    if isinstance(damping, bool) or not isinstance(damping, (int, float)):
        raise TypeError("damping must be a real number")
    damping = float(damping)
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")

    left, singular_values, right_transpose = torch.linalg.svd(
        jacobian, full_matrices=False
    )
    if damping == 0.0:
        tolerance = (
            torch.finfo(singular_values.dtype).eps
            * max(jacobian.shape[-2:])
            * singular_values[..., :1]
        )
        factors = torch.where(
            singular_values > tolerance,
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )
    else:
        factors = singular_values / (singular_values.square() + damping * damping)
    return (
        right_transpose.transpose(-2, -1)
        @ torch.diag_embed(factors)
        @ left.transpose(-2, -1)
    )


def singularity_metrics(
    panda_spatial_jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Panda minimum singular value and Yoshikawa manipulability."""

    require_tensor(
        "panda_spatial_jacobian",
        panda_spatial_jacobian,
        trailing_shape=(6, 7),
    )
    if not panda_spatial_jacobian.is_floating_point():
        raise TypeError("panda_spatial_jacobian must have a floating dtype")

    singular_values = torch.linalg.svdvals(panda_spatial_jacobian)
    sigma_min = singular_values[..., -1]
    manipulability = singular_values.prod(dim=-1).clamp_min(0.0)
    return sigma_min, manipulability
