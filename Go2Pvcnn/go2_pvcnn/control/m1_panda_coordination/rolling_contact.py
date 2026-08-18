"""Pure rolling-contact kinematics for the M1 wheel set."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import GENERALIZED_DOF, require_tensor


@dataclass(frozen=True)
class RollingContactCfg:
    """Physical wheel constants in canonical FAR/FBL/RAR/RBL order."""

    wheel_radius_m: float = 0.095
    wheel_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.wheel_radius_m) or self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be finite and positive")
        if len(self.wheel_signs) != 4 or any(
            sign not in (-1.0, 1.0) for sign in self.wheel_signs
        ):
            raise ValueError(
                "wheel_signs must contain four values in {-1.0, 1.0}"
            )


@dataclass(frozen=True)
class RollingContactMetrics:
    """Instantaneous bottom-point velocities expressed in the root heading frame."""

    contact_velocity_heading: torch.Tensor
    max_longitudinal_residual_mps: float
    max_lateral_slip_mps: float


def wheel_speed_from_base_velocity(
    vx: torch.Tensor, cfg: RollingContactCfg
) -> torch.Tensor:
    """Map one longitudinal base command to four signed wheel angular speeds."""

    require_tensor("vx", vx, trailing_shape=())
    if vx.ndim != 0:
        raise ValueError("vx must be one scalar tensor")
    signs = vx.new_tensor(cfg.wheel_signs)
    return signs * vx / cfg.wheel_radius_m


def contact_point_linear_jacobian(
    body_jacobian: torch.Tensor, point_offset_w: torch.Tensor
) -> torch.Tensor:
    """Map generalized velocity to a rigid body's offset-point linear velocity."""

    require_tensor(
        "body_jacobian",
        body_jacobian,
        trailing_shape=(6, GENERALIZED_DOF),
    )
    require_tensor("point_offset_w", point_offset_w, trailing_shape=(3,))
    if body_jacobian.ndim != 2:
        raise ValueError("body_jacobian must be one 6-by-31 matrix")
    if point_offset_w.ndim != 1:
        raise ValueError("point_offset_w must be one 3-vector")
    if point_offset_w.dtype != body_jacobian.dtype:
        raise TypeError("point_offset_w dtype must match body_jacobian")
    if point_offset_w.device != body_jacobian.device:
        raise ValueError("point_offset_w device must match body_jacobian")

    x, y, z = point_offset_w
    skew = point_offset_w.new_zeros((3, 3))
    skew[0, 1], skew[0, 2] = -z, y
    skew[1, 0], skew[1, 2] = z, -x
    skew[2, 0], skew[2, 1] = -y, x
    return body_jacobian[:3] - skew @ body_jacobian[3:]


def build_wheel_contact_jacobian(
    body_jacobians: torch.Tensor, cfg: RollingContactCfg
) -> torch.Tensor:
    """Build the four bottom-point linear Jacobians as one 12-by-31 matrix."""

    require_tensor(
        "body_jacobians",
        body_jacobians,
        trailing_shape=(4, 6, GENERALIZED_DOF),
    )
    if body_jacobians.ndim != 3:
        raise ValueError("body_jacobians must be one 4-by-6-by-31 tensor")
    offset = body_jacobians.new_tensor((0.0, 0.0, -cfg.wheel_radius_m))
    return torch.stack(
        [
            contact_point_linear_jacobian(body_jacobians[index], offset)
            for index in range(4)
        ]
    ).reshape(12, GENERALIZED_DOF)


def rolling_contact_metrics(
    contact_jacobian: torch.Tensor,
    generalized_velocity: torch.Tensor,
    yaw: float,
) -> RollingContactMetrics:
    """Measure longitudinal rolling residual and lateral slip at wheel bottoms."""

    require_tensor(
        "contact_jacobian",
        contact_jacobian,
        trailing_shape=(12, GENERALIZED_DOF),
    )
    require_tensor(
        "generalized_velocity",
        generalized_velocity,
        trailing_shape=(GENERALIZED_DOF,),
    )
    if contact_jacobian.ndim != 2:
        raise ValueError("contact_jacobian must be one 12-by-31 matrix")
    if generalized_velocity.ndim != 1:
        raise ValueError("generalized_velocity must be one 31-vector")
    if generalized_velocity.dtype != contact_jacobian.dtype:
        raise TypeError("generalized_velocity dtype must match contact_jacobian")
    if generalized_velocity.device != contact_jacobian.device:
        raise ValueError("generalized_velocity device must match contact_jacobian")
    if not math.isfinite(float(yaw)):
        raise ValueError("yaw must be finite")

    velocity_w = (contact_jacobian @ generalized_velocity).reshape(4, 3)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    world_to_heading = velocity_w.new_tensor(
        ((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    velocity_heading = velocity_w @ world_to_heading.transpose(0, 1)
    return RollingContactMetrics(
        contact_velocity_heading=velocity_heading,
        max_longitudinal_residual_mps=float(
            velocity_heading[:, 0].abs().max().item()
        ),
        max_lateral_slip_mps=float(velocity_heading[:, 1].abs().max().item()),
    )
