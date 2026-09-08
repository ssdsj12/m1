"""Pure rigid-frame transforms for bimanual snapshots.

Quaternions use Isaac's ``wxyz`` ordering.  Functions preserve the input
tensor's device and floating dtype and never read simulator state.
"""

from __future__ import annotations

import torch


FLOATING_BASE_DOF = 6
ARTICULATION_JOINT_DOF = 53
GENERALIZED_DOF = FLOATING_BASE_DOF + ARTICULATION_JOINT_DOF


def damped_cartesian_joint_delta(
    spatial_jacobian: torch.Tensor,
    cartesian_error: torch.Tensor,
    *,
    damping: float = 0.05,
    max_abs_joint_delta: float = 0.15,
) -> torch.Tensor:
    """Map a small 3D translation or 6D pose error to a bounded joint step."""

    if spatial_jacobian.shape != (6, 7):
        raise ValueError("spatial_jacobian must have shape (6, 7)")
    if not isinstance(cartesian_error, torch.Tensor) or cartesian_error.shape not in {
        (3,),
        (6,),
    }:
        raise ValueError("cartesian_error must have shape (3,) or (6,)")
    if not torch.isfinite(cartesian_error).all().item():
        raise ValueError("cartesian_error must contain only finite values")
    if damping <= 0.0 or max_abs_joint_delta <= 0.0:
        raise ValueError("damping and max_abs_joint_delta must be positive")
    rows = cartesian_error.shape[0]
    task_jacobian = spatial_jacobian[:rows]
    regularizer = torch.eye(
        rows, dtype=task_jacobian.dtype, device=task_jacobian.device
    ) * float(damping) ** 2
    delta = task_jacobian.T @ torch.linalg.solve(
        task_jacobian @ task_jacobian.T + regularizer,
        cartesian_error.to(dtype=task_jacobian.dtype, device=task_jacobian.device),
    )
    return torch.clamp(
        delta,
        min=-float(max_abs_joint_delta),
        max=float(max_abs_joint_delta),
    )


def embed_fixed_base_vector(value: torch.Tensor) -> torch.Tensor:
    """Embed a 53-DoF fixed-root vector in the frozen 59-DoF contract."""

    if value.shape[-1] == GENERALIZED_DOF:
        return value.clone()
    if value.shape[-1] != ARTICULATION_JOINT_DOF:
        raise ValueError("generalized vector must end in 53 or 59 columns")
    result = value.new_zeros(value.shape[:-1] + (GENERALIZED_DOF,))
    result[..., FLOATING_BASE_DOF:] = value
    return result


def embed_fixed_base_jacobian(value: torch.Tensor) -> torch.Tensor:
    """Prepend six locked-base columns to a fixed-root PhysX Jacobian."""

    if value.ndim < 2 or value.shape[-2] != 6:
        raise ValueError("spatial Jacobian must have six rows")
    if value.shape[-1] == GENERALIZED_DOF:
        return value.clone()
    if value.shape[-1] != ARTICULATION_JOINT_DOF:
        raise ValueError("spatial Jacobian must end in 53 or 59 columns")
    result = value.new_zeros(value.shape[:-1] + (GENERALIZED_DOF,))
    result[..., FLOATING_BASE_DOF:] = value
    return result


def embed_fixed_base_mass_matrix(value: torch.Tensor) -> torch.Tensor:
    """Embed fixed-root joint inertia with an identity locked-base block."""

    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("mass matrix must be square")
    if value.shape[-1] == GENERALIZED_DOF:
        return value.clone()
    if value.shape[-1] != ARTICULATION_JOINT_DOF:
        raise ValueError("mass matrix must be 53x53 or 59x59")
    result = value.new_zeros(value.shape[:-2] + (GENERALIZED_DOF, GENERALIZED_DOF))
    result[..., :FLOATING_BASE_DOF, :FLOATING_BASE_DOF] = torch.eye(
        FLOATING_BASE_DOF, dtype=value.dtype, device=value.device
    )
    result[..., FLOATING_BASE_DOF:, FLOATING_BASE_DOF:] = value
    return result


def _require_last_dim(name: str, value: torch.Tensor, size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape[-1:] != (size,):
        raise ValueError(f"{name} must end in dimension {size}; got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _normalize_quat(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    _require_last_dim("quaternion_wxyz", quaternion_wxyz, 4)
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion_wxyz.dtype).eps).item():
        raise ValueError("quaternion_wxyz must have non-zero norm")
    return quaternion_wxyz / norm


def _quat_conjugate(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    result = quaternion_wxyz.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _quat_multiply(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    left_wxyz, right_wxyz = torch.broadcast_tensors(left_wxyz, right_wxyz)
    lw, lx, ly, lz = left_wxyz.unbind(dim=-1)
    rw, rx, ry, rz = right_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate(quaternion_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    _require_last_dim("vectors", vectors, 3)
    quaternion_wxyz = _normalize_quat(quaternion_wxyz)
    quaternion_wxyz, vectors = torch.broadcast_tensors(
        quaternion_wxyz, torch.cat((vectors, vectors[..., :1]), dim=-1)
    )
    vectors = vectors[..., :3]
    vector_part = quaternion_wxyz[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vectors, dim=-1)
    return vectors + quaternion_wxyz[..., :1] * twice_cross + torch.linalg.cross(
        vector_part, twice_cross, dim=-1
    )


def _quat_to_rotvec(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    quaternion_wxyz = _normalize_quat(quaternion_wxyz)
    quaternion_wxyz = torch.where(
        quaternion_wxyz[..., :1] < 0.0,
        -quaternion_wxyz,
        quaternion_wxyz,
    )
    vector = quaternion_wxyz[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion_wxyz[..., :1].clamp_min(1.0e-12))
    scale = torch.where(vector_norm > 1.0e-10, angle / vector_norm, 2.0 * torch.ones_like(vector_norm))
    return scale * vector


def _quat_to_matrix(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    quaternion_wxyz = _normalize_quat(quaternion_wxyz)
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    two = quaternion_wxyz.new_tensor(2.0)
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion_wxyz.shape[:-1] + (3, 3))


def pose_in_base(
    base_position_w: torch.Tensor,
    base_quaternion_wxyz: torch.Tensor,
    target_position_w: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return target position and rotation vector relative to the moving base."""

    _require_last_dim("base_position_w", base_position_w, 3)
    _require_last_dim("target_position_w", target_position_w, 3)
    base_quaternion_wxyz = _normalize_quat(base_quaternion_wxyz)
    target_quaternion_wxyz = _normalize_quat(target_quaternion_wxyz)
    base_inverse = _quat_conjugate(base_quaternion_wxyz)
    position_b = _quat_rotate(base_inverse, target_position_w - base_position_w)
    orientation_b = _quat_to_rotvec(
        _quat_multiply(base_inverse, target_quaternion_wxyz)
    )
    return torch.cat((position_b, orientation_b), dim=-1)


def twist_in_base(
    base_position_w: torch.Tensor,
    base_quaternion_wxyz: torch.Tensor,
    base_linear_velocity_w: torch.Tensor,
    base_angular_velocity_w: torch.Tensor,
    target_position_w: torch.Tensor,
    target_linear_velocity_w: torch.Tensor,
    target_angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Return target twist relative to a translating and rotating base frame."""

    for name, value in (
        ("base_position_w", base_position_w),
        ("base_linear_velocity_w", base_linear_velocity_w),
        ("base_angular_velocity_w", base_angular_velocity_w),
        ("target_position_w", target_position_w),
        ("target_linear_velocity_w", target_linear_velocity_w),
        ("target_angular_velocity_w", target_angular_velocity_w),
    ):
        _require_last_dim(name, value, 3)
    base_inverse = _quat_conjugate(_normalize_quat(base_quaternion_wxyz))
    transported_linear_w = (
        target_linear_velocity_w
        - base_linear_velocity_w
        - torch.linalg.cross(
            base_angular_velocity_w,
            target_position_w - base_position_w,
            dim=-1,
        )
    )
    angular_w = target_angular_velocity_w - base_angular_velocity_w
    return torch.cat(
        (
            _quat_rotate(base_inverse, transported_linear_w),
            _quat_rotate(base_inverse, angular_w),
        ),
        dim=-1,
    )


def vectors_in_base(
    base_quaternion_wxyz: torch.Tensor, vectors_w: torch.Tensor
) -> torch.Tensor:
    """Rotate one vector or a batch of vectors from world into base axes."""

    _require_last_dim("vectors_w", vectors_w, 3)
    base_inverse = _quat_conjugate(_normalize_quat(base_quaternion_wxyz))
    if vectors_w.ndim > base_inverse.ndim:
        for _ in range(vectors_w.ndim - base_inverse.ndim):
            base_inverse = base_inverse.unsqueeze(-2)
    return _quat_rotate(base_inverse, vectors_w)


def spatial_jacobian_in_base(
    base_quaternion_wxyz: torch.Tensor, jacobian_w: torch.Tensor
) -> torch.Tensor:
    """Rotate both linear and angular rows of a spatial Jacobian into base."""

    if not isinstance(jacobian_w, torch.Tensor) or jacobian_w.ndim != 2:
        raise ValueError("jacobian_w must be a rank-2 torch.Tensor")
    if jacobian_w.shape[0] != 6:
        raise ValueError(f"jacobian_w must have six rows; got {tuple(jacobian_w.shape)}")
    if not torch.isfinite(jacobian_w).all().item():
        raise ValueError("jacobian_w must contain only finite values")
    rotation_base_world = _quat_to_matrix(
        _quat_conjugate(_normalize_quat(base_quaternion_wxyz))
    )
    result = jacobian_w.clone()
    result[:3] = rotation_base_world @ jacobian_w[:3]
    result[3:] = rotation_base_world @ jacobian_w[3:]
    return result


def physx_jacobian_body_row(
    body_id: int, body_count: int, jacobian_body_count: int
) -> int:
    """Resolve current and legacy PhysX Jacobian body-row layouts."""

    if jacobian_body_count == body_count:
        return body_id
    if jacobian_body_count == body_count - 1 and body_id > 0:
        return body_id - 1
    raise ValueError("PhysX Jacobian layout does not match articulation bodies")


__all__ = [
    "embed_fixed_base_jacobian",
    "embed_fixed_base_mass_matrix",
    "embed_fixed_base_vector",
    "physx_jacobian_body_row",
    "pose_in_base",
    "spatial_jacobian_in_base",
    "twist_in_base",
    "vectors_in_base",
]
