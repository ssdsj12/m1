"""Link-level Newton--Euler reaction wrench utilities for the Panda subtree."""

from __future__ import annotations

import torch


DTYPE = torch.float64
RNE_FILTER_ALPHA = 0.5


def _check(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != DTYPE or value.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU float64 tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def low_pass_wrench(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Apply the fixed 200 Hz causal smoothing used by RNE diagnostics."""

    _check("previous", previous, (6,))
    _check("current", current, (6,))
    return RNE_FILTER_ALPHA * current + (1.0 - RNE_FILTER_ALPHA) * previous


def _rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)
    return rotation.transpose(0, 1) @ vector


def _rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternions to world-from-local rotations."""

    w, x, y, z = quaternions.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def recursive_newton_euler_reaction(
    masses: torch.Tensor,
    com_pos_w: torch.Tensor,
    inertia_quat_w: torch.Tensor,
    inertia_com_local: torch.Tensor,
    linear_acc_w: torch.Tensor,
    angular_vel_w: torch.Tensor,
    angular_acc_w: torch.Tensor,
    gravity_w: torch.Tensor,
    *,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return the child-on-parent reaction of a rigid-link subtree.

    Pose, velocity and acceleration inputs use world axes.
    ``inertia_com_local`` is about each COM in its PhysX principal frame and is
    rotated to world axes with ``inertia_quat_w``.  Link external contacts are
    intentionally excluded: this function models the internal Panda-to-M1
    installation reaction.
    """

    terms = recursive_newton_euler_terms(
        masses,
        com_pos_w,
        inertia_quat_w,
        inertia_com_local,
        linear_acc_w,
        angular_vel_w,
        angular_acc_w,
        gravity_w,
        base_pos_w=base_pos_w,
    )
    world_reaction = torch.cat(
        (terms["required_force_w"], -terms["required_moment_w"])
    )
    force_b = _rotate_inverse(base_quat_w, world_reaction[:3])
    moment_b = _rotate_inverse(base_quat_w, world_reaction[3:])
    return torch.cat((force_b, moment_b))


def recursive_newton_euler_terms(
    masses: torch.Tensor,
    com_pos_w: torch.Tensor,
    inertia_quat_w: torch.Tensor,
    inertia_com_local: torch.Tensor,
    linear_acc_w: torch.Tensor,
    angular_vel_w: torch.Tensor,
    angular_acc_w: torch.Tensor,
    gravity_w: torch.Tensor,
    *,
    base_pos_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return world-frame RNE force and moment components for diagnostics."""

    if masses.ndim != 1 or masses.shape[0] == 0:
        raise ValueError("masses must have shape (links,) with at least one link")
    links = masses.shape[0]
    for name, value, shape in (
        ("masses", masses, (links,)),
        ("com_pos_w", com_pos_w, (links, 3)),
        ("inertia_quat_w", inertia_quat_w, (links, 4)),
        ("inertia_com_local", inertia_com_local, (links, 3, 3)),
        ("linear_acc_w", linear_acc_w, (links, 3)),
        ("angular_vel_w", angular_vel_w, (links, 3)),
        ("angular_acc_w", angular_acc_w, (links, 3)),
        ("gravity_w", gravity_w, (3,)),
        ("base_pos_w", base_pos_w, (3,)),
    ):
        _check(name, value, shape)
    if not torch.all(masses > 0.0).item():
        raise ValueError("masses must be strictly positive")

    rotations_w_local = _rotation_matrices(inertia_quat_w)
    inertia_com_w = torch.bmm(
        torch.bmm(rotations_w_local, inertia_com_local),
        rotations_w_local.transpose(1, 2),
    )
    angular_momentum_rate = torch.bmm(
        inertia_com_w, angular_acc_w.unsqueeze(-1)
    ).squeeze(-1)
    angular_momentum_rate += torch.linalg.cross(
        angular_vel_w,
        torch.bmm(inertia_com_w, angular_vel_w.unsqueeze(-1)).squeeze(-1),
        dim=-1,
    )
    required_force = masses.unsqueeze(-1) * (linear_acc_w - gravity_w)
    required_moment = angular_momentum_rate + torch.linalg.cross(
        com_pos_w - base_pos_w, required_force, dim=-1
    )
    # PhysX link linear accelerations use the opposite translational reaction
    # convention from the incoming-joint force tensor; angular reaction keeps
    # the Newton--Euler sign after the parent/child conversion.
    return {
        "required_force_w": required_force.sum(dim=0),
        "angular_momentum_moment_w": angular_momentum_rate.sum(dim=0),
        "lever_arm_moment_w": torch.linalg.cross(
            com_pos_w - base_pos_w, required_force, dim=-1
        ).sum(dim=0),
        "required_moment_w": required_moment.sum(dim=0),
    }


__all__ = [
    "low_pass_wrench",
    "recursive_newton_euler_reaction",
    "recursive_newton_euler_terms",
]
