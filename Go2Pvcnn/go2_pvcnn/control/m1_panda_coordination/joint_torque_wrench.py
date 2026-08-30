"""Jacobian-transpose diagnostic estimate of a Panda mount wrench."""

from __future__ import annotations

import torch


def wrench_from_joint_torque(
    jacobian_b: torch.Tensor,
    joint_torque: torch.Tensor,
    *,
    damping: float = 1.0e-4,
) -> torch.Tensor:
    """Solve ``J.T @ wrench ~= tau`` in the least-squares sense.

    This is a diagnostic estimate only: unmodelled gravity, friction and base
    constraint torques are intentionally not hidden and are reported by the
    residual.
    """
    if jacobian_b.shape != (6, 7) or joint_torque.shape != (7,):
        raise ValueError("jacobian_b must be (6,7) and joint_torque must be (7,)")
    if jacobian_b.dtype != torch.float64 or joint_torque.dtype != torch.float64:
        raise TypeError("jacobian_b and joint_torque must be float64")
    if not torch.isfinite(jacobian_b).all() or not torch.isfinite(joint_torque).all():
        raise ValueError("inputs must be finite")
    if not isinstance(damping, (float, int)) or damping < 0 or not torch.isfinite(torch.tensor(float(damping))):
        raise ValueError("damping must be finite and non-negative")
    gram = jacobian_b @ jacobian_b.transpose(0, 1)
    if damping:
        gram = gram + float(damping) ** 2 * torch.eye(6, dtype=torch.float64)
    wrench = torch.linalg.solve(gram, jacobian_b @ joint_torque)
    return wrench


__all__ = ["wrench_from_joint_torque"]
