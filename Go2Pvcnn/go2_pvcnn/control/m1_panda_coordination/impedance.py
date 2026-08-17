"""Feedforward plus impedance effort composition for 23 WBC joints."""

from __future__ import annotations

import torch

from .contracts import CONTROLLED_DOF, require_tensor


def apply_impedance(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_des: torch.Tensor,
    qd_des: torch.Tensor,
    tau_ff: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    effort_limit: torch.Tensor,
) -> torch.Tensor:
    """Compose and symmetrically clamp one finite 23-channel effort command."""

    require_tensor("q", q, trailing_shape=(CONTROLLED_DOF,))
    if not q.is_floating_point():
        raise TypeError("q must have a floating dtype")
    for name, value in (
        ("qd", qd),
        ("q_des", q_des),
        ("qd_des", qd_des),
        ("tau_ff", tau_ff),
        ("kp", kp),
        ("kd", kd),
        ("effort_limit", effort_limit),
    ):
        require_tensor(name, value, trailing_shape=(CONTROLLED_DOF,))
        if value.shape != q.shape:
            raise ValueError(f"{name} shape must match q")
        if value.dtype != q.dtype:
            raise TypeError(f"{name} dtype must match q")
        if value.device != q.device:
            raise ValueError(f"{name} device must match q")
    if torch.any(effort_limit <= 0.0).item():
        raise ValueError("effort_limit must be positive")

    effort = tau_ff + kp * (q_des - q) + kd * (qd_des - qd)
    return torch.clamp(effort, min=-effort_limit, max=effort_limit)
