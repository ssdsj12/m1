"""Kinematic bounds for M1 + Panda coordination."""

from __future__ import annotations

import math

import torch

from .contracts import require_tensor


def compute_velocity_bounds(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    v_max: torch.Tensor,
    a_max: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect position, velocity, and one-step acceleration bounds."""

    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError("dt must be a real number")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if not q.is_floating_point():
        raise TypeError("q must have a floating dtype")

    require_tensor("q", q, trailing_shape=tuple(q.shape[-1:]))
    for name, value in (
        ("qd", qd),
        ("q_min", q_min),
        ("q_max", q_max),
        ("v_max", v_max),
        ("a_max", a_max),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        require_tensor(name, value, trailing_shape=tuple(value.shape[-1:]))
        if value.shape != q.shape:
            raise ValueError(f"q and {name} shape must match")
        if value.dtype != q.dtype:
            raise TypeError(f"q and {name} dtype must match")
        if value.device != q.device:
            raise ValueError(f"q and {name} device must match")

    if torch.any(q_min > q_max).item():
        raise ValueError("q_min must not exceed q_max")
    if torch.any(v_max < 0.0).item():
        raise ValueError("v_max must be non-negative")
    if torch.any(a_max < 0.0).item():
        raise ValueError("a_max must be non-negative")

    lower = torch.maximum(
        torch.maximum((q_min - q) / dt, -v_max),
        qd - a_max * dt,
    )
    upper = torch.minimum(
        torch.minimum((q_max - q) / dt, v_max),
        qd + a_max * dt,
    )
    infeasible = lower > upper
    if infeasible.any().item():
        indices = torch.nonzero(infeasible.reshape(-1), as_tuple=False).flatten()
        rendered = ", ".join(str(index) for index in indices.tolist())
        raise ValueError(f"velocity bounds are infeasible at indices: {rendered}")
    return lower, upper
