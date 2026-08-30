"""Causal sensor correction for a dynamics-model wrench prior."""

from __future__ import annotations

import torch


def sensor_calibrated_wrench(
    model_wrench_b: torch.Tensor,
    measured_wrench_b: torch.Tensor,
    *,
    observation_gain: float = 0.98,
) -> torch.Tensor:
    """Fuse a model prior with the current six-axis sensor observation."""
    if model_wrench_b.shape != measured_wrench_b.shape or model_wrench_b.shape[-1] != 6:
        raise ValueError("wrenches must have matching shape (..., 6)")
    if model_wrench_b.dtype != measured_wrench_b.dtype:
        raise TypeError("wrenches must use the same dtype")
    if not torch.isfinite(model_wrench_b).all() or not torch.isfinite(measured_wrench_b).all():
        raise ValueError("wrenches must be finite")
    gain = float(observation_gain)
    if not 0.0 <= gain <= 1.0:
        raise ValueError("observation_gain must be in [0, 1]")
    return model_wrench_b + gain * (measured_wrench_b - model_wrench_b)


__all__ = ["sensor_calibrated_wrench"]
