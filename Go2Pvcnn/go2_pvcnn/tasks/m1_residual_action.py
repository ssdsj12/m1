"""Stateful bounded residual composition for M1 hybrid actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch


M1_ACTION_DIM = 16
M1_LEG_ACTION_DIM = 12


def _require_finite_bound(name: str, value: float, *, positive: bool) -> None:
    valid = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    if positive:
        valid = valid and value > 0
        condition = "> 0"
    else:
        valid = valid and value >= 0
        condition = ">= 0"
    if not valid:
        raise ValueError(f"{name} must be finite and {condition}")


@dataclass(frozen=True)
class M1ResidualActionComposerCfg:
    """Physical limits and existing M1 action scales."""

    leg_action_scale: float = 0.25
    wheel_action_scale: float = 8.0
    leg_residual_limit_rad: float = 0.05
    wheel_residual_limit_rad_s: float = 1.0
    leg_slew_limit_rad_per_step: float = 0.01
    wheel_slew_limit_rad_s_per_step: float = 0.2

    def __post_init__(self) -> None:
        _require_finite_bound(
            "leg_action_scale", self.leg_action_scale, positive=True
        )
        _require_finite_bound(
            "wheel_action_scale", self.wheel_action_scale, positive=True
        )
        _require_finite_bound(
            "leg_residual_limit_rad", self.leg_residual_limit_rad, positive=False
        )
        _require_finite_bound(
            "wheel_residual_limit_rad_s",
            self.wheel_residual_limit_rad_s,
            positive=False,
        )
        _require_finite_bound(
            "leg_slew_limit_rad_per_step",
            self.leg_slew_limit_rad_per_step,
            positive=False,
        )
        _require_finite_bound(
            "wheel_slew_limit_rad_s_per_step",
            self.wheel_slew_limit_rad_s_per_step,
            positive=False,
        )


class M1ResidualActionComposer:
    """Compose bounded physical residuals with normalized M1 base actions."""

    def __init__(
        self,
        cfg: M1ResidualActionComposerCfg,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch.dtype")

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (num_envs, M1_ACTION_DIM)
        self._physical_residual = torch.zeros(
            shape, device=self.device, dtype=self.dtype
        )
        self._amplitude_clipped = torch.zeros(
            shape, device=self.device, dtype=torch.bool
        )
        self._slew_clipped = torch.zeros(
            shape, device=self.device, dtype=torch.bool
        )
        self._physical_limit = torch.tensor(
            [cfg.leg_residual_limit_rad] * M1_LEG_ACTION_DIM
            + [cfg.wheel_residual_limit_rad_s]
            * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        self._slew_limit = torch.tensor(
            [cfg.leg_slew_limit_rad_per_step] * M1_LEG_ACTION_DIM
            + [cfg.wheel_slew_limit_rad_s_per_step]
            * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        self._action_scale = torch.tensor(
            [cfg.leg_action_scale] * M1_LEG_ACTION_DIM
            + [cfg.wheel_action_scale] * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)

    def _validate_action_tensor(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_shape = (self.num_envs, M1_ACTION_DIM)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if value.device != self.device:
            raise ValueError(
                f"{name} must be on device {self.device}, got {value.device}"
            )
        if value.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    def compose(
        self, base_action: torch.Tensor, normalized_residual: torch.Tensor
    ) -> torch.Tensor:
        """Return the base action plus an amplitude- and slew-limited residual."""
        self._validate_action_tensor("base_action", base_action)
        self._validate_action_tensor("normalized_residual", normalized_residual)

        amplitude_clipped = normalized_residual.abs() > 1.0
        clipped_normalized = normalized_residual.clamp(-1.0, 1.0)
        target_physical = clipped_normalized * self._physical_limit
        requested_delta = target_physical - self._physical_residual
        limited_delta = torch.maximum(
            torch.minimum(requested_delta, self._slew_limit), -self._slew_limit
        )
        physical_residual = self._physical_residual + limited_delta
        slew_clipped = requested_delta.abs() > self._slew_limit
        combined_action = base_action + physical_residual / self._action_scale

        self._physical_residual = physical_residual.detach().clone()
        self._amplitude_clipped = amplitude_clipped.detach().clone()
        self._slew_clipped = slew_clipped.detach().clone()
        return combined_action

    def _normalize_env_ids(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            integer_dtypes = {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            if env_ids.dtype not in integer_dtypes:
                raise TypeError("env_ids must contain integers")
            values = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            values = list(env_ids)
        else:
            raise TypeError(
                "env_ids must be a one-dimensional integer tensor or integer sequence"
            )

        if any(
            not isinstance(index, int) or isinstance(index, bool) for index in values
        ):
            raise TypeError("env_ids must contain integers")
        for index in values:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"env_ids contains out-of-range index {index}")
        return torch.tensor(values, device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear all state, or only state belonging to selected environments."""
        if env_ids is None:
            self._physical_residual.zero_()
            self._amplitude_clipped.zero_()
            self._slew_clipped.zero_()
            return

        normalized_ids = self._normalize_env_ids(env_ids)
        self._physical_residual[normalized_ids] = 0
        self._amplitude_clipped[normalized_ids] = False
        self._slew_clipped[normalized_ids] = False

    @property
    def physical_residual(self) -> torch.Tensor:
        return self._physical_residual.clone()

    @property
    def amplitude_clipped(self) -> torch.Tensor:
        return self._amplitude_clipped.clone()

    @property
    def slew_clipped(self) -> torch.Tensor:
        return self._slew_clipped.clone()
