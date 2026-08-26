"""Stateful eight-channel residual commands for the M1 + Panda reference WBC."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


RESIDUAL_DIM = 8
WRENCH_DIM = 6
RESIDUAL_NAMES = (
    "Fx",
    "Fy",
    "Fz",
    "Mx",
    "My",
    "Mz",
    "delta_height",
    "delta_stance",
)


def _finite_positive_tuple(name: str, values: tuple[float, ...], size: int) -> None:
    if not isinstance(values, tuple) or len(values) != size:
        raise ValueError(f"{name} must be a {size}-tuple")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise ValueError(f"{name} must contain finite positive values")


def _finite_tuple(name: str, values: tuple[float, ...], size: int) -> None:
    if not isinstance(values, tuple) or len(values) != size:
        raise ValueError(f"{name} must be a {size}-tuple")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{name} must contain finite real values")


def _validate_count(num_envs: int) -> None:
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")


def _validate_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")


def _normalize_env_ids(
    env_ids: torch.Tensor | Sequence[int] | None,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    if isinstance(env_ids, torch.Tensor):
        if env_ids.dtype != torch.long:
            raise TypeError("env_ids tensor must have dtype torch.int64")
        values = env_ids.detach().to(device=device).reshape(-1)
    elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
        if any(not isinstance(value, int) or isinstance(value, bool) for value in env_ids):
            raise TypeError("env_ids must contain integers")
        values = torch.tensor(list(env_ids), dtype=torch.long, device=device)
    else:
        raise TypeError("env_ids must be an int64 tensor, integer sequence, or None")
    if values.numel() and ((values < 0) | (values >= num_envs)).any().item():
        raise IndexError(f"env_ids must be in [0,{num_envs})")
    if torch.unique(values).numel() != values.numel():
        raise ValueError("env_ids must not contain duplicates")
    return values


@dataclass(frozen=True)
class WholeBodyResidualCfg:
    """Physical authority and per-control-step rate limit for the 8D contract."""

    physical_limits: tuple[float, ...] = (
        30.0,
        30.0,
        50.0,
        15.0,
        15.0,
        8.0,
        0.04,
        0.08,
    )
    slew_fraction_per_step: float = 0.05

    def __post_init__(self) -> None:
        _finite_positive_tuple("physical_limits", self.physical_limits, RESIDUAL_DIM)
        value = self.slew_fraction_per_step
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError("slew_fraction_per_step must be finite and in (0,1]")


@dataclass(frozen=True)
class WholeBodyResidualCommand:
    physical: torch.Tensor
    wrench_b: torch.Tensor
    delta_height: torch.Tensor
    delta_stance: torch.Tensor


@dataclass(frozen=True)
class WholeBodyResidualDiagnostics:
    normalized_clipped: torch.Tensor
    physical_target: torch.Tensor
    amplitude_saturated: torch.Tensor
    slew_saturated: torch.Tensor


class WholeBodyResidualComposer:
    """Convert normalized residuals into bounded stateful physical commands."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
        cfg: WholeBodyResidualCfg | None = None,
    ) -> None:
        _validate_count(num_envs)
        _validate_dtype(dtype)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = cfg or WholeBodyResidualCfg()
        self._limits = torch.tensor(
            self.cfg.physical_limits, dtype=dtype, device=self.device
        )
        self._slew = self._limits * float(self.cfg.slew_fraction_per_step)
        self._previous_physical = torch.zeros(
            num_envs, RESIDUAL_DIM, dtype=dtype, device=self.device
        )

    @property
    def previous_physical(self) -> torch.Tensor:
        return self._previous_physical.clone()

    def _validate_action(self, normalized: torch.Tensor) -> None:
        if not isinstance(normalized, torch.Tensor):
            raise TypeError("normalized residual must be a torch.Tensor")
        if tuple(normalized.shape) != (self.num_envs, RESIDUAL_DIM):
            raise ValueError(
                "normalized residual shape must be "
                f"({self.num_envs}, {RESIDUAL_DIM})"
            )
        if normalized.dtype != self.dtype:
            raise TypeError("normalized residual dtype must match composer dtype")
        if normalized.device != self.device:
            raise ValueError("normalized residual device must match composer device")
        if not torch.isfinite(normalized).all().item():
            raise ValueError("normalized residual must contain only finite values")

    def step(
        self, normalized: torch.Tensor
    ) -> tuple[WholeBodyResidualCommand, WholeBodyResidualDiagnostics]:
        self._validate_action(normalized)
        clipped = torch.clamp(normalized, -1.0, 1.0)
        target = clipped * self._limits
        delta = torch.clamp(
            target - self._previous_physical,
            min=-self._slew,
            max=self._slew,
        )
        physical = self._previous_physical + delta
        amplitude_saturated = normalized != clipped
        slew_saturated = ~torch.isclose(
            physical, target, atol=1.0e-12, rtol=0.0
        )
        self._previous_physical = physical.detach().clone()
        physical_out = physical.clone()
        command = WholeBodyResidualCommand(
            physical=physical_out,
            wrench_b=physical_out[:, :WRENCH_DIM].clone(),
            delta_height=physical_out[:, 6].clone(),
            delta_stance=physical_out[:, 7].clone(),
        )
        diagnostics = WholeBodyResidualDiagnostics(
            normalized_clipped=clipped.clone(),
            physical_target=target.clone(),
            amplitude_saturated=amplitude_saturated.clone(),
            slew_saturated=slew_saturated.clone(),
        )
        return command, diagnostics

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        ids = _normalize_env_ids(
            env_ids, num_envs=self.num_envs, device=self.device
        )
        self._previous_physical[ids] = 0.0


@dataclass(frozen=True)
class MountWrenchFeedbackCfg:
    """Filtering and feedback gains for the base-frame mount wrench."""

    force_gain: float = 0.15
    moment_gain: float = 0.10
    filter_alpha: float = 0.20
    bias_warmup_samples: int = 0
    reference_wrench: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    command_limits: tuple[float, ...] = (30.0, 30.0, 50.0, 15.0, 15.0, 8.0)

    def __post_init__(self) -> None:
        for name in ("force_gain", "moment_gain"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.filter_alpha, bool)
            or not isinstance(self.filter_alpha, (int, float))
            or not math.isfinite(float(self.filter_alpha))
            or not 0.0 < float(self.filter_alpha) <= 1.0
        ):
            raise ValueError("filter_alpha must be finite and in (0,1]")
        if (
            not isinstance(self.bias_warmup_samples, int)
            or isinstance(self.bias_warmup_samples, bool)
            or self.bias_warmup_samples < 0
        ):
            raise ValueError("bias_warmup_samples must be a non-negative integer")
        _finite_tuple("reference_wrench", self.reference_wrench, WRENCH_DIM)
        _finite_positive_tuple("command_limits", self.command_limits, WRENCH_DIM)


class MountWrenchFeedback:
    """Maintain per-environment mount-wrench filter and frozen warm-up bias."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
        cfg: MountWrenchFeedbackCfg | None = None,
    ) -> None:
        _validate_count(num_envs)
        _validate_dtype(dtype)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = cfg or MountWrenchFeedbackCfg()
        self._filtered_wrench = torch.zeros(
            num_envs, WRENCH_DIM, dtype=dtype, device=self.device
        )
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._bias_sum = torch.zeros_like(self._filtered_wrench)
        self._bias_sample_count = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._reference = torch.tensor(
            self.cfg.reference_wrench, dtype=dtype, device=self.device
        )
        self._limits = torch.tensor(
            self.cfg.command_limits, dtype=dtype, device=self.device
        )
        self._gains = torch.tensor(
            (self.cfg.force_gain,) * 3 + (self.cfg.moment_gain,) * 3,
            dtype=dtype,
            device=self.device,
        )

    @property
    def filtered_wrench(self) -> torch.Tensor:
        return self._filtered_wrench.clone()

    @property
    def initialized(self) -> torch.Tensor:
        return self._initialized.clone()

    @property
    def bias_sample_count(self) -> torch.Tensor:
        return self._bias_sample_count.clone()

    def _validate_wrench(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != (self.num_envs, WRENCH_DIM):
            raise ValueError(
                f"{name} shape must be ({self.num_envs}, {WRENCH_DIM})"
            )
        if value.dtype != self.dtype:
            raise TypeError(f"{name} dtype must match feedback dtype")
        if value.device != self.device:
            raise ValueError(f"{name} device must match feedback device")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")

    def update(
        self,
        measured_wrench_b: torch.Tensor,
        residual_wrench_b: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_wrench("measured_wrench_b", measured_wrench_b)
        self._validate_wrench("residual_wrench_b", residual_wrench_b)
        initialized = self._initialized.unsqueeze(-1)
        alpha = float(self.cfg.filter_alpha)
        filtered = torch.where(
            initialized,
            alpha * measured_wrench_b + (1.0 - alpha) * self._filtered_wrench,
            measured_wrench_b,
        )
        bias_sum = self._bias_sum.clone()
        bias_count = self._bias_sample_count.clone()
        warmup = int(self.cfg.bias_warmup_samples)
        if warmup:
            collecting = bias_count < warmup
            bias_sum[collecting] += filtered[collecting]
            bias_count[collecting] += 1
        denominator = bias_count.clamp_min(1).to(self.dtype).unsqueeze(-1)
        bias = torch.where(
            (bias_count > 0).unsqueeze(-1),
            bias_sum / denominator,
            torch.zeros_like(bias_sum),
        )
        corrected = filtered - bias
        command = residual_wrench_b + self._gains * (
            self._reference - corrected
        )
        command = torch.clamp(command, min=-self._limits, max=self._limits)
        self._filtered_wrench = filtered.detach().clone()
        self._bias_sum = bias_sum.detach().clone()
        self._bias_sample_count = bias_count.detach().clone()
        self._initialized.fill_(True)
        return command.clone()

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        ids = _normalize_env_ids(
            env_ids, num_envs=self.num_envs, device=self.device
        )
        self._filtered_wrench[ids] = 0.0
        self._initialized[ids] = False
        self._bias_sum[ids] = 0.0
        self._bias_sample_count[ids] = 0


__all__ = [
    "MountWrenchFeedback",
    "MountWrenchFeedbackCfg",
    "RESIDUAL_DIM",
    "RESIDUAL_NAMES",
    "WholeBodyResidualCfg",
    "WholeBodyResidualCommand",
    "WholeBodyResidualComposer",
    "WholeBodyResidualDiagnostics",
]
