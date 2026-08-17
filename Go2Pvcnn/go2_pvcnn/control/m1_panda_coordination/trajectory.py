"""Seeded band-limited six-dimensional pose trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import require_tensor


@dataclass(frozen=True)
class BandLimitedTrajectoryCfg:
    position_amplitude: float = 0.08
    orientation_amplitude: float = 0.15
    minimum_frequency_hz: float = 0.05
    maximum_frequency_hz: float = 0.25
    component_count: int = 4


@dataclass(frozen=True)
class TrajectorySample:
    pose: torch.Tensor
    twist: torch.Tensor
    acceleration: torch.Tensor


class BandLimitedPoseTrajectory:
    """Analytic sum-of-sinusoids trajectory with bounded axis amplitudes."""

    def __init__(self, cfg: BandLimitedTrajectoryCfg | None = None):
        self.cfg = cfg or BandLimitedTrajectoryCfg()
        if (
            isinstance(self.cfg.component_count, bool)
            or not isinstance(self.cfg.component_count, int)
            or self.cfg.component_count <= 0
        ):
            raise ValueError("component_count must be a positive integer")
        if (
            not math.isfinite(self.cfg.minimum_frequency_hz)
            or not math.isfinite(self.cfg.maximum_frequency_hz)
            or self.cfg.minimum_frequency_hz <= 0.0
            or self.cfg.maximum_frequency_hz < self.cfg.minimum_frequency_hz
        ):
            raise ValueError("trajectory frequency range must be finite and positive")
        if self.cfg.position_amplitude < 0.0 or self.cfg.orientation_amplitude < 0.0:
            raise ValueError("trajectory amplitudes must be non-negative")
        self._center: torch.Tensor | None = None
        self._frequencies_hz: torch.Tensor | None = None
        self._amplitudes: torch.Tensor | None = None

    @property
    def frequencies_hz(self) -> torch.Tensor:
        if self._frequencies_hz is None:
            raise RuntimeError("trajectory must be reset before reading frequencies")
        return self._frequencies_hz.clone()

    def reset(self, center_pose: torch.Tensor, *, seed: int) -> None:
        require_tensor("center_pose", center_pose, trailing_shape=(6,))
        if center_pose.ndim != 1 or not center_pose.is_floating_point():
            raise ValueError("center_pose must be one floating 6-vector")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        raw_amplitude = torch.rand(6, self.cfg.component_count, generator=generator)
        raw_amplitude /= raw_amplitude.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(raw_amplitude.dtype).eps
        )
        signs = torch.where(
            torch.rand(6, self.cfg.component_count, generator=generator) >= 0.5,
            1.0,
            -1.0,
        )
        limits = torch.tensor(
            [self.cfg.position_amplitude] * 3
            + [self.cfg.orientation_amplitude] * 3
        ).unsqueeze(-1)
        frequencies = self.cfg.minimum_frequency_hz + (
            self.cfg.maximum_frequency_hz - self.cfg.minimum_frequency_hz
        ) * torch.rand(6, self.cfg.component_count, generator=generator)

        self._center = center_pose.detach().clone()
        self._amplitudes = (raw_amplitude * signs * limits).to(
            device=center_pose.device, dtype=center_pose.dtype
        )
        self._frequencies_hz = frequencies.to(
            device=center_pose.device, dtype=center_pose.dtype
        )

    def sample(self, time_s: float) -> TrajectorySample:
        if self._center is None or self._amplitudes is None or self._frequencies_hz is None:
            raise RuntimeError("trajectory must be reset before sampling")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
            raise TypeError("time_s must be a real number")
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")

        angular_frequency = 2.0 * math.pi * self._frequencies_hz
        phase = angular_frequency * time_s
        # A half raised cosine preserves the configured amplitude bound while
        # starting every episode with continuous zero Cartesian velocity.
        offset = (0.5 * self._amplitudes * (1.0 - torch.cos(phase))).sum(dim=-1)
        twist = (
            0.5 * self._amplitudes * angular_frequency * torch.sin(phase)
        ).sum(dim=-1)
        acceleration = (
            0.5 * self._amplitudes * angular_frequency.square() * torch.cos(phase)
        ).sum(dim=-1)
        return TrajectorySample(
            pose=self._center + offset,
            twist=twist,
            acceleration=acceleration,
        )
