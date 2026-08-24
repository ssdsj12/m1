"""Seeded Panda-hand wrench curriculum for coordinated M1 + Panda training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from go2_pvcnn.tasks.m1_panda_teacher import base_wrench_to_body_local


CONTINUOUS_MODE = 0
PULSE_MODE = 1
INTERMITTENT_MODE = 2


@dataclass(frozen=True)
class CoordinatedDisturbanceCfg:
    force_limit_n: float = 20.0
    torque_limit_nm: float = 5.0
    hold_time_min_s: float = 0.25
    hold_time_max_s: float = 1.0
    curriculum_start_scale: float = 0.10
    curriculum_steps: int = 50_000
    mode_probabilities: tuple[float, float, float] = (0.50, 0.30, 0.20)
    pulse_on_fraction: float = 0.20
    intermittent_period_s: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "force_limit_n",
            "torque_limit_nm",
            "hold_time_min_s",
            "hold_time_max_s",
            "curriculum_start_scale",
            "pulse_on_fraction",
            "intermittent_period_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.hold_time_min_s > self.hold_time_max_s:
            raise ValueError("hold times must be ordered")
        if self.curriculum_start_scale > 1.0:
            raise ValueError("curriculum_start_scale must not exceed one")
        if self.pulse_on_fraction > 1.0:
            raise ValueError("pulse_on_fraction must not exceed one")
        if (
            not isinstance(self.curriculum_steps, int)
            or isinstance(self.curriculum_steps, bool)
            or self.curriculum_steps <= 0
        ):
            raise ValueError("curriculum_steps must be a positive integer")
        if (
            len(self.mode_probabilities) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                for value in self.mode_probabilities
            )
            or not math.isclose(
                sum(self.mode_probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-8
            )
        ):
            raise ValueError("mode probabilities must be nonnegative and sum to one")


class CoordinatedDisturbanceScheduler:
    """Generate independent base-frame wrench segments at the control frequency."""

    def __init__(
        self,
        cfg: CoordinatedDisturbanceCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(cfg, CoordinatedDisturbanceCfg):
            raise TypeError("cfg must be CoordinatedDisturbanceCfg")
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if (
            isinstance(step_dt, bool)
            or not isinstance(step_dt, (int, float))
            or not math.isfinite(step_dt)
            or step_dt <= 0.0
        ):
            raise ValueError("step_dt must be finite and positive")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch dtype")

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.dtype = dtype
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._global_step = 0
        self._current = torch.zeros(num_envs, 6, device=self.device, dtype=dtype)
        self._target = torch.zeros_like(self._current)
        self._duration = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._elapsed = torch.zeros_like(self._duration)
        self._remaining = torch.zeros_like(self._duration)
        self._mode = torch.zeros_like(self._duration)
        self._limits = torch.tensor(
            [cfg.force_limit_n] * 3 + [cfg.torque_limit_nm] * 3,
            device=self.device,
            dtype=dtype,
        )
        self._probabilities = torch.tensor(
            cfg.mode_probabilities, device=self.device, dtype=dtype
        )
        self._minimum_duration = max(1, math.ceil(cfg.hold_time_min_s / step_dt))
        self._maximum_duration = max(
            self._minimum_duration, math.ceil(cfg.hold_time_max_s / step_dt)
        )
        self._intermittent_period = max(
            1, math.ceil(cfg.intermittent_period_s / step_dt)
        )
        self._intermittent_on_steps = max(
            1, math.ceil(self._intermittent_period * cfg.pulse_on_fraction)
        )

    @property
    def curriculum_scale(self) -> float:
        progress = min(self._global_step / self.cfg.curriculum_steps, 1.0)
        return self.cfg.curriculum_start_scale + (
            1.0 - self.cfg.curriculum_start_scale
        ) * progress

    @property
    def global_step(self) -> int:
        return self._global_step

    def _sample_segments(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        if count == 0:
            return
        uniform = torch.rand(
            count,
            6,
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )
        self._target[env_ids] = (
            uniform.mul(2.0).sub(1.0) * self._limits * self.curriculum_scale
        )
        duration = torch.randint(
            self._minimum_duration,
            self._maximum_duration + 1,
            (count,),
            device=self.device,
            generator=self._generator,
        )
        self._duration[env_ids] = duration
        self._elapsed[env_ids] = 0
        self._remaining[env_ids] = duration
        self._mode[env_ids] = torch.multinomial(
            self._probabilities,
            num_samples=count,
            replacement=True,
            generator=self._generator,
        )

    def advance(self) -> torch.Tensor:
        needs_sample = self._remaining == 0
        if bool(needs_sample.any()):
            self._sample_segments(needs_sample.nonzero(as_tuple=False).flatten())

        pulse_steps = torch.ceil(
            self._duration.to(self.dtype) * self.cfg.pulse_on_fraction
        ).to(torch.long)
        continuous_on = self._mode == CONTINUOUS_MODE
        pulse_on = (self._mode == PULSE_MODE) & (self._elapsed < pulse_steps)
        intermittent_on = (self._mode == INTERMITTENT_MODE) & (
            (self._elapsed % self._intermittent_period)
            < self._intermittent_on_steps
        )
        active = continuous_on | pulse_on | intermittent_on
        self._current = torch.where(
            active.unsqueeze(1), self._target, torch.zeros_like(self._target)
        )
        self._elapsed += 1
        self._remaining -= 1
        self._global_step += 1
        return self._current.clone()

    def _normalize_env_ids(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("env_ids must be a one-dimensional integer tensor")
            values = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            values = list(env_ids)
        else:
            raise TypeError("env_ids must be an integer tensor or sequence")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("env_ids must contain integers")
        if any(value < 0 or value >= self.num_envs for value in values):
            raise IndexError("env_ids contains an out-of-range index")
        return torch.tensor(values, device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset selected segment rows without rewinding the curriculum clock."""
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._normalize_env_ids(env_ids)
        )
        self._current[ids] = 0
        self._target[ids] = 0
        self._duration[ids] = 0
        self._elapsed[ids] = 0
        self._remaining[ids] = 0
        self._mode[ids] = CONTINUOUS_MODE

    @property
    def current_wrench_b(self) -> torch.Tensor:
        return self._current.clone()


__all__ = [
    "CONTINUOUS_MODE",
    "INTERMITTENT_MODE",
    "PULSE_MODE",
    "CoordinatedDisturbanceCfg",
    "CoordinatedDisturbanceScheduler",
    "base_wrench_to_body_local",
]
