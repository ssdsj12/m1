"""Pure disturbance scheduling helpers for M1 + Panda Teacher training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import re

import torch


HOLD_MODE = 0
RAMP_MODE = 1
PULSE_MODE = 2


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    return (
        vector * (2.0 * quat[..., :1].square() - 1.0)
        + 2.0
        * quat[..., :1]
        * torch.linalg.cross(quat_vector, vector, dim=-1)
        + 2.0
        * quat_vector
        * torch.sum(quat_vector * vector, dim=-1, keepdim=True)
    )


def _quat_rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    return (
        vector * (2.0 * quat[..., :1].square() - 1.0)
        - 2.0
        * quat[..., :1]
        * torch.linalg.cross(quat_vector, vector, dim=-1)
        + 2.0
        * quat_vector
        * torch.sum(quat_vector * vector, dim=-1, keepdim=True)
    )


def base_wrench_to_body_local(
    force_b: torch.Tensor,
    torque_b: torch.Tensor,
    base_quat_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a BASE_LINK-frame wrench into live target-body local axes."""
    tensors = {
        "force_b": force_b,
        "torque_b": torque_b,
        "base_quat_w": base_quat_w,
        "body_quat_w": body_quat_w,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    reference = force_b
    for name, value in tensors.items():
        if value.device != reference.device:
            raise ValueError(
                f"{name} must be on device {reference.device}, got {value.device}"
            )
        if value.dtype != reference.dtype:
            raise TypeError(
                f"{name} must have dtype {reference.dtype}, got {value.dtype}"
            )
    if force_b.shape != torque_b.shape or force_b.ndim < 1 or force_b.shape[-1] != 3:
        raise ValueError("force_b and torque_b must have identical shape [..., 3]")
    expected_quat_shape = force_b.shape[:-1] + (4,)
    if base_quat_w.shape != expected_quat_shape or body_quat_w.shape != expected_quat_shape:
        raise ValueError(
            "base_quat_w and body_quat_w must match the wrench batch shape with last dimension 4"
        )

    force_w = _quat_rotate(base_quat_w, force_b)
    torque_w = _quat_rotate(base_quat_w, torque_b)
    return (
        _quat_rotate_inverse(body_quat_w, force_w),
        _quat_rotate_inverse(body_quat_w, torque_w),
    )


def clear_external_wrench(robot) -> None:
    """Disable persistent wrenches across supported Isaac Lab 2.1/5.1 APIs."""
    empty = torch.zeros(0, 3, device=robot.device)
    try:
        robot.set_external_force_and_torque(empty, empty)
    except RuntimeError as error:
        warp_empty_assignment = (
            "argument 'forces' expects an array with 2 dimension(s) but "
            "the passed array has 1 dimension(s)"
        ) in str(error)
        if warp_empty_assignment:
            zeros = torch.zeros(
                robot.num_instances,
                robot.num_bodies,
                3,
                device=robot.device,
            )
            robot.set_external_force_and_torque(zeros, zeros)
            return
        known_empty_assignment = re.fullmatch(
            r"shape mismatch: value tensor of shape \[0\] cannot be broadcast to indexing result "
            r"of shape \[\d+, 3\]",
            str(error),
        )
        if known_empty_assignment is None or robot.has_external_wrench is not False:
            raise

@dataclass(frozen=True)
class M1PandaDisturbanceCfg:
    """Validated six-dimensional disturbance curriculum configuration."""

    force_limit_n: tuple[float, float, float]
    torque_limit_nm: tuple[float, float, float]
    hold_time_min_s: float
    hold_time_max_s: float
    curriculum_start_scale: float = 0.25
    curriculum_steps: int = 50_000
    mode_probabilities: tuple[float, float, float] = (1.0, 0.0, 0.0)
    pulse_on_fraction: float = 0.20

    def __post_init__(self) -> None:
        for name, values in (
            ("force_limit_n", self.force_limit_n),
            ("torque_limit_nm", self.torque_limit_nm),
        ):
            if len(values) != 3 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
                for value in values
            ):
                raise ValueError(f"{name} must contain three finite positive values")
        if (
            isinstance(self.hold_time_min_s, bool)
            or isinstance(self.hold_time_max_s, bool)
            or not isinstance(self.hold_time_min_s, (int, float))
            or not isinstance(self.hold_time_max_s, (int, float))
            or not math.isfinite(self.hold_time_min_s)
            or not math.isfinite(self.hold_time_max_s)
            or self.hold_time_min_s <= 0.0
            or self.hold_time_min_s > self.hold_time_max_s
        ):
            raise ValueError("hold times must be finite, positive, and ordered")
        if (
            isinstance(self.curriculum_start_scale, bool)
            or not isinstance(self.curriculum_start_scale, (int, float))
            or not math.isfinite(self.curriculum_start_scale)
            or not 0.0 < self.curriculum_start_scale <= 1.0
        ):
            raise ValueError("curriculum_start_scale must be in (0, 1]")
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
            raise ValueError(
                "mode_probabilities must be three nonnegative values summing to one"
            )
        if (
            isinstance(self.pulse_on_fraction, bool)
            or not isinstance(self.pulse_on_fraction, (int, float))
            or not math.isfinite(self.pulse_on_fraction)
            or not 0.0 < self.pulse_on_fraction <= 1.0
        ):
            raise ValueError("pulse_on_fraction must be in (0, 1]")


def stage_disturbance_cfg(stage: str) -> M1PandaDisturbanceCfg:
    """Return the exact approved disturbance defaults for one Teacher stage."""
    if stage == "A0":
        return M1PandaDisturbanceCfg(
            force_limit_n=(10.0, 10.0, 10.0),
            torque_limit_nm=(2.0, 2.0, 2.0),
            hold_time_min_s=1.0,
            hold_time_max_s=2.0,
            curriculum_steps=50_000,
        )
    if stage == "A1":
        return M1PandaDisturbanceCfg(
            force_limit_n=(20.0, 20.0, 20.0),
            torque_limit_nm=(5.0, 5.0, 5.0),
            hold_time_min_s=0.25,
            hold_time_max_s=1.0,
            curriculum_steps=75_000,
            mode_probabilities=(0.50, 0.30, 0.20),
            pulse_on_fraction=0.20,
        )
    raise ValueError(f"stage must be 'A0' or 'A1', got {stage!r}")


class M1PandaDisturbanceScheduler:
    """Generate independent, seeded base-frame wrench segments per environment."""

    def __init__(
        self,
        cfg: M1PandaDisturbanceCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
        initial_global_step: int = 0,
    ) -> None:
        if not isinstance(cfg, M1PandaDisturbanceCfg):
            raise TypeError("cfg must be an M1PandaDisturbanceCfg")
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        if (
            isinstance(step_dt, bool)
            or not isinstance(step_dt, (int, float))
            or not math.isfinite(step_dt)
            or step_dt <= 0.0
        ):
            raise ValueError("step_dt must be finite and positive")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(initial_global_step, int)
            or isinstance(initial_global_step, bool)
            or initial_global_step < 0
        ):
            raise ValueError(
                "initial_global_step must be a nonnegative integer"
            )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch.dtype")

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.dtype = dtype
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._current = torch.zeros(num_envs, 6, device=self.device, dtype=dtype)
        self._start = torch.zeros_like(self._current)
        self._target = torch.zeros_like(self._current)
        self._duration_steps = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self._elapsed_steps = torch.zeros_like(self._duration_steps)
        self._remaining_steps = torch.zeros_like(self._duration_steps)
        self._mode = torch.full_like(self._duration_steps, HOLD_MODE)
        self._global_step = initial_global_step
        self._limits = torch.tensor(
            cfg.force_limit_n + cfg.torque_limit_nm,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        self._mode_probabilities = torch.tensor(
            cfg.mode_probabilities, device=self.device, dtype=self.dtype
        )
        self._min_duration_steps = max(
            1, math.ceil(cfg.hold_time_min_s / self.step_dt)
        )
        self._max_duration_steps = max(
            self._min_duration_steps,
            math.ceil(cfg.hold_time_max_s / self.step_dt),
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
        self._start[env_ids] = self._current[env_ids]
        uniform = torch.rand(
            count,
            6,
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )
        self._target[env_ids] = (
            uniform.mul(2.0).sub(1.0)
            * self._limits
            * self.curriculum_scale
        )
        durations = torch.randint(
            self._min_duration_steps,
            self._max_duration_steps + 1,
            (count,),
            device=self.device,
            generator=self._generator,
        )
        self._duration_steps[env_ids] = durations
        self._elapsed_steps[env_ids] = 0
        self._remaining_steps[env_ids] = durations
        self._mode[env_ids] = torch.multinomial(
            self._mode_probabilities,
            num_samples=count,
            replacement=True,
            generator=self._generator,
        )

    def advance(self) -> torch.Tensor:
        """Advance all segment envelopes by one control step."""
        needs_sample = self._remaining_steps == 0
        if bool(needs_sample.any()):
            self._sample_segments(needs_sample.nonzero(as_tuple=False).flatten())

        fraction = (self._elapsed_steps + 1).to(self.dtype) / self._duration_steps.to(
            self.dtype
        )
        hold = self._target
        ramp = self._start + fraction.unsqueeze(1) * (self._target - self._start)
        pulse = torch.where(
            (fraction <= self.cfg.pulse_on_fraction).unsqueeze(1),
            self._target,
            torch.zeros_like(self._target),
        )
        self._current = torch.where(
            (self._mode == HOLD_MODE).unsqueeze(1),
            hold,
            torch.where((self._mode == RAMP_MODE).unsqueeze(1), ramp, pulse),
        )
        self._elapsed_steps += 1
        self._remaining_steps -= 1
        self._global_step += 1
        return self._current.clone()

    def _normalize_env_ids(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if env_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("env_ids must contain integers")
            values = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            values = list(env_ids)
        else:
            raise TypeError("env_ids must be an integer tensor or sequence")
        if any(
            not isinstance(index, int) or isinstance(index, bool) for index in values
        ):
            raise TypeError("env_ids must contain integers")
        for index in values:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"env_ids contains out-of-range index {index}")
        return torch.tensor(values, device=self.device, dtype=torch.long)

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        """Clear all segment state, or only selected environments."""
        if env_ids is None:
            normalized_ids = torch.arange(self.num_envs, device=self.device)
        else:
            normalized_ids = self._normalize_env_ids(env_ids)
        self._current[normalized_ids] = 0
        self._start[normalized_ids] = 0
        self._target[normalized_ids] = 0
        self._duration_steps[normalized_ids] = 0
        self._elapsed_steps[normalized_ids] = 0
        self._remaining_steps[normalized_ids] = 0
        self._mode[normalized_ids] = HOLD_MODE

    @property
    def current_wrench_b(self) -> torch.Tensor:
        return self._current.clone()

    @property
    def target_wrench_b(self) -> torch.Tensor:
        return self._target.clone()

    @property
    def duration_steps(self) -> torch.Tensor:
        return self._duration_steps.clone()

    @property
    def remaining_steps(self) -> torch.Tensor:
        return self._remaining_steps.clone()


__all__ = [
    "HOLD_MODE",
    "RAMP_MODE",
    "PULSE_MODE",
    "M1PandaDisturbanceCfg",
    "M1PandaDisturbanceScheduler",
    "base_wrench_to_body_local",
    "clear_external_wrench",
    "stage_disturbance_cfg",
]
