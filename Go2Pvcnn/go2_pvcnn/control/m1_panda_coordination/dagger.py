"""Deterministic DAgger action selection and supervised Student S1 losses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .student_contracts import STUDENT_ACTION_DIM
from .student_model import StudentOutput


@dataclass(frozen=True)
class DaggerStageCfg:
    name: str
    teacher_probability: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if (
            not isinstance(self.teacher_probability, (int, float))
            or isinstance(self.teacher_probability, bool)
            or not math.isfinite(float(self.teacher_probability))
            or not 0.0 <= float(self.teacher_probability) <= 1.0
        ):
            raise ValueError("teacher_probability must be finite and in [0,1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class DaggerSelection:
    executed: torch.Tensor
    teacher_executed: torch.Tensor
    safe_to_execute_student: torch.Tensor


def _require_action(value: torch.Tensor, *, label: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != STUDENT_ACTION_DIM:
        raise ValueError(
            f"{label} must have shape [E,{STUDENT_ACTION_DIM}], got {tuple(value.shape)}"
        )
    if not value.dtype.is_floating_point:
        raise TypeError(f"{label} must have floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain only finite values")


def select_dagger_action(
    student: torch.Tensor,
    teacher: torch.Tensor,
    safe_to_execute_student: torch.Tensor,
    cfg: DaggerStageCfg,
    rollout_step: int,
) -> DaggerSelection:
    """Choose per-environment Teacher/Student actions reproducibly.

    Any unsafe Student proposal is unconditionally replaced by the Teacher label.
    The random stream is local and therefore cannot perturb simulation RNG state.
    """

    _require_action(student, label="student")
    _require_action(teacher, label="teacher")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher must have the same shape")
    if student.device != teacher.device or student.dtype != teacher.dtype:
        raise ValueError("student and teacher must share device and dtype")
    if not isinstance(safe_to_execute_student, torch.Tensor):
        raise TypeError("safe_to_execute_student must be a torch.Tensor")
    if safe_to_execute_student.dtype != torch.bool:
        raise TypeError("safe_to_execute_student must be a boolean tensor")
    if safe_to_execute_student.shape != (student.shape[0],):
        raise ValueError("safe_to_execute_student must have shape [E]")
    if safe_to_execute_student.device != student.device:
        raise ValueError("safe_to_execute_student must share the action device")
    if not isinstance(cfg, DaggerStageCfg):
        raise TypeError("cfg must be a DaggerStageCfg")
    if (
        not isinstance(rollout_step, int)
        or isinstance(rollout_step, bool)
        or rollout_step < 0
    ):
        raise ValueError("rollout_step must be a nonnegative integer")

    generator = torch.Generator(device=student.device)
    generator.manual_seed(cfg.seed + rollout_step)
    teacher_executed = (
        torch.rand(student.shape[0], device=student.device, generator=generator)
        < float(cfg.teacher_probability)
    ) | ~safe_to_execute_student
    executed = torch.where(teacher_executed.unsqueeze(-1), teacher, student)
    return DaggerSelection(
        executed=executed,
        teacher_executed=teacher_executed,
        safe_to_execute_student=safe_to_execute_student.clone(),
    )


@dataclass(frozen=True)
class StudentLossCfg:
    action: float = 1.0
    wrench: float = 0.25
    safety: float = 0.25
    slew: float = 0.05
    saturation: float = 0.05
    hard_sample_multiplier: float = 2.0

    def __post_init__(self) -> None:
        for name in ("action", "wrench", "safety", "slew", "saturation"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not isinstance(self.hard_sample_multiplier, (int, float))
            or isinstance(self.hard_sample_multiplier, bool)
            or not math.isfinite(float(self.hard_sample_multiplier))
            or float(self.hard_sample_multiplier) < 1.0
        ):
            raise ValueError("hard_sample_multiplier must be finite and at least 1")


def _require_loss_tensor(
    value: torch.Tensor,
    *,
    label: str,
    shape: tuple[int, ...],
    reference: torch.Tensor,
    boolean: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}, got {tuple(value.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{label} must be on device {reference.device}")
    if boolean:
        if value.dtype != torch.bool:
            raise TypeError(f"{label} must be boolean")
    else:
        if value.dtype != reference.dtype or not value.dtype.is_floating_point:
            raise TypeError(f"{label} must share the floating output dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must contain only finite values")


def student_dagger_loss(
    output: StudentOutput,
    target_action: torch.Tensor,
    target_wrench: torch.Tensor,
    target_safety: torch.Tensor,
    hard_mask: torch.Tensor,
    previous_action: torch.Tensor,
    cfg: StudentLossCfg,
) -> dict[str, torch.Tensor]:
    """Return the six scalar normalized S1 DAgger loss terms."""

    if not isinstance(output, StudentOutput):
        raise TypeError("output must be a StudentOutput")
    if not isinstance(cfg, StudentLossCfg):
        raise TypeError("cfg must be a StudentLossCfg")
    _require_action(output.action, label="output.action")
    batch = output.action.shape[0]
    reference = output.action
    _require_loss_tensor(
        output.raw_action,
        label="output.raw_action",
        shape=(batch, STUDENT_ACTION_DIM),
        reference=reference,
    )
    _require_loss_tensor(
        output.wrench_hat,
        label="output.wrench_hat",
        shape=(batch, 6),
        reference=reference,
    )
    _require_loss_tensor(
        output.safety_logit,
        label="output.safety_logit",
        shape=(batch, 1),
        reference=reference,
    )
    _require_loss_tensor(
        target_action,
        label="target_action",
        shape=(batch, STUDENT_ACTION_DIM),
        reference=reference,
    )
    _require_loss_tensor(
        target_wrench,
        label="target_wrench",
        shape=(batch, 6),
        reference=reference,
    )
    _require_loss_tensor(
        target_safety,
        label="target_safety",
        shape=(batch,),
        reference=reference,
    )
    _require_loss_tensor(
        hard_mask,
        label="hard_mask",
        shape=(batch,),
        reference=reference,
        boolean=True,
    )
    _require_loss_tensor(
        previous_action,
        label="previous_action",
        shape=(batch, STUDENT_ACTION_DIM),
        reference=reference,
    )
    if not bool(((target_safety >= 0.0) & (target_safety <= 1.0)).all()):
        raise ValueError("target_safety must be in [0,1]")

    weights = torch.where(
        hard_mask,
        torch.as_tensor(cfg.hard_sample_multiplier, device=reference.device, dtype=reference.dtype),
        torch.ones((), device=reference.device, dtype=reference.dtype),
    )

    def weighted_mean(per_sample: torch.Tensor) -> torch.Tensor:
        return (per_sample * weights).mean()

    action = weighted_mean((output.action - target_action).square().mean(dim=-1))
    wrench = weighted_mean((output.wrench_hat - target_wrench).square().mean(dim=-1))
    safety = weighted_mean(
        F.binary_cross_entropy_with_logits(
            output.safety_logit.squeeze(-1), target_safety, reduction="none"
        )
    )
    slew = weighted_mean(
        F.smooth_l1_loss(output.action, previous_action, reduction="none").mean(dim=-1)
    )
    saturation = weighted_mean(
        torch.relu(output.raw_action.abs() - 1.0).mean(dim=-1)
    )
    total = (
        float(cfg.action) * action
        + float(cfg.wrench) * wrench
        + float(cfg.safety) * safety
        + float(cfg.slew) * slew
        + float(cfg.saturation) * saturation
    )
    return {
        "total": total,
        "action": action,
        "wrench": wrench,
        "safety": safety,
        "slew": slew,
        "saturation": saturation,
    }


__all__ = [
    "DaggerSelection",
    "DaggerStageCfg",
    "StudentLossCfg",
    "select_dagger_action",
    "student_dagger_loss",
]
