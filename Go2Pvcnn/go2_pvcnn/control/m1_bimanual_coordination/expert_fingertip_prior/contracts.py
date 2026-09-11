"""Frozen, hand-agnostic contracts for the offline fingertip-motion prior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import re

import torch


DEXMANIPNET_REVISION = "3933fae5fe83498fb314a0924aa21d5038fba5a5"
MANIPTRANS_COMMIT = "a3d08cfe3c3a5868a7f057533bcaf759c5af4705"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
MODEL_INPUT_DIM = 42
MIXTURE_COMPONENTS = 4
PRIOR_HORIZON = 20
PRIOR_DT = 0.01
STUDENT_ARTIFACT_FORMAT_VERSION = 1


class PriorPhase(IntEnum):
    APPROACH = 0
    PRELOAD = 1
    GRASP = 2
    MANIPULATE = 3
    HOLD = 4
    RELEASE = 5
    UNKNOWN = 6


PHASE_ORDER = tuple(PriorPhase)
LEFT_REFLECTION = torch.diag(torch.tensor((1.0, -1.0, 1.0), dtype=torch.float32))
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _validate_float_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class ExpertWindow:
    """One converted 100 Hz geometry-only training example."""

    fingertip_position_palm: torch.Tensor
    fingertip_velocity_palm: torch.Tensor
    contact_mask: torch.Tensor
    phase: PriorPhase
    future_fingertip_velocity_palm: torch.Tensor
    source_group: str
    source_sha256: str

    def __post_init__(self) -> None:
        _validate_float_tensor("fingertip_position_palm", self.fingertip_position_palm, (5, 3))
        _validate_float_tensor("fingertip_velocity_palm", self.fingertip_velocity_palm, (5, 3))
        if not isinstance(self.contact_mask, torch.Tensor):
            raise TypeError("contact_mask must be a torch.Tensor")
        if self.contact_mask.dtype != torch.bool:
            raise TypeError("contact_mask must have dtype torch.bool")
        if self.contact_mask.shape != (5,):
            raise ValueError("contact_mask must have shape (5,)")
        if not isinstance(self.phase, PriorPhase):
            raise TypeError("phase must be a PriorPhase")
        _validate_float_tensor(
            "future_fingertip_velocity_palm",
            self.future_fingertip_velocity_palm,
            (PRIOR_HORIZON, 5, 3),
        )
        tensors = (
            self.fingertip_position_palm,
            self.fingertip_velocity_palm,
            self.contact_mask,
            self.future_fingertip_velocity_palm,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("expert window tensors must share a device")
        if not isinstance(self.source_group, str) or not self.source_group:
            raise ValueError("source_group must be a non-empty string")
        _validate_sha256("source_sha256", self.source_sha256)

    def network_input(self) -> torch.Tensor:
        """Return the frozen 15 + 15 + 5 + 7 geometry-only input order."""

        phase = torch.nn.functional.one_hot(
            torch.tensor(int(self.phase), device=self.contact_mask.device),
            num_classes=len(PHASE_ORDER),
        ).to(dtype=torch.float32)
        values = torch.cat(
            (
                self.fingertip_position_palm.reshape(-1),
                self.fingertip_velocity_palm.reshape(-1),
                self.contact_mask.to(dtype=torch.float32),
                phase,
            )
        )
        if values.shape != (MODEL_INPUT_DIM,):
            raise RuntimeError("expert window network input violates the frozen dimension")
        return values

    @property
    def target(self) -> torch.Tensor:
        return self.future_fingertip_velocity_palm


@dataclass(frozen=True)
class MixtureDistribution:
    """Four-component diagonal Gaussian distribution over future tip velocities."""

    logits: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {"logits": self.logits, "mean": self.mean, "log_std": self.log_std}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must have a floating-point dtype")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
        if self.logits.ndim < 1 or self.logits.shape[-1] != MIXTURE_COMPONENTS:
            raise ValueError("logits must have trailing shape (4,)")
        expected = (*self.logits.shape[:-1], MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3)
        if self.mean.shape != expected or self.log_std.shape != expected:
            raise ValueError(f"mean and log_std must have shape {expected}")
        if len({value.dtype for value in tensors.values()}) != 1:
            raise TypeError("mixture tensors must share a dtype")
        if len({value.device for value in tensors.values()}) != 1:
            raise ValueError("mixture tensors must share a device")


@dataclass(frozen=True)
class StudentArtifactMetadata:
    """Strict metadata accompanying the only deployable prior artifact."""

    format_version: int
    input_dim: int
    mixture_components: int
    horizon: int
    dt: float
    finger_order: tuple[str, ...]
    phase_order: tuple[PriorPhase, ...]
    mirror_matrix: torch.Tensor
    dataset_aggregate_sha256: str
    teacher_ensemble_manifest_sha256: str
    teacher_seed: int
    distillation_seed: int
    code_commit: str
    weight_sha256: str
    hidden: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = (
            ("format_version", self.format_version, STUDENT_ARTIFACT_FORMAT_VERSION),
            ("input_dim", self.input_dim, MODEL_INPUT_DIM),
            ("mixture_components", self.mixture_components, MIXTURE_COMPONENTS),
            ("horizon", self.horizon, PRIOR_HORIZON),
        )
        for name, value, required in expected:
            if value != required:
                raise ValueError(f"{name} must equal {required}")
        if not isinstance(self.dt, float) or self.dt != PRIOR_DT:
            raise ValueError(f"dt must equal {PRIOR_DT}")
        if self.finger_order != FINGER_ORDER:
            raise ValueError("finger_order does not match the frozen order")
        if self.phase_order != PHASE_ORDER:
            raise ValueError("phase_order does not match the frozen order")
        _validate_float_tensor("mirror_matrix", self.mirror_matrix, (3, 3))
        if not torch.equal(self.mirror_matrix, LEFT_REFLECTION):
            raise ValueError("mirror_matrix does not match the frozen left reflection")
        for name in ("dataset_aggregate_sha256", "teacher_ensemble_manifest_sha256", "weight_sha256"):
            _validate_sha256(name, getattr(self, name))
        if not isinstance(self.code_commit, str) or _COMMIT_RE.fullmatch(self.code_commit) is None:
            raise ValueError("code_commit must be a lowercase 40-character commit SHA")
        for name in ("teacher_seed", "distillation_seed"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        if not self.hidden or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.hidden
        ):
            raise ValueError("hidden must contain positive integer widths")


__all__ = [
    "DEXMANIPNET_REVISION",
    "FINGER_ORDER",
    "LEFT_REFLECTION",
    "MANIPTRANS_COMMIT",
    "MIXTURE_COMPONENTS",
    "MODEL_INPUT_DIM",
    "PHASE_ORDER",
    "PRIOR_DT",
    "PRIOR_HORIZON",
    "STUDENT_ARTIFACT_FORMAT_VERSION",
    "ExpertWindow",
    "MixtureDistribution",
    "PriorPhase",
    "StudentArtifactMetadata",
]
