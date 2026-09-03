"""Frozen feature, normalization, and artifact contracts for z16 control."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .contracts import BimanualSnapshot
from .full_action_teacher import TeacherSolution


STATE_DIM = 111
ACTION_DIM = 43
HORIZON = 25
TASK_FEATURE_DIM = 50
LATENT_DIM = 16
MODEL_FORMAT_VERSION = 1
LATENT_GROUPS = {
    "support": (0, 4),
    "palms": (4, 10),
    "hands": (10, 14),
    "platform": (14, 16),
}


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}.{index}" for index in range(count))


FEATURE_ORDER = (
    *_names("base_state", 13),
    *_names("m1_q", 16),
    *_names("m1_qd", 16),
    *_names("platform_q_qd", 2),
    *_names("left_arm.q", 7),
    *_names("left_arm.qd", 7),
    *_names("right_arm.q", 7),
    *_names("right_arm.qd", 7),
    *_names("left_hand.q", 6),
    *_names("left_hand.qd", 6),
    *_names("right_hand.q", 6),
    *_names("right_hand.qd", 6),
    *_names("box.pose_b", 6),
    *_names("box.twist_b", 6),
)
assert len(FEATURE_ORDER) == STATE_DIM


def pack_state_features(snapshot: BimanualSnapshot) -> torch.Tensor:
    """Pack the frozen 111-value state order as a CPU float32 vector."""

    if not isinstance(snapshot, BimanualSnapshot):
        raise TypeError("snapshot must be BimanualSnapshot")
    values = torch.cat(
        (
            snapshot.base_state,
            snapshot.m1_q,
            snapshot.m1_qd,
            snapshot.platform_q_qd,
            snapshot.left_arm.q,
            snapshot.left_arm.qd,
            snapshot.right_arm.q,
            snapshot.right_arm.qd,
            snapshot.left_hand.q,
            snapshot.left_hand.qd,
            snapshot.right_hand.q,
            snapshot.right_hand.qd,
            snapshot.box.pose_b,
            snapshot.box.twist_b,
        )
    )
    if values.shape != (STATE_DIM,) or not torch.isfinite(values).all().item():
        raise ValueError("packed state must be finite with shape (111,)")
    return values.to(device="cpu", dtype=torch.float32)


def pack_teacher_task_features(solution: TeacherSolution) -> torch.Tensor:
    """Pack each teacher node as box/palms/wrenches/platform = 50 values."""

    if not isinstance(solution, TeacherSolution):
        raise TypeError("solution must be TeacherSolution")
    values = torch.cat(
        (
            solution.box_trajectory_b,
            solution.left_palm_trajectory_b,
            solution.right_palm_trajectory_b,
            solution.left_wrench_b,
            solution.right_wrench_b,
            solution.platform_trajectory,
        ),
        dim=1,
    )
    if values.shape != (HORIZON, TASK_FEATURE_DIM):
        raise ValueError("teacher task features must have shape (25, 50)")
    return values.to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class LatentNormalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("mean", "scale"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            value = value.detach().to(device="cpu", dtype=torch.float32).clone()
            if value.shape != (STATE_DIM,) or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite with shape (111,)")
            object.__setattr__(self, name, value)
        if not torch.all(self.scale > 0.0).item():
            raise ValueError("normalization scale must be strictly positive")

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value.to(dtype=torch.float32) - self.mean) / self.scale

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=torch.float32) * self.scale + self.mean


@dataclass(frozen=True)
class LatentArtifactMetadata:
    format_version: int
    state_dim: int
    action_dim: int
    horizon: int
    task_feature_dim: int
    latent_dim: int
    action_order: tuple[str, ...]
    feature_order: tuple[str, ...]
    normalization_sha256: str
    dataset_sha256: str
    training_seed: int

    def __post_init__(self) -> None:
        expected = (
            ("format_version", self.format_version, MODEL_FORMAT_VERSION),
            ("state_dim", self.state_dim, STATE_DIM),
            ("action_dim", self.action_dim, ACTION_DIM),
            ("horizon", self.horizon, HORIZON),
            ("task_feature_dim", self.task_feature_dim, TASK_FEATURE_DIM),
            ("latent_dim", self.latent_dim, LATENT_DIM),
        )
        for name, value, required in expected:
            if value != required:
                raise ValueError(f"{name} must equal {required}")
        if len(self.action_order) != ACTION_DIM or len(set(self.action_order)) != ACTION_DIM:
            raise ValueError("action_order must contain 43 unique names")
        if self.feature_order != FEATURE_ORDER:
            raise ValueError("feature_order does not match the frozen order")
        for name in ("normalization_sha256", "dataset_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise TypeError("training_seed must be an integer")

    def validate_runtime(
        self,
        *,
        action_order: tuple[str, ...],
        state_dim: int,
        normalization_sha256: str,
    ) -> None:
        if action_order != self.action_order:
            raise ValueError("action_order is incompatible with the model")
        if state_dim != self.state_dim:
            raise ValueError("state_dim is incompatible with the model")
        if normalization_sha256 != self.normalization_sha256:
            raise ValueError("normalization SHA-256 is incompatible with the model")


__all__ = [
    "ACTION_DIM",
    "FEATURE_ORDER",
    "HORIZON",
    "LATENT_DIM",
    "LATENT_GROUPS",
    "MODEL_FORMAT_VERSION",
    "STATE_DIM",
    "TASK_FEATURE_DIM",
    "LatentArtifactMetadata",
    "LatentNormalizer",
    "pack_state_features",
    "pack_teacher_task_features",
]
