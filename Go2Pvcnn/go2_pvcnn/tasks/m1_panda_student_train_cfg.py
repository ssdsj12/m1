"""Small explicit supervised-training configuration for Student S1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudentTrainCfg:
    stage: str
    teacher_probability: float | None
    learning_rate: float = 3.0e-4
    batch_size: int = 512
    epochs: int = 20
    seed: int = 17
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be non-empty")
        if self.teacher_probability is not None and not 0.0 <= self.teacher_probability <= 1.0:
            raise ValueError("teacher_probability must be in [0,1]")
        if self.learning_rate <= 0.0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate, batch_size and epochs must be positive")


__all__ = ["StudentTrainCfg"]
