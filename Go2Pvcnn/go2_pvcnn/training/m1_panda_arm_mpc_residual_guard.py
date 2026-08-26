"""Stability-first selection and atomic manifest for residual PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class ResidualEvalMetrics:
    hard_failure_count: int
    mpc_feasible_rate: float
    qp_feasible_rate: float
    four_contact_rate: float
    roll_pitch_rms: float
    base_height_rms: float
    ee_position_error: float
    ee_orientation_error: float
    wrench_error: float
    slip: float
    intervention_ratio: float
    saturation_fraction: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hard_failure_count, int) or isinstance(self.hard_failure_count, bool) or self.hard_failure_count < 0:
            raise ValueError("hard_failure_count must be a non-negative integer")
        if not isinstance(self.saturation_fraction, tuple) or len(self.saturation_fraction) != 8:
            raise ValueError("saturation_fraction must be an 8-tuple")
        for name, value in asdict(self).items():
            if name in ("hard_failure_count", "saturation_fraction"):
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in self.saturation_fraction):
            raise ValueError("saturation_fraction entries must be finite and in [0,1]")

    @property
    def rank(self) -> tuple[float, ...]:
        return (
            float(self.hard_failure_count),
            self.roll_pitch_rms,
            self.base_height_rms,
            self.ee_position_error,
            self.ee_orientation_error,
            self.intervention_ratio,
        )

    @property
    def eligible(self) -> bool:
        return (
            self.hard_failure_count == 0
            and self.mpc_feasible_rate >= 0.99
            and self.qp_feasible_rate >= 1.0
            and self.four_contact_rate >= 1.0
            and self.roll_pitch_rms <= math.radians(10.0)
            and self.ee_position_error <= 0.015
            and self.ee_orientation_error <= 0.08
            and max(self.saturation_fraction) < 0.01
        )


def metrics_better(candidate: ResidualEvalMetrics, incumbent: ResidualEvalMetrics) -> bool:
    if not isinstance(candidate, ResidualEvalMetrics) or not isinstance(incumbent, ResidualEvalMetrics):
        raise TypeError("candidate and incumbent must be ResidualEvalMetrics")
    return candidate.rank < incumbent.rank


@dataclass(frozen=True)
class ResidualGuardDecision:
    eligible: bool
    save_best: bool
    stop_reason: str | None
    rollback_checkpoint: str | None


class ResidualTrainingGuard:
    def __init__(self, *, patience_updates: int = 50) -> None:
        if not isinstance(patience_updates, int) or isinstance(patience_updates, bool) or patience_updates <= 0:
            raise ValueError("patience_updates must be a positive integer")
        self.patience_updates = patience_updates
        self.best_metrics: ResidualEvalMetrics | None = None
        self.best_checkpoint: str | None = None
        self.best_iteration: int | None = None
        self._without_improvement = 0

    @property
    def accepted(self) -> bool:
        return self.best_metrics is not None

    def observe(self, iteration: int, metrics: ResidualEvalMetrics, checkpoint: str) -> ResidualGuardDecision:
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        if not isinstance(metrics, ResidualEvalMetrics):
            raise TypeError("metrics must be ResidualEvalMetrics")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("checkpoint must be a non-empty string")
        improved = False
        if metrics.eligible and (
            self.best_metrics is None or metrics_better(metrics, self.best_metrics)
        ):
            self.best_metrics = metrics
            self.best_checkpoint = checkpoint
            self.best_iteration = iteration
            self._without_improvement = 0
            improved = True
        elif self.best_metrics is not None:
            self._without_improvement += 1
        stop = (
            "eligible_patience"
            if self.best_metrics is not None and self._without_improvement >= self.patience_updates
            else None
        )
        return ResidualGuardDecision(
            eligible=metrics.eligible,
            save_best=improved,
            stop_reason=stop,
            rollback_checkpoint=self.best_checkpoint if stop else None,
        )


def write_residual_manifest(
    target: str | os.PathLike[str], *, guard: ResidualTrainingGuard, stop_reason: str
) -> None:
    if not isinstance(guard, ResidualTrainingGuard):
        raise TypeError("guard must be a ResidualTrainingGuard")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accepted": guard.accepted,
        "stop_reason": str(stop_reason),
        "best_iteration": guard.best_iteration,
        "best_checkpoint": guard.best_checkpoint,
        "best_metrics": None if guard.best_metrics is None else asdict(guard.best_metrics),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "ResidualEvalMetrics", "ResidualGuardDecision", "ResidualTrainingGuard",
    "metrics_better", "write_residual_manifest",
]
