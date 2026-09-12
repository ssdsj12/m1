"""Runtime-only, fail-closed adapter for the frozen O6 fingertip prior.

This module deliberately depends only on the artifact/model contracts and the
live O6 geometry contract.  It never discovers data, loads source hands, or
performs offline work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time

import torch

from ..contracts import BimanualPhase
from .artifact import LoadedStudent, load_student_artifact
from .contracts import MIXTURE_COMPONENTS, MODEL_INPUT_DIM, PHASE_ORDER, PRIOR_HORIZON, PriorPhase


_SAFE_PHASES = frozenset(
    {
        BimanualPhase.DONE,
        BimanualPhase.HOLD_SAFE,
        BimanualPhase.LOWER_SAFE,
        BimanualPhase.SAFE_RELEASE,
        BimanualPhase.TERMINATED,
    }
)
_PHASE_MAP = {
    BimanualPhase.APPROACH: PriorPhase.APPROACH,
    BimanualPhase.PRELOAD: PriorPhase.PRELOAD,
    BimanualPhase.GRASP: PriorPhase.GRASP,
    BimanualPhase.LIFT: PriorPhase.MANIPULATE,
    BimanualPhase.HOLD: PriorPhase.HOLD,
    BimanualPhase.LOWER: PriorPhase.MANIPULATE,
    BimanualPhase.RELEASE: PriorPhase.RELEASE,
}


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class PriorRuntimeCfg:
    """Bounded runtime behavior; all defaults keep the prior conservative."""

    log_std_min: float = -7.0
    log_std_max: float = 3.0
    precision_min: float = 1.0e-4
    precision_max: float = 1.0e4
    logit_weight: float = 1.0e-2
    inference_timeout_ms: float = 20.0
    safe_phases: frozenset[BimanualPhase] = field(default_factory=lambda: _SAFE_PHASES)

    def __post_init__(self) -> None:
        lower = _positive_finite("precision_min", self.precision_min)
        upper = _positive_finite("precision_max", self.precision_max)
        if lower > upper:
            raise ValueError("precision_min must not exceed precision_max")
        for name in ("log_std_min", "log_std_max", "logit_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.log_std_min) > float(self.log_std_max):
            raise ValueError("log_std_min must not exceed log_std_max")
        _positive_finite("inference_timeout_ms", self.inference_timeout_ms)
        if not isinstance(self.safe_phases, frozenset) or not self.safe_phases:
            raise TypeError("safe_phases must be a non-empty frozenset")
        if not all(isinstance(phase, BimanualPhase) for phase in self.safe_phases):
            raise TypeError("safe_phases must contain BimanualPhase values")


@dataclass(frozen=True)
class O6FingertipPriorInput:
    """Raw measured O6 geometry before strict runtime validation."""

    fingertip_positions_b: torch.Tensor
    contact_jacobian: torch.Tensor
    qd: torch.Tensor
    contact_mask: torch.Tensor
    phase: BimanualPhase


@dataclass(frozen=True)
class FingertipPriorTarget:
    """The first future-node velocity target consumed by the Hand QP."""

    mean_velocity: torch.Tensor
    precision: torch.Tensor
    component: int
    probability: float

    def __post_init__(self) -> None:
        for name in ("mean_velocity", "precision"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.float64 or value.device.type != "cpu" or value.shape != (15,):
                raise ValueError(f"{name} must be a CPU float64 tensor with shape (15,)")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
        if not torch.all(self.precision >= 0.0).item():
            raise ValueError("precision must be non-negative")
        if type(self.component) is not int or not 0 <= self.component < MIXTURE_COMPONENTS:
            raise ValueError("component is out of range")
        if isinstance(self.probability, bool) or not isinstance(self.probability, float):
            raise TypeError("probability must be a float")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be a finite probability")
        object.__setattr__(self, "mean_velocity", self.mean_velocity.clone())
        object.__setattr__(self, "precision", self.precision.clone())


@dataclass(frozen=True)
class FingertipPriorDiagnostics:
    enabled: bool
    reason: str | None
    inference_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if self.reason is not None and (type(self.reason) is not str or not self.reason):
            raise ValueError("reason must be a non-empty string or None")
        if isinstance(self.inference_ms, bool) or not isinstance(self.inference_ms, float):
            raise TypeError("inference_ms must be a float")
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0.0:
            raise ValueError("inference_ms must be finite and non-negative")
        if self.enabled != (self.reason is None):
            raise ValueError("enabled and reason disagree")


@dataclass(frozen=True)
class PriorQueryResult:
    target: FingertipPriorTarget | None
    diagnostics: FingertipPriorDiagnostics

    def __post_init__(self) -> None:
        if self.target is not None and not isinstance(self.target, FingertipPriorTarget):
            raise TypeError("target must be FingertipPriorTarget or None")
        if not isinstance(self.diagnostics, FingertipPriorDiagnostics):
            raise TypeError("diagnostics must be FingertipPriorDiagnostics")
        if self.diagnostics.enabled != (self.target is not None):
            raise ValueError("target and diagnostics disagree")


def _disabled(reason: str, inference_ms: float = 0.0) -> PriorQueryResult:
    return PriorQueryResult(
        target=None,
        diagnostics=FingertipPriorDiagnostics(enabled=False, reason=reason, inference_ms=float(inference_ms)),
    )


def _valid_tensor(value: object, *, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.shape == shape
        and value.dtype == dtype
        and value.device.type == "cpu"
        and bool(torch.isfinite(value).all().item())
    )


def _phase_one_hot(phase: BimanualPhase) -> torch.Tensor | None:
    prior_phase = _PHASE_MAP.get(phase)
    if prior_phase is None:
        return None
    return torch.nn.functional.one_hot(
        torch.tensor(int(prior_phase)), num_classes=len(PHASE_ORDER)
    ).to(dtype=torch.float32)


class FrozenO6FingertipPrior:
    """Read-only adapter that turns a gate-approved student into a QP target."""

    def __init__(self, loaded: LoadedStudent, *, cfg: PriorRuntimeCfg | None = None) -> None:
        if not isinstance(loaded, LoadedStudent):
            raise TypeError("loaded must be a LoadedStudent")
        if loaded.metrics.get("production_approved") is not True:
            raise ValueError("runtime requires a production_approved student artifact")
        if not isinstance(loaded.model, torch.nn.Module):
            raise TypeError("loaded student model must be a torch module")
        self.cfg = PriorRuntimeCfg() if cfg is None else cfg
        if not isinstance(self.cfg, PriorRuntimeCfg):
            raise TypeError("cfg must be PriorRuntimeCfg")
        self._model = loaded.model
        self._model.eval()
        self._lock = threading.RLock()

    @classmethod
    def from_artifact(cls, path: str | Path, *, cfg: PriorRuntimeCfg | None = None) -> "FrozenO6FingertipPrior":
        """Load only the artifact loader's fully verified, real approved export."""

        loaded = load_student_artifact(path)
        # Construction follows loading so rejected artifacts cannot leak a usable adapter.
        return cls(loaded, cfg=cfg)

    def _network_input(self, sample: O6FingertipPriorInput) -> torch.Tensor | None:
        if not isinstance(sample, O6FingertipPriorInput):
            return None
        if not _valid_tensor(sample.fingertip_positions_b, shape=(5, 3), dtype=torch.float64):
            return None
        if not _valid_tensor(sample.contact_jacobian, shape=(15, 6), dtype=torch.float64):
            return None
        if not _valid_tensor(sample.qd, shape=(6,), dtype=torch.float64):
            return None
        if not isinstance(sample.contact_mask, torch.Tensor) or sample.contact_mask.shape != (5,) or sample.contact_mask.dtype != torch.bool or sample.contact_mask.device.type != "cpu":
            return None
        if not isinstance(sample.phase, BimanualPhase):
            return None
        phase = _phase_one_hot(sample.phase)
        if phase is None:
            return None
        velocity = sample.contact_jacobian @ sample.qd
        values = torch.cat(
            (
                sample.fingertip_positions_b.reshape(-1),
                velocity,
                sample.contact_mask.to(dtype=torch.float64),
                phase.to(dtype=torch.float64),
            )
        )
        if values.shape != (MODEL_INPUT_DIM,) or not torch.isfinite(values).all().item():
            return None
        return values.to(dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _distribution_tensors(output: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        try:
            logits, mean, log_std = output.logits, output.mean, output.log_std  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return None
        expected_mean = (1, MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3)
        for value, shape in ((logits, (1, MIXTURE_COMPONENTS)), (mean, expected_mean), (log_std, expected_mean)):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu" or value.shape != shape:
                return None
            if not torch.isfinite(value).all().item():
                return None
        return logits, mean, log_std

    def target(self, sample: O6FingertipPriorInput, baseline_qd: torch.Tensor) -> PriorQueryResult:
        """Return an independent same-cycle target, or an explicit disabled result."""

        if isinstance(sample, O6FingertipPriorInput) and isinstance(sample.phase, BimanualPhase) and sample.phase in self.cfg.safe_phases:
            return _disabled("safe_phase")
        network_input = self._network_input(sample)
        if network_input is None or not _valid_tensor(baseline_qd, shape=(6,), dtype=torch.float64):
            return _disabled("invalid_input")
        try:
            baseline_tip_velocity = sample.contact_jacobian @ baseline_qd
            started = time.perf_counter_ns()
            with self._lock, torch.no_grad():
                self._model.eval()
                output = self._model(network_input)
                tensors = self._distribution_tensors(output)
                if tensors is None:
                    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
                    return _disabled("nonfinite_prior" if any(
                        isinstance(getattr(output, name, None), torch.Tensor)
                        and not torch.isfinite(getattr(output, name)).all().item()
                        for name in ("logits", "mean", "log_std")
                    ) else "invalid_prior", elapsed_ms)
                logits, means, log_std = tensors
                bounded_log_std = log_std[:, :, 0].clamp(self.cfg.log_std_min, self.cfg.log_std_max)
                precision = torch.exp(-2.0 * bounded_log_std).clamp(
                    self.cfg.precision_min, self.cfg.precision_max
                )
                precision[:, :, sample.contact_mask, :] = 0.0
                first_means = means[:, :, 0].reshape(MIXTURE_COMPONENTS, 15).to(dtype=torch.float64)
                component_precision = precision.reshape(MIXTURE_COMPONENTS, 15).to(dtype=torch.float64)
                probabilities = logits[0].log_softmax(dim=-1).exp()
                if not torch.isfinite(probabilities).all().item() or not torch.all(probabilities >= 0.0).item() or not torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1.0e-6, rtol=0.0).item():
                    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
                    return _disabled("invalid_probabilities", elapsed_ms)
                distance = ((baseline_tip_velocity.reshape(1, 15) - first_means).square() * component_precision).sum(dim=1)
                score = distance - float(self.cfg.logit_weight) * logits[0].log_softmax(dim=-1).to(dtype=torch.float64)
                if not torch.isfinite(score).all().item():
                    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
                    return _disabled("invalid_precision", elapsed_ms)
                component = int(torch.argmin(score).item())
                elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
                if elapsed_ms > self.cfg.inference_timeout_ms:
                    return _disabled("timeout", elapsed_ms)
                return PriorQueryResult(
                    target=FingertipPriorTarget(
                        mean_velocity=first_means[component],
                        precision=component_precision[component],
                        component=component,
                        probability=float(probabilities[component].item()),
                    ),
                    diagnostics=FingertipPriorDiagnostics(enabled=True, reason=None, inference_ms=elapsed_ms),
                )
        except (RuntimeError, TypeError, ValueError, ArithmeticError) as error:
            del error
            return _disabled("prior_exception")


__all__ = [
    "FingertipPriorDiagnostics",
    "FingertipPriorTarget",
    "FrozenO6FingertipPrior",
    "O6FingertipPriorInput",
    "PriorQueryResult",
    "PriorRuntimeCfg",
]
