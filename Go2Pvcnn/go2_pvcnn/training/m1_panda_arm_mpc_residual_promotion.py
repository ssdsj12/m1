"""Noise-calibrated fixed-condition promotion for residual PPO checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

from .m1_panda_arm_mpc_residual_guard import ResidualEvalMetrics


ENGINEERING_FLOORS = {
    "roll_pitch_rms": 1.0e-4,
    "base_height_rms": 2.0e-5,
    "ee_position_error": 5.0e-5,
    "ee_orientation_error": 5.0e-5,
    "wrench_error": 0.1,
    "slip": 2.0e-5,
    "intervention_ratio": 1.0 / 4000.0,
}
RANK_METRICS = (
    "roll_pitch_rms",
    "base_height_rms",
    "ee_position_error",
    "ee_orientation_error",
    "intervention_ratio",
)
REQUIRED_SEEDS = (42, 43, 44)


class MetricComparison(Enum):
    BETTER = -1
    EQUIVALENT = 0
    WORSE = 1


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    aggregate_baseline: ResidualEvalMetrics
    aggregate_candidate: ResidualEvalMetrics
    decisive_metric: str | None


@dataclass(frozen=True)
class PromotedCandidate:
    checkpoint: str
    completed_updates: int
    sha256: str
    decision: PromotionDecision


def _validate_tolerances(tolerances: Mapping[str, float]) -> None:
    if set(tolerances) != set(ENGINEERING_FLOORS):
        raise ValueError("tolerances must contain exactly the promotion metrics")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in tolerances.values()
    ):
        raise ValueError("tolerances must be finite and positive")


def calibrate_tolerances(
    zero_pairs: Sequence[tuple[ResidualEvalMetrics, ResidualEvalMetrics]],
) -> dict[str, float]:
    if len(zero_pairs) != 9:
        raise ValueError("noise calibration requires exactly nine zero pairs")
    for pair in zero_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, ResidualEvalMetrics) for value in pair)
        ):
            raise TypeError("zero pairs must contain ResidualEvalMetrics pairs")
    return {
        name: max(
            floor,
            2.0
            * max(
                abs(getattr(left, name) - getattr(right, name))
                for left, right in zero_pairs
            ),
        )
        for name, floor in ENGINEERING_FLOORS.items()
    }


def compare_with_tolerances(
    candidate: ResidualEvalMetrics,
    baseline: ResidualEvalMetrics,
    tolerances: Mapping[str, float],
) -> MetricComparison:
    if not isinstance(candidate, ResidualEvalMetrics) or not isinstance(
        baseline, ResidualEvalMetrics
    ):
        raise TypeError("candidate and baseline must be ResidualEvalMetrics")
    _validate_tolerances(tolerances)
    if candidate.hard_failure_count < baseline.hard_failure_count:
        return MetricComparison.BETTER
    if candidate.hard_failure_count > baseline.hard_failure_count:
        return MetricComparison.WORSE
    for name in RANK_METRICS:
        delta = getattr(candidate, name) - getattr(baseline, name)
        tolerance = float(tolerances[name])
        if delta < -tolerance:
            return MetricComparison.BETTER
        if delta > tolerance:
            return MetricComparison.WORSE
    return MetricComparison.EQUIVALENT


def _aggregate(values: Sequence[ResidualEvalMetrics]) -> ResidualEvalMetrics:
    count = len(values)
    if count == 0:
        raise ValueError("cannot aggregate empty metrics")

    def mean(name: str) -> float:
        return sum(float(getattr(value, name)) for value in values) / count

    return ResidualEvalMetrics(
        hard_failure_count=sum(value.hard_failure_count for value in values),
        mpc_feasible_rate=mean("mpc_feasible_rate"),
        qp_feasible_rate=mean("qp_feasible_rate"),
        four_contact_rate=mean("four_contact_rate"),
        roll_pitch_rms=mean("roll_pitch_rms"),
        base_height_rms=mean("base_height_rms"),
        ee_position_error=mean("ee_position_error"),
        ee_orientation_error=mean("ee_orientation_error"),
        wrench_error=mean("wrench_error"),
        slip=mean("slip"),
        intervention_ratio=mean("intervention_ratio"),
        saturation_fraction=tuple(
            sum(value.saturation_fraction[index] for value in values) / count
            for index in range(8)
        ),
    )


def _rejected(
    reason: str,
    baselines: Sequence[ResidualEvalMetrics],
    candidates: Sequence[ResidualEvalMetrics],
) -> PromotionDecision:
    return PromotionDecision(
        accepted=False,
        reason=reason,
        aggregate_baseline=_aggregate(baselines),
        aggregate_candidate=_aggregate(candidates),
        decisive_metric=None,
    )


def evaluate_candidate(
    seed_results: Mapping[int, tuple[ResidualEvalMetrics, ResidualEvalMetrics]],
    tolerances: Mapping[str, float],
) -> PromotionDecision:
    _validate_tolerances(tolerances)
    if set(seed_results) != set(REQUIRED_SEEDS):
        raise ValueError("candidate evaluation requires seeds 42, 43, and 44")
    baselines: list[ResidualEvalMetrics] = []
    candidates: list[ResidualEvalMetrics] = []
    for seed in REQUIRED_SEEDS:
        pair = seed_results[seed]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("seed results must contain baseline/candidate pairs")
        baseline, candidate = pair
        if not isinstance(baseline, ResidualEvalMetrics) or not isinstance(
            candidate, ResidualEvalMetrics
        ):
            raise TypeError("seed results must contain ResidualEvalMetrics")
        baselines.append(baseline)
        candidates.append(candidate)
        if not candidate.eligible or candidate.hard_failure_count > baseline.hard_failure_count:
            return _rejected(
                f"seed_{seed}_hard_gate", baselines, candidates
            )
        if compare_with_tolerances(candidate, baseline, tolerances) is MetricComparison.WORSE:
            return _rejected(
                f"seed_{seed}_rank_regression", baselines, candidates
            )
        if candidate.wrench_error - baseline.wrench_error > tolerances["wrench_error"]:
            return _rejected(
                f"seed_{seed}_wrench_regression", baselines, candidates
            )
        if candidate.slip - baseline.slip > tolerances["slip"]:
            return _rejected(f"seed_{seed}_slip_regression", baselines, candidates)

    aggregate_baseline = _aggregate(baselines)
    aggregate_candidate = _aggregate(candidates)
    comparison = compare_with_tolerances(
        aggregate_candidate, aggregate_baseline, tolerances
    )
    decisive_metric = None
    if comparison is not MetricComparison.EQUIVALENT:
        for name in RANK_METRICS:
            if abs(getattr(aggregate_candidate, name) - getattr(aggregate_baseline, name)) > tolerances[name]:
                decisive_metric = name
                break
    if comparison is not MetricComparison.BETTER:
        reason = (
            "aggregate_equivalent"
            if comparison is MetricComparison.EQUIVALENT
            else "aggregate_regression"
        )
        return PromotionDecision(
            False, reason, aggregate_baseline, aggregate_candidate, decisive_metric
        )
    return PromotionDecision(
        True,
        "accepted",
        aggregate_baseline,
        aggregate_candidate,
        decisive_metric,
    )


def select_promoted_candidate(
    candidates: Sequence[PromotedCandidate],
    tolerances: Mapping[str, float],
) -> PromotedCandidate | None:
    _validate_tolerances(tolerances)
    accepted = [value for value in candidates if value.decision.accepted]
    if not accepted:
        return None
    selected = accepted[0]
    for candidate in accepted[1:]:
        comparison = compare_with_tolerances(
            candidate.decision.aggregate_candidate,
            selected.decision.aggregate_candidate,
            tolerances,
        )
        if comparison is MetricComparison.BETTER or (
            comparison is MetricComparison.EQUIVALENT
            and candidate.completed_updates < selected.completed_updates
        ):
            selected = candidate
    return selected


__all__ = [
    "ENGINEERING_FLOORS",
    "MetricComparison",
    "PromotedCandidate",
    "PromotionDecision",
    "calibrate_tolerances",
    "compare_with_tolerances",
    "evaluate_candidate",
    "select_promoted_candidate",
]
