"""Pure diagnostic gate for the exact ten-update residual PPO pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class PilotIterationRecord:
    update: int
    learning_rate: float
    value_loss: float
    kl_mean: float
    kl_max: float
    kl_aborted: bool
    completed_mini_batches: int
    grad_norm: float
    active_action_std_min: float
    active_action_std_max: float
    completed_rewards: tuple[float, ...]
    environment_metrics: tuple[tuple[str, float], ...]

    @classmethod
    def from_summary(cls, summary) -> "PilotIterationRecord":
        return cls(
            update=int(summary.iteration) + 1,
            learning_rate=float(summary.learning_rate),
            value_loss=float(summary.value_loss),
            kl_mean=float(summary.kl_mean),
            kl_max=float(summary.kl_max),
            kl_aborted=bool(summary.kl_aborted),
            completed_mini_batches=int(summary.completed_mini_batches),
            grad_norm=float(summary.grad_norm),
            active_action_std_min=float(summary.active_action_std_min),
            active_action_std_max=float(summary.active_action_std_max),
            completed_rewards=tuple(float(value) for value in summary.completed_rewards),
            environment_metrics=tuple(
                sorted(
                    (str(name), float(value))
                    for name, value in summary.environment_metrics
                )
            ),
        )


@dataclass(frozen=True)
class PilotDecision:
    accepted: bool
    reasons: tuple[str, ...]
    kl_abort_count: int
    median_completed_mini_batches: float
    median_value_loss: float


def evaluate_pilot(
    records: tuple[PilotIterationRecord, ...],
) -> PilotDecision:
    if len(records) != 10:
        raise ValueError("pilot requires exactly 10 iteration records")
    if tuple(record.update for record in records) != tuple(range(1, 11)):
        raise ValueError("pilot records must cover updates 1 through 10")

    reasons: list[str] = []
    optimizer_values = tuple(
        value
        for record in records
        for value in (
            record.learning_rate,
            record.value_loss,
            record.kl_mean,
            record.kl_max,
            record.grad_norm,
            record.active_action_std_min,
            record.active_action_std_max,
        )
    )
    optimizer_finite = all(math.isfinite(value) for value in optimizer_values)
    if not optimizer_finite:
        reasons.append("nonfinite_optimizer_diagnostic")
    if not all(
        math.isfinite(value)
        for record in records
        for value in record.completed_rewards
    ):
        reasons.append("nonfinite_reward_diagnostic")

    required_metrics = (
        "hard_failure_count",
        "mpc_feasible_rate",
        "qp_feasible_rate",
        "four_contact_rate",
        *(f"saturation_fraction_{index}" for index in range(8)),
    )
    metrics_per_record = tuple(dict(record.environment_metrics) for record in records)
    missing = sorted(
        {
            name
            for metrics in metrics_per_record
            for name in required_metrics
            if name not in metrics
        }
    )
    reasons.extend(f"missing_environment_metric:{name}" for name in missing)
    nonfinite_metrics = sorted(
        {
            name
            for metrics in metrics_per_record
            for name, value in metrics.items()
            if not math.isfinite(value)
        }
    )
    reasons.extend(
        f"nonfinite_environment_diagnostic:{name}"
        for name in nonfinite_metrics
    )

    if not missing and not nonfinite_metrics:
        if any(metrics["hard_failure_count"] != 0.0 for metrics in metrics_per_record):
            reasons.append("hard_failure_count_nonzero")
        if any(metrics["mpc_feasible_rate"] < 0.99 for metrics in metrics_per_record):
            reasons.append("mpc_feasible_rate_below_0.99")
        if any(metrics["qp_feasible_rate"] != 1.0 for metrics in metrics_per_record):
            reasons.append("qp_feasible_rate_not_exactly_1")
        if any(metrics["four_contact_rate"] != 1.0 for metrics in metrics_per_record):
            reasons.append("four_contact_rate_not_exactly_1")
        if any(
            metrics[f"saturation_fraction_{index}"] >= 0.01
            for metrics in metrics_per_record
            for index in range(8)
        ):
            reasons.append("saturation_fraction_not_below_0.01")

    kl_abort_count = sum(record.kl_aborted for record in records)
    if kl_abort_count > 3:
        reasons.append("kl_abort_count_above_3")
    median_batches = float(
        statistics.median(record.completed_mini_batches for record in records)
    )
    if median_batches < 6.0:
        reasons.append("median_completed_mini_batches_below_6")
    finite_value_losses = [
        record.value_loss for record in records if math.isfinite(record.value_loss)
    ]
    median_value_loss = (
        float(statistics.median(finite_value_losses))
        if finite_value_losses
        else 0.0
    )
    if optimizer_finite and median_value_loss >= 100.0:
        reasons.append("median_value_loss_not_below_100")
    if optimizer_finite and any(
        record.active_action_std_min < 0.005
        or record.active_action_std_max > 0.02
        for record in records
    ):
        reasons.append("action_std_outside_0.005_0.02")

    return PilotDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        kl_abort_count=kl_abort_count,
        median_completed_mini_batches=median_batches,
        median_value_loss=median_value_loss,
    )


__all__ = ["PilotDecision", "PilotIterationRecord", "evaluate_pilot"]
