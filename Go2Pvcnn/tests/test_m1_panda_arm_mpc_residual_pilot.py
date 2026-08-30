from __future__ import annotations

from dataclasses import replace

import pytest

from go2_pvcnn.training.m1_panda_arm_mpc_residual_pilot import (
    PilotIterationRecord,
    evaluate_pilot,
)


def _record(update: int, **overrides) -> PilotIterationRecord:
    metrics = {
        "hard_failure_count": 0.0,
        "mpc_feasible_rate": 1.0,
        "qp_feasible_rate": 1.0,
        "four_contact_rate": 1.0,
        **{f"saturation_fraction_{index}": 0.0 for index in range(8)},
    }
    values = {
        "update": update,
        "learning_rate": 1.0e-5,
        "value_loss": 10.0,
        "kl_mean": 0.005,
        "kl_max": 0.01,
        "kl_aborted": False,
        "completed_mini_batches": 8,
        "grad_norm": 0.5,
        "active_action_std_min": 0.01,
        "active_action_std_max": 0.01,
        "completed_rewards": (),
        "environment_metrics": tuple(sorted(metrics.items())),
    }
    values.update(overrides)
    return PilotIterationRecord(**values)


def _records(**record_overrides):
    return tuple(_record(update, **record_overrides) for update in range(1, 11))


def _with_metric(records, name, value):
    changed = []
    for record in records:
        metrics = dict(record.environment_metrics)
        metrics[name] = value
        changed.append(
            replace(record, environment_metrics=tuple(sorted(metrics.items())))
        )
    return tuple(changed)


def test_exact_safe_pilot_is_accepted():
    decision = evaluate_pilot(_records())

    assert decision.accepted is True
    assert decision.reasons == ()
    assert decision.kl_abort_count == 0
    assert decision.median_completed_mini_batches == pytest.approx(8.0)
    assert decision.median_value_loss == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("records", "reason"),
    (
        (_with_metric(_records(), "hard_failure_count", 1.0), "hard_failure"),
        (_with_metric(_records(), "mpc_feasible_rate", 0.989), "mpc_feasible"),
        (_with_metric(_records(), "qp_feasible_rate", 0.999), "qp_feasible"),
        (_with_metric(_records(), "four_contact_rate", 0.999), "four_contact"),
        (_with_metric(_records(), "saturation_fraction_0", 0.01), "saturation"),
        (_records(active_action_std_min=0.0049), "action_std"),
        (_records(active_action_std_max=0.0201), "action_std"),
        (_records(value_loss=100.0), "value_loss"),
    ),
)
def test_pilot_rejects_each_frozen_physical_or_optimizer_gate(records, reason):
    decision = evaluate_pilot(records)

    assert decision.accepted is False
    assert any(reason in value for value in decision.reasons)


def test_pilot_rejects_four_kl_aborts_but_accepts_three():
    records = list(_records())
    for index in range(4):
        records[index] = replace(records[index], kl_aborted=True)

    assert evaluate_pilot(tuple(records)).accepted is False
    records[3] = replace(records[3], kl_aborted=False)
    assert evaluate_pilot(tuple(records)).accepted is True


def test_pilot_requires_median_six_of_eight_mini_batches():
    failing = tuple(
        replace(record, completed_mini_batches=5 if index < 6 else 8)
        for index, record in enumerate(_records())
    )
    passing = tuple(
        replace(record, completed_mini_batches=6 if index < 6 else 8)
        for index, record in enumerate(_records())
    )

    assert evaluate_pilot(failing).accepted is False
    assert evaluate_pilot(passing).accepted is True


@pytest.mark.parametrize("count", (9, 11))
def test_pilot_requires_exactly_ten_ordered_updates(count):
    with pytest.raises(ValueError, match="exactly 10"):
        evaluate_pilot(tuple(_record(update) for update in range(1, count + 1)))

    wrong_order = list(_records())
    wrong_order[-1] = replace(wrong_order[-1], update=9)
    with pytest.raises(ValueError, match="updates 1 through 10"):
        evaluate_pilot(tuple(wrong_order))


@pytest.mark.parametrize(
    "records",
    (
        _records(value_loss=float("nan")),
        _records(completed_rewards=(float("inf"),)),
        _with_metric(_records(), "mpc_feasible_rate", float("nan")),
    ),
)
def test_pilot_rejects_nonfinite_optimizer_reward_or_environment(records):
    decision = evaluate_pilot(records)

    assert decision.accepted is False
    assert any("nonfinite" in reason for reason in decision.reasons)
