import json

import pytest

from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import (
    ResidualEvalMetrics,
    ResidualTrainingGuard,
    ResidualTrainingSafetyGuard,
    metrics_better,
    write_residual_manifest,
)


def _metrics(**overrides):
    values = dict(
        hard_failure_count=0,
        mpc_feasible_rate=1.0,
        qp_feasible_rate=1.0,
        four_contact_rate=1.0,
        roll_pitch_rms=0.02,
        base_height_rms=0.01,
        ee_position_error=0.01,
        ee_orientation_error=0.04,
        wrench_error=0.1,
        slip=0.0,
        intervention_ratio=0.05,
        saturation_fraction=(0.0,) * 8,
    )
    values.update(overrides)
    return ResidualEvalMetrics(**values)


def test_metrics_use_stability_first_lexicographic_order():
    stable = _metrics(ee_position_error=0.014)
    unsafe_but_accurate = _metrics(hard_failure_count=1, ee_position_error=0.0)
    more_stable = _metrics(roll_pitch_rms=0.01, ee_position_error=0.014)
    assert metrics_better(stable, unsafe_but_accurate)
    assert metrics_better(more_stable, stable)


def test_saturation_equal_to_one_percent_is_not_eligible():
    guard = ResidualTrainingGuard(patience_updates=2)
    decision = guard.observe(1, _metrics(saturation_fraction=(0.01,) + (0.0,) * 7), "bad.pt")
    assert not decision.eligible
    assert guard.best_checkpoint is None


def test_guard_selects_best_and_requests_rollback_after_patience():
    guard = ResidualTrainingGuard(patience_updates=2)
    first = guard.observe(1, _metrics(), "model_1.pt")
    guard.observe(2, _metrics(roll_pitch_rms=0.03), "model_2.pt")
    last = guard.observe(3, _metrics(roll_pitch_rms=0.04), "model_3.pt")
    assert first.save_best
    assert last.stop_reason == "eligible_patience"
    assert last.rollback_checkpoint == "model_1.pt"


def test_manifest_is_atomic_and_false_without_eligible_best(tmp_path):
    target = tmp_path / "manifest.json"
    write_residual_manifest(target, guard=ResidualTrainingGuard(), stop_reason="no_candidate")
    payload = json.loads(target.read_text())
    assert payload["accepted"] is False
    assert payload["best_checkpoint"] is None
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_records_eligible_best(tmp_path):
    guard = ResidualTrainingGuard()
    guard.observe(7, _metrics(), "model_7.pt")
    target = tmp_path / "manifest.json"
    write_residual_manifest(target, guard=guard, stop_reason="complete")
    payload = json.loads(target.read_text())
    assert payload["accepted"] is True
    assert payload["best_iteration"] == 7
    assert payload["best_checkpoint"] == "model_7.pt"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"hard_failure_count": 1}, "hard_failure"),
        ({"mpc_feasible_rate": 0.98}, "mpc_infeasible"),
        ({"qp_feasible_rate": 0.999}, "qp_infeasible"),
        ({"four_contact_rate": 0.999}, "lost_wheel_contact"),
        ({"saturation_fraction": (0.01,) + (0.0,) * 7}, "residual_saturation"),
    ],
)
def test_training_safety_guard_stops_only_on_physical_gates(overrides, reason):
    guard = ResidualTrainingSafetyGuard()

    assert guard.observe(_metrics(**overrides)) == reason


def test_training_safety_guard_does_not_rank_trajectory_error():
    guard = ResidualTrainingSafetyGuard()

    assert guard.observe(_metrics(ee_position_error=0.02)) is None
