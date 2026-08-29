import pytest

from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import ResidualEvalMetrics
from go2_pvcnn.training.m1_panda_arm_mpc_residual_promotion import (
    ENGINEERING_FLOORS,
    MetricComparison,
    PromotedCandidate,
    calibrate_tolerances,
    compare_with_tolerances,
    evaluate_candidate,
    select_promoted_candidate,
)


def _metrics(**overrides):
    values = dict(
        hard_failure_count=0,
        mpc_feasible_rate=1.0,
        qp_feasible_rate=1.0,
        four_contact_rate=1.0,
        roll_pitch_rms=0.001,
        base_height_rms=0.0002,
        ee_position_error=0.01,
        ee_orientation_error=0.04,
        wrench_error=80.0,
        slip=0.002,
        intervention_ratio=0.0,
        saturation_fraction=(0.0,) * 8,
    )
    values.update(overrides)
    return ResidualEvalMetrics(**values)


def _tolerances():
    return dict(ENGINEERING_FLOORS)


def _three_seed_results(candidate=None):
    candidate = candidate or _metrics(ee_position_error=0.009)
    return {seed: (_metrics(), candidate) for seed in (42, 43, 44)}


def test_calibration_uses_twice_max_delta_or_floor():
    pairs = [
        (_metrics(roll_pitch_rms=0.001), _metrics(roll_pitch_rms=0.00106))
    ] * 9

    tolerances = calibrate_tolerances(pairs)

    assert tolerances["roll_pitch_rms"] == pytest.approx(0.00012)
    assert tolerances["ee_position_error"] == pytest.approx(5.0e-5)


def test_calibration_requires_exactly_nine_pairs():
    with pytest.raises(ValueError, match="exactly nine"):
        calibrate_tolerances([(_metrics(), _metrics())])


def test_zero_equivalence_is_not_improvement():
    baseline = _metrics()

    assert (
        compare_with_tolerances(baseline, baseline, _tolerances())
        is MetricComparison.EQUIVALENT
    )


def test_ee_may_improve_when_stability_is_equivalent():
    baseline = _metrics(roll_pitch_rms=0.001, ee_position_error=0.01)
    candidate = _metrics(roll_pitch_rms=0.00105, ee_position_error=0.009)

    assert (
        compare_with_tolerances(candidate, baseline, _tolerances())
        is MetricComparison.BETTER
    )


def test_stability_regression_beats_ee_improvement():
    baseline = _metrics(roll_pitch_rms=0.001, ee_position_error=0.01)
    candidate = _metrics(roll_pitch_rms=0.0012, ee_position_error=0.001)

    assert (
        compare_with_tolerances(candidate, baseline, _tolerances())
        is MetricComparison.WORSE
    )


def test_all_three_seeds_must_pass_hard_gates():
    results = _three_seed_results()
    baseline, _ = results[44]
    results[44] = (baseline, _metrics(hard_failure_count=1))

    decision = evaluate_candidate(results, _tolerances())

    assert not decision.accepted
    assert decision.reason == "seed_44_hard_gate"


def test_raw_wrench_regression_beyond_tolerance_rejects_candidate():
    results = _three_seed_results(
        _metrics(ee_position_error=0.009, wrench_error=80.2)
    )

    decision = evaluate_candidate(results, _tolerances())

    assert not decision.accepted
    assert decision.reason == "seed_42_wrench_regression"


def test_candidate_requires_real_aggregate_improvement():
    decision = evaluate_candidate(
        _three_seed_results(candidate=_metrics()), _tolerances()
    )

    assert not decision.accepted
    assert decision.reason == "aggregate_equivalent"


def test_tournament_uses_lower_update_count_for_equivalent_candidates():
    decision = evaluate_candidate(_three_seed_results(), _tolerances())
    later = PromotedCandidate("u050.pt", 50, "b" * 64, decision)
    earlier = PromotedCandidate("u025.pt", 25, "a" * 64, decision)

    selected = select_promoted_candidate([later, earlier], _tolerances())

    assert selected is earlier
