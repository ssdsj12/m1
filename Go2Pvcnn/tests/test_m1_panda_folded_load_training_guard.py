from __future__ import annotations

import json
from pathlib import Path

import pytest

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import stage_spec
from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import (
    AtomicStageArtifacts,
    EpisodeRecord,
    FoldedLoadTrainingGuard,
    sha256_file,
)


def _episode(
    command=(0.05, 0.0, 0.0),
    *,
    timeout=True,
    contact=False,
    orientation=False,
    vx_error=0.01,
    wz_error=0.02,
    stationary_vx=0.0,
    stationary_wz=0.0,
):
    steps = 10
    return EpisodeRecord(
        command=command,
        steps=steps,
        time_out=timeout,
        base_contact=contact,
        bad_orientation=orientation,
        vx_error_sq_sum=steps * vx_error**2,
        wz_error_sq_sum=steps * wz_error**2,
        stationary_abs_vx_sum=steps * stationary_vx,
        stationary_abs_wz_sum=steps * stationary_wz,
    )


def _eligible_window(count_per_direction=40, stationary=40):
    records = []
    records += [_episode((0.05, 0.0, 0.0)) for _ in range(count_per_direction)]
    records += [_episode((-0.05, 0.0, 0.0)) for _ in range(count_per_direction)]
    records += [_episode((0.0, 0.0, 0.15)) for _ in range(count_per_direction)]
    records += [_episode((0.0, 0.0, -0.15)) for _ in range(count_per_direction)]
    records += [_episode((0.0, 0.0, 0.0)) for _ in range(stationary)]
    return records


def test_catastrophe_stops_before_any_eligible_best():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    failures = [_episode(contact=True) for _ in range(51)] + [_episode() for _ in range(49)]
    assert guard.update(1, failures).stop is False
    decision = guard.update(2, failures)
    assert decision.stop is True
    assert decision.reason == "hard_failure_rate_gt_0.50_for_2_updates"
    assert guard.eligible_best is None


def test_medium_catastrophe_counter_recovers_and_requires_five_updates():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    failures = [_episode(contact=True) for _ in range(21)] + [_episode() for _ in range(79)]
    for iteration in range(1, 5):
        assert not guard.update(iteration, failures).stop
    assert guard.update(5, failures).reason == "hard_failure_rate_gt_0.20_for_5_updates"

    recovered = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    for iteration in range(1, 5):
        recovered.update(iteration, failures)
    recovered.update(5, [_episode() for _ in range(100)])
    assert recovered.medium_failure_updates == 0


def test_nonfinite_mask_leak_and_fold_failure_stop_immediately():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    assert guard.update(1, [], finite=False).reason == "nonfinite"
    assert guard.update(2, [], inactive_action_max=1e-12).reason == "inactive_action_leak"
    assert guard.update(3, [], fold_hard_failure=True).reason == "fold_hard_failure"


def test_command_level_eligibility_checks_shared_and_directional_gates():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    decision = guard.update(1, _eligible_window())
    assert decision.eligible and decision.save_best
    assert decision.snapshot is not None
    assert decision.snapshot.completed_episodes == 200
    assert decision.snapshot.vx_rmse == pytest.approx(0.01)
    assert decision.snapshot.wz_rmse == pytest.approx(0.02)
    assert decision.snapshot.bucket_counts == (
        ("forward", 40), ("left", 40), ("reverse", 40), ("right", 40)
    )

    bad_reverse = _eligible_window()
    for index in range(40, 80):
        bad_reverse[index] = _episode((-0.05, 0.0, 0.0), vx_error=0.05)
    assert not FoldedLoadTrainingGuard(stage_spec("L0-C0")).update(1, bad_reverse).eligible


def test_eligible_plateau_never_stops_normal_learning():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    first = guard.update(1, _eligible_window())
    assert first.eligible and first.save_best and not first.stop

    decision = first
    for iteration in range(2, 3001):
        decision = guard.update(iteration, [])
        assert decision.reason not in {
            "eligible_patience_50_updates",
            "max_iterations_600",
        }
        assert decision.stop is False
    assert decision.reason is None


def test_stationary_and_minimum_bucket_gates_are_enforced():
    fast_stationary = _eligible_window()
    for index in range(160, 200):
        fast_stationary[index] = _episode((0.0, 0.0, 0.0), stationary_vx=0.031)
    assert not FoldedLoadTrainingGuard(stage_spec("L0-C0")).update(1, fast_stationary).eligible

    sparse = _eligible_window(count_per_direction=24, stationary=104)
    assert len(sparse) == 200
    assert not FoldedLoadTrainingGuard(stage_spec("L0-C0")).update(1, sparse).eligible


def test_dr_level_requires_latest_400_completed_episodes():
    guard = FoldedLoadTrainingGuard(stage_spec("L2-D1"))
    assert guard.update(1, _eligible_window()).snapshot is None
    decision = guard.update(2, _eligible_window())
    assert decision.eligible
    assert decision.snapshot.completed_episodes == 400


def test_atomic_artifacts_accept_only_three_passing_seed_reports(tmp_path: Path):
    artifacts = AtomicStageArtifacts(tmp_path)
    best = tmp_path / "model_best.pt"
    best.write_bytes(b"accepted-policy")
    reports = []
    for seed in (42, 43, 44):
        report = artifacts.write_evaluation(seed, {"seed": seed, "passed": True})
        reports.append(report)
    decision = artifacts.finalize_evaluations(best, reports)
    assert decision["accepted"] is True
    assert sha256_file(best) == sha256_file(tmp_path / "model_final.pt")
    assert json.loads((tmp_path / "evaluation_aggregate.json").read_text())["accepted"] is True


def test_failed_or_wrong_seed_evaluation_cannot_promote(tmp_path: Path):
    artifacts = AtomicStageArtifacts(tmp_path)
    best = tmp_path / "model_best.pt"; best.write_bytes(b"candidate")
    reports = [
        artifacts.write_evaluation(seed, {"seed": seed, "passed": seed != 43})
        for seed in (42, 43, 44)
    ]
    decision = artifacts.finalize_evaluations(best, reports)
    assert decision["accepted"] is False
    assert not (tmp_path / "model_final.pt").exists()


def test_diagnostic_evaluations_are_recorded_but_never_promoted(tmp_path: Path):
    artifacts = AtomicStageArtifacts(tmp_path)
    best = tmp_path / "model_best.pt"
    best.write_bytes(b"diagnostic-policy")
    checkpoint_sha = sha256_file(best)
    reports = [
        artifacts.write_evaluation(
            seed,
            {
                "seed": seed,
                "passed": True,
                "checkpoint_sha256": checkpoint_sha,
            },
        )
        for seed in (42, 43, 44)
    ]

    decision = artifacts.finalize_diagnostics(best, reports)

    assert decision["diagnostic_only"] is True
    assert decision["reports_passed"] is True
    assert decision["accepted"] is False
    assert decision["final_checkpoint"] is None
    assert not (tmp_path / "model_final.pt").exists()
    aggregate = json.loads((tmp_path / "evaluation_aggregate.json").read_text())
    assert aggregate == decision


def test_diagnostic_evaluation_rejects_report_from_another_checkpoint(tmp_path: Path):
    artifacts = AtomicStageArtifacts(tmp_path)
    best = tmp_path / "model_best.pt"
    best.write_bytes(b"diagnostic-policy")
    reports = [
        artifacts.write_evaluation(
            seed,
            {"seed": seed, "passed": False, "checkpoint_sha256": "wrong"},
        )
        for seed in (42, 43, 44)
    ]

    with pytest.raises(ValueError, match="checkpoint SHA"):
        artifacts.finalize_diagnostics(best, reports)


def test_diagnostic_evaluation_rejects_invalid_fixed_seed_set(tmp_path: Path):
    artifacts = AtomicStageArtifacts(tmp_path)
    best = tmp_path / "model_best.pt"
    best.write_bytes(b"diagnostic-policy")
    checkpoint_sha = sha256_file(best)
    reports = [
        artifacts.write_evaluation(
            seed,
            {"seed": seed, "passed": False, "checkpoint_sha256": checkpoint_sha},
        )
        for seed in (42, 43, 44)
    ]
    reports[-1].write_text(
        json.dumps({"seed": None, "passed": False, "checkpoint_sha256": checkpoint_sha}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seeds 42, 43, and 44"):
        artifacts.finalize_diagnostics(best, reports)
