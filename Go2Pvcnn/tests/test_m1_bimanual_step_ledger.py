from __future__ import annotations

import pytest

from scripts.m1_dual_panda_o6_40k_runner import (
    RunLock,
    StepLedger,
    _artifact_stem,
    _parser,
    executed_physics_steps_from_artifacts,
    select_idle_gpus,
    wait_for_idle_gpus,
)
from scripts.m1_dual_panda_o6_bimanual_probe import _write_progress


PHASE_BUDGETS = {
    "contact_preload": 8_000,
    "lift_hold": 12_000,
    "lower_release": 12_000,
    "robustness": 8_000,
}


def test_four_workers_allocate_exact_non_divisible_phase_budget() -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)

    allocations = ledger.allocate_phase(
        "lift_hold", worker_lanes={"gpu4": 16, "gpu5": 16, "gpu6": 16, "gpu7": 16}
    )

    assert sum(item.physics_steps * item.lane_count for item in allocations) == 12_000
    assert sorted(item.physics_steps for item in allocations) == [187, 187, 188, 188]


def test_partial_commit_returns_unexecuted_reservation() -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)
    reservation = ledger.reserve(
        phase="contact_preload", worker="gpu4", lane_count=8, physics_steps=100
    )

    ledger.commit(reservation.reservation_id, executed_physics_steps=60)

    assert ledger.phase_consumed("contact_preload") == 480
    assert ledger.phase_remaining("contact_preload") == 7_520
    assert ledger.reserved_env_steps == 0


def test_commit_cannot_exceed_reserved_physics_steps() -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)
    reservation = ledger.reserve(
        phase="contact_preload", worker="gpu4", lane_count=4, physics_steps=10
    )

    try:
        ledger.commit(reservation.reservation_id, executed_physics_steps=11)
    except ValueError as error:
        assert "reservation" in str(error)
    else:
        raise AssertionError("over-commit was accepted")


def test_atomic_save_and_load_preserve_consumption(tmp_path) -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)
    reservation = ledger.reserve(
        phase="contact_preload", worker="gpu4", lane_count=4, physics_steps=25
    )
    ledger.commit(reservation.reservation_id, executed_physics_steps=20)
    state = tmp_path / "run-state.json"
    ledger.save(state)

    restored = StepLedger.load(state)

    assert restored.consumed_env_steps == 80
    assert restored.phase_remaining("contact_preload") == 7_920


def test_idle_gpu_selection_uses_current_resources_not_fixed_indices() -> None:
    rows = [
        {"index": 0, "memory_used_mb": 20_000, "utilization_percent": 95},
        {"index": 1, "memory_used_mb": 15_000, "utilization_percent": 40},
        {"index": 2, "memory_used_mb": 10, "utilization_percent": 0},
        {"index": 3, "memory_used_mb": 30, "utilization_percent": 2},
        {"index": 4, "memory_used_mb": 20, "utilization_percent": 1},
        {"index": 5, "memory_used_mb": 40, "utilization_percent": 0},
    ]

    assert select_idle_gpus(rows, count=4) == (2, 4, 3, 5)


def test_allocate_phase_can_reserve_a_small_exact_batch() -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)

    allocations = ledger.allocate_phase(
        "lift_hold",
        worker_lanes={"gpu0": 1, "gpu2": 1, "gpu5": 1, "gpu7": 1},
        env_steps=400,
    )

    assert sum(item.env_steps for item in allocations) == 400
    assert {item.physics_steps for item in allocations} == {100}


def test_progress_recovers_executed_steps_when_final_report_is_missing(tmp_path) -> None:
    progress = tmp_path / "worker-progress.json"
    _write_progress(progress, executed_physics_steps=73, num_envs=2)

    assert executed_physics_steps_from_artifacts(
        report_path=tmp_path / "missing-report.json",
        progress_path=progress,
        expected_lanes=2,
        reserved_physics_steps=100,
    ) == 73


def test_wait_for_idle_gpus_retries_without_binding_fixed_indices() -> None:
    attempts = iter((RuntimeError("all busy"), (1, 3, 4, 6)))

    def discover(*, count: int):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        assert count == 4
        return result

    assert wait_for_idle_gpus(
        count=4, poll_seconds=0.0, discover_fn=discover
    ) == (1, 3, 4, 6)


def test_parser_accepts_one_explicit_gpu() -> None:
    args = _parser().parse_args(
        ["--phase", "lift_hold", "--physical-gpus", "0", "--num-envs", "1"]
    )

    assert args.physical_gpus == [0]


def test_run_lock_rejects_a_second_budget_runner(tmp_path) -> None:
    lock_path = tmp_path / "run-state.lock"

    with RunLock(lock_path):
        with pytest.raises(RuntimeError, match="already active"):
            with RunLock(lock_path):
                pass


def test_artifact_stem_keeps_each_reservation_distinct() -> None:
    ledger = StepLedger(total_env_steps=40_000, phase_budgets=PHASE_BUDGETS)
    reservation = ledger.reserve(
        phase="lift_hold", worker="gpu0", lane_count=1, physics_steps=400
    )

    stem = _artifact_stem(0, reservation)

    assert stem == f"gpu0-{reservation.reservation_id[:12]}"
