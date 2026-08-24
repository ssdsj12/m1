from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_coordinated_training_guard import (
    AtomicCheckpointController,
    TrainingGuard,
    sha256_file,
)
from rsl_rl.runners.on_policy_runner import IterationSummary


def summary(
    it: int,
    timeout: float = 1.0,
    contact: float = 0.0,
    orientation: float = 0.0,
    base: float = 2.0,
    ee: float = 1.0,
    reward: float = 100.0,
    count: int = 100,
) -> IterationSummary:
    metrics = {
        "Termination/time_out": (timeout,) * count,
        "Termination/base_contact": (contact,) * count,
        "Termination/bad_orientation": (orientation,) * count,
        "Reward/base_target": (base,) * count,
        "Reward/ee_tracking": (ee,) * count,
    }
    return IterationSummary(
        it,
        it * 16384,
        (reward,) * count,
        tuple(metrics.items()),
        1.0e-4,
        0.01,
        (("wrench_scale", 0.5),),
    )


class FakeRunner:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1)
        self.current_learning_iteration = 0
        self.load_calls: list[tuple[Path, bool, bool]] = []

    def save(self, path) -> None:
        torch.save(
            {
                "model_state_dict": self.module.state_dict(),
                "iter": self.current_learning_iteration,
            },
            path,
        )

    def load(self, path, load_optimizer=True, keep_std=False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.module.load_state_dict(payload["model_state_dict"])
        self.current_learning_iteration = int(payload["iter"])
        self.load_calls.append((Path(path), load_optimizer, keep_std))
        return None


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


def test_guard_waits_for_100_completed_episodes() -> None:
    decision = TrainingGuard().observe(summary(1, count=99))
    assert decision.snapshot is None and not decision.save_best


def test_rank_minimizes_hard_failure_before_task_score_and_keeps_earlier_ties() -> None:
    guard = TrainingGuard()
    first = guard.observe(summary(1, contact=0.01, base=2.0))
    worse_safety = guard.observe(summary(2, contact=0.02, base=100.0))
    equal = guard.observe(summary(3, contact=0.01, base=2.0))
    assert first.save_best
    assert not worse_safety.save_best and not equal.save_best
    assert guard.eligible_best is not None
    assert guard.eligible_best.iteration == 1


def test_nonfinite_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainingGuard().observe(summary(1, reward=float("nan")))


def test_ineligible_diagnostic_best_is_not_accepted() -> None:
    guard = TrainingGuard()
    guard.observe(summary(1, timeout=0.5, contact=0.1))
    assert guard.diagnostic_best is not None
    assert guard.eligible_best is None and not guard.accepted


def test_patience_catastrophe_recovery_and_cap() -> None:
    patience = TrainingGuard()
    patience.observe(summary(1))
    assert [patience.observe(summary(it)).stop_reason for it in range(2, 52)][-1] == (
        "eligible_patience"
    )

    catastrophe = TrainingGuard()
    catastrophe.observe(summary(1))
    for it in range(2, 26):
        assert catastrophe.observe(summary(it, contact=0.21)).stop_reason is None
    catastrophe.observe(summary(26, contact=0.0))
    assert catastrophe.catastrophe_updates == 0
    assert [
        catastrophe.observe(summary(it, contact=0.21)).stop_reason
        for it in range(27, 52)
    ][-1] == "catastrophe"

    assert TrainingGuard(max_iterations=1).observe(summary(1)).stop_reason == (
        "max_iterations"
    )


def test_update_without_completed_episode_cannot_replace_best() -> None:
    guard = TrainingGuard()
    guard.observe(summary(1))
    empty = IterationSummary(2, 32768, (), (), 1.0e-4, 0.01, ())

    decision = guard.observe(empty)

    assert decision.snapshot is not None
    assert not decision.save_best
    assert guard.eligible_best is not None
    assert guard.eligible_best.iteration == 1


def test_atomic_controller_hashes_best_and_rolls_back_final(
    tmp_path: Path, fake_runner: FakeRunner
) -> None:
    controller = AtomicCheckpointController(tmp_path, TrainingGuard(max_iterations=1))
    fake_runner.current_learning_iteration = 1
    assert controller.on_iteration(fake_runner, summary(1)) == "max_iterations"

    with torch.no_grad():
        fake_runner.module.weight.fill_(123.0)
    fields = controller.finalize(fake_runner, "max_iterations")

    assert fields["accepted"] is True
    assert fields["status"] == "accepted"
    assert Path(fields["final_checkpoint"]).name == "model_final.pt"
    assert fields["rollback_source_sha256"] == sha256_file(
        tmp_path / "model_best.pt"
    )
    assert fields["final_checkpoint_sha256"] == sha256_file(
        tmp_path / "model_final.pt"
    )
    assert fake_runner.load_calls == [(tmp_path / "model_best.pt", False, True)]
    assert fake_runner.module.weight.item() != pytest.approx(123.0)

    metadata = json.loads(
        (tmp_path / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["eligible"] is True
    assert metadata["checkpoint_sha256"] == fields["rollback_source_sha256"]
    assert metadata["environment_metrics"] == {"wrench_scale": 0.5}


def test_finalize_marks_diagnostic_fallback_unaccepted(
    tmp_path: Path, fake_runner: FakeRunner
) -> None:
    controller = AtomicCheckpointController(tmp_path, TrainingGuard(max_iterations=1))
    fake_runner.current_learning_iteration = 1
    controller.on_iteration(fake_runner, summary(1, timeout=0.5, contact=0.1))

    fields = controller.finalize(fake_runner, "max_iterations")

    assert fields["accepted"] is False
    assert fields["status"] == "completed_without_eligible_best"
