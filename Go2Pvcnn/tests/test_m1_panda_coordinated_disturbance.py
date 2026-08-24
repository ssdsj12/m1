from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_coordinated_disturbance import (
    CONTINUOUS_MODE,
    INTERMITTENT_MODE,
    PULSE_MODE,
    CoordinatedDisturbanceCfg,
    CoordinatedDisturbanceScheduler,
)


def fixed_mode_scheduler(mode: int, duration_steps: int, seed: int):
    probabilities = {
        CONTINUOUS_MODE: (1.0, 0.0, 0.0),
        PULSE_MODE: (0.0, 1.0, 0.0),
        INTERMITTENT_MODE: (0.0, 0.0, 1.0),
    }[mode]
    cfg = replace(
        CoordinatedDisturbanceCfg(),
        hold_time_min_s=duration_steps * 0.005,
        hold_time_max_s=duration_steps * 0.005,
        mode_probabilities=probabilities,
    )
    return CoordinatedDisturbanceScheduler(cfg, 1, "cpu", 0.005, seed=seed)


def test_defaults_freeze_the_approved_curriculum() -> None:
    cfg = CoordinatedDisturbanceCfg()
    assert cfg.force_limit_n == 20.0
    assert cfg.torque_limit_nm == 5.0
    assert (cfg.hold_time_min_s, cfg.hold_time_max_s) == (0.25, 1.0)
    assert cfg.curriculum_start_scale == 0.10
    assert cfg.curriculum_steps == 50_000
    assert cfg.mode_probabilities == (0.50, 0.30, 0.20)
    assert cfg.pulse_on_fraction == 0.20


def test_same_seed_reproduces_and_envs_are_independent() -> None:
    left = CoordinatedDisturbanceScheduler(
        CoordinatedDisturbanceCfg(), 8, "cpu", 0.005, seed=7
    )
    right = CoordinatedDisturbanceScheduler(
        CoordinatedDisturbanceCfg(), 8, "cpu", 0.005, seed=7
    )
    wrench = left.advance()
    assert torch.equal(wrench, right.advance())
    assert torch.unique(wrench, dim=0).shape[0] > 1
    assert torch.all(wrench[:, :3].abs() <= 2.0 + 1.0e-6)
    assert torch.all(wrench[:, 3:].abs() <= 0.5 + 1.0e-6)


@pytest.mark.parametrize(
    ("mode", "expected_nonzero_steps"),
    [(CONTINUOUS_MODE, 200), (PULSE_MODE, 40), (INTERMITTENT_MODE, 40)],
)
def test_mode_envelopes_have_exact_duty(mode: int, expected_nonzero_steps: int) -> None:
    scheduler = fixed_mode_scheduler(mode=mode, duration_steps=200, seed=9)
    values = torch.stack([scheduler.advance()[0] for _ in range(200)])
    assert int((values.abs().sum(dim=1) > 0).sum()) == expected_nonzero_steps


def test_curriculum_reaches_full_scale_and_selective_reset_keeps_progress() -> None:
    scheduler = CoordinatedDisturbanceScheduler(
        CoordinatedDisturbanceCfg(curriculum_steps=2),
        4,
        "cpu",
        0.005,
        seed=3,
    )
    scheduler.advance()
    before = scheduler.current_wrench_b.clone()
    scheduler.reset([1, 3])
    scheduler.advance()
    assert scheduler.curriculum_scale == pytest.approx(1.0)
    assert torch.equal(scheduler.current_wrench_b[[0, 2]], before[[0, 2]])
    assert not torch.equal(
        scheduler.current_wrench_b[[1, 3]], torch.zeros(2, 6)
    )


def test_invalid_configuration_and_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        CoordinatedDisturbanceCfg(mode_probabilities=(0.4, 0.4, 0.4))
    scheduler = CoordinatedDisturbanceScheduler(
        CoordinatedDisturbanceCfg(), 2, "cpu", 0.005, seed=1
    )
    with pytest.raises(IndexError, match="out-of-range"):
        scheduler.reset([2])
