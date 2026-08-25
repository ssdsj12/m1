from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "go2_pvcnn"
    / "tasks"
    / "m1_panda_folded_load_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location(
    "m1_panda_folded_load_curriculum_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
curriculum = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = curriculum
SPEC.loader.exec_module(curriculum)


def test_training_iteration_contract_allows_full_3000_updates():
    assert curriculum.MAX_TRAINING_ITERATIONS == 3000
    assert curriculum.validate_max_iterations(1) == 1
    assert curriculum.validate_max_iterations(3000) == 3000
    for invalid in (0, 3001, -1, True, 3.5):
        with pytest.raises((TypeError, ValueError)):
            curriculum.validate_max_iterations(invalid)


def test_stage_order_command_limits_and_parents_are_exact():
    assert curriculum.STAGE_ORDER == (
        "L0-C0",
        "L1-C1",
        "L1-C2",
        "L1-C3",
        "L1-C4",
        "L2-D1",
        "L2-D2",
        "L2-D3",
    )
    expected = {
        "L0-C0": (None, 0.05, 0.15, 200),
        "L1-C1": ("L0-C0", 0.08, 0.25, 200),
        "L1-C2": ("L1-C1", 0.12, 0.35, 200),
        "L1-C3": ("L1-C2", 0.16, 0.48, 200),
        "L1-C4": ("L1-C3", 0.20, 0.60, 200),
        "L2-D1": ("L1-C4", 0.20, 0.60, 400),
        "L2-D2": ("L2-D1", 0.20, 0.60, 400),
        "L2-D3": ("L2-D2", 0.20, 0.60, 400),
    }
    for name, values in expected.items():
        spec = curriculum.stage_spec(name)
        assert (spec.parent, spec.vx_limit, spec.wz_limit, spec.completed_episode_window) == values


def test_reset_ranges_are_exact_and_protected_fields_stay_zero():
    expected = {
        "L0-C0": ((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, (1.0, 1.0)),
        "L2-D1": ((0.01, 0.01, 0.01), 0.005, 0.01, 0.02, (0.95, 1.05)),
        "L2-D2": ((0.015, 0.015, 0.025), 0.01, 0.025, 0.05, (0.90, 1.10)),
        "L2-D3": ((0.03, 0.03, 0.05), 0.02, 0.05, 0.10, (0.80, 1.20)),
    }
    root_xy = {"L0-C0": 0.0, "L2-D1": 0.005, "L2-D2": 0.01, "L2-D3": 0.02}
    for name, (rpy, leg, linear, angular, friction) in expected.items():
        reset = curriculum.stage_spec(name).reset
        assert reset.root_xy == root_xy[name]
        assert reset.root_rpy == rpy
        assert reset.leg_position == leg
        assert reset.root_linear_velocity == linear
        assert reset.root_angular_velocity == angular
        assert reset.friction == friction
        assert reset.root_z == (0.0, 0.0)
        assert reset.wheel_position == (0.0, 0.0)
        assert reset.panda_position == (0.0, 0.0)
        assert reset.panda_velocity == (0.0, 0.0)
        assert reset.restitution == (0.0, 0.0)


def test_stage_lookup_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown folded-load stage"):
        curriculum.stage_spec("C9")


def test_command_sampler_is_seeded_bounded_and_matches_family_probabilities():
    stage = curriculum.stage_spec("L1-C2")
    first = curriculum.sample_episode_commands(10_000, stage, seed=7)
    second = curriculum.sample_episode_commands(10_000, stage, seed=7)
    torch.testing.assert_close(first.twist, second.twist)
    torch.testing.assert_close(first.family, second.family)

    counts = torch.bincount(first.family, minlength=4).float() / 10_000
    torch.testing.assert_close(
        counts,
        torch.tensor([0.20, 0.25, 0.20, 0.35]),
        atol=0.015,
        rtol=0.0,
    )
    assert first.twist.shape == (10_000, 3)
    assert first.twist[:, 1].eq(0.0).all()
    assert first.twist[:, 0].abs().max() <= stage.vx_limit
    assert first.twist[:, 2].abs().max() <= stage.wz_limit

    stationary = first.family == curriculum.CommandFamily.STATIONARY
    straight = first.family == curriculum.CommandFamily.STRAIGHT
    turning = first.family == curriculum.CommandFamily.TURN_IN_PLACE
    combined = first.family == curriculum.CommandFamily.COMBINED
    assert first.twist[stationary].eq(0.0).all()
    assert first.twist[straight, 0].ne(0.0).all()
    assert first.twist[straight, 2].eq(0.0).all()
    assert first.twist[turning, 0].eq(0.0).all()
    assert first.twist[turning, 2].ne(0.0).all()
    assert first.twist[combined, 0].ne(0.0).all()
    assert first.twist[combined, 2].ne(0.0).all()


def test_command_sampler_validates_count_and_preserves_requested_device():
    stage = curriculum.stage_spec("L0-C0")
    with pytest.raises(ValueError, match="num_envs"):
        curriculum.sample_episode_commands(0, stage, seed=1)
    commands = curriculum.sample_episode_commands(8, stage, seed=1, device="cpu")
    assert commands.twist.device.type == "cpu"
    assert commands.family.device.type == "cpu"


def test_balanced_eval_table_is_deterministic_and_covers_directional_buckets():
    stage = curriculum.stage_spec("L2-D3")
    commands = curriculum.balanced_eval_commands(64, stage, device="cpu")
    torch.testing.assert_close(
        commands, curriculum.balanced_eval_commands(64, stage, device="cpu")
    )
    assert commands.shape == (64, 3)
    assert commands[:, 1].eq(0.0).all()
    assert commands[:, 0].abs().max() <= stage.vx_limit
    assert commands[:, 2].abs().max() <= stage.wz_limit
    buckets = curriculum.classify_command_buckets(commands)
    assert buckets["stationary"].sum() >= 8
    for name in ("forward", "reverse", "left", "right"):
        assert buckets[name].sum() >= 8


def test_balanced_eval_table_requires_compatible_size():
    with pytest.raises(ValueError, match="multiple of 16"):
        curriculum.balanced_eval_commands(63, curriculum.stage_spec("L0-C0"))
