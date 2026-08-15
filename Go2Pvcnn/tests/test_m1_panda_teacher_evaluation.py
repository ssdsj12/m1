from __future__ import annotations

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_teacher_evaluation import (
    TeacherEvaluationAccumulator,
    aggregate_candidate_summaries,
    validate_full_scale_summary,
)


def _summary(checkpoint, seed, *, timeout, contact, bad, reward):
    total = timeout + contact + bad
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": f"sha-{checkpoint}",
        "base_checkpoint_sha256": "base-sha",
        "frozen_actor_sha256": "frozen-sha",
        "seed": seed,
        "num_envs": 64,
        "steps": 2000,
        "curriculum_scale": 1.0,
        "axis_abs_wrench_seen": [19.5, 19.5, 19.5, 4.8, 4.8, 4.8],
        "termination_counts": {
            "time_out": timeout,
            "base_contact": contact,
            "bad_orientation": bad,
        },
        "termination_rates": {
            "time_out": timeout / total,
            "base_contact": contact / total,
            "bad_orientation": bad / total,
        },
        "reward_sum": reward * 64 * 2000,
        "reward_count": 64 * 2000,
        "mean_reward": reward,
        "finite": True,
    }


def test_accumulator_reports_rates_and_reward_over_all_env_steps():
    acc = TeacherEvaluationAccumulator(num_envs=2)
    acc.update(
        rewards=torch.tensor([1.0, 3.0]),
        termination_counts={
            "time_out": 1,
            "base_contact": 0,
            "bad_orientation": 0,
        },
    )

    summary = acc.finalize(
        checkpoint="model_1.pt",
        checkpoint_sha256="abc",
        base_checkpoint_sha256="base",
        seed=42,
        steps=1,
        curriculum_scale=1.0,
        axis_abs_wrench_seen=[19.5, 19.5, 19.5, 4.8, 4.8, 4.8],
        frozen_actor_sha256="frozen",
    )

    assert summary["mean_reward"] == pytest.approx(2.0)
    assert summary["reward_sum"] == pytest.approx(4.0)
    assert summary["reward_count"] == 2
    assert summary["termination_counts"]["time_out"] == 1
    assert summary["termination_rates"]["time_out"] == pytest.approx(1.0)


def test_full_scale_gate_rejects_an_underexcited_axis():
    summary = {
        "curriculum_scale": 1.0,
        "axis_abs_wrench_seen": [19.5, 19.5, 18.9, 4.8, 4.8, 4.8],
        "finite": True,
        "checkpoint_sha256": "checkpoint",
        "base_checkpoint_sha256": "base",
        "frozen_actor_sha256": "frozen",
    }

    with pytest.raises(ValueError, match="force axis"):
        validate_full_scale_summary(summary)


def test_full_scale_gate_rejects_underexcited_torque_nonfinite_and_scale():
    valid = _summary("model.pt", 42, timeout=8, contact=1, bad=1, reward=2.0)
    validate_full_scale_summary(valid)

    torque = dict(valid, axis_abs_wrench_seen=[19.5] * 3 + [4.74, 4.8, 4.8])
    with pytest.raises(ValueError, match="torque axis"):
        validate_full_scale_summary(torque)

    with pytest.raises(ValueError, match="finite"):
        validate_full_scale_summary(dict(valid, finite=False))
    with pytest.raises(ValueError, match="curriculum_scale"):
        validate_full_scale_summary(dict(valid, curriculum_scale=0.999))


def test_candidate_aggregation_uses_approved_rates_and_rank_order():
    summaries = [
        _summary("model_a.pt", 42, timeout=80, contact=10, bad=10, reward=4.0),
        _summary("model_a.pt", 43, timeout=80, contact=10, bad=10, reward=4.0),
        _summary("model_a.pt", 44, timeout=80, contact=10, bad=10, reward=4.0),
    ]

    aggregate = aggregate_candidate_summaries(summaries)

    assert aggregate["timeout_survival_rate"] == pytest.approx(0.8)
    assert aggregate["base_contact_rate"] == pytest.approx(0.1)
    assert aggregate["bad_orientation_rate"] == pytest.approx(0.1)
    assert aggregate["mean_reward"] == pytest.approx(4.0)
    assert aggregate["accepted"] is True
    assert aggregate["rank_key"] == pytest.approx([0.8, -0.1, -0.1, 4.0])


def test_candidate_aggregation_requires_exact_seeds_and_one_checkpoint():
    missing_seed = [
        _summary("model_a.pt", 42, timeout=8, contact=1, bad=1, reward=1.0),
        _summary("model_a.pt", 43, timeout=8, contact=1, bad=1, reward=1.0),
    ]
    with pytest.raises(ValueError, match="seeds"):
        aggregate_candidate_summaries(missing_seed)

    mixed = missing_seed + [
        _summary("model_b.pt", 44, timeout=8, contact=1, bad=1, reward=1.0)
    ]
    with pytest.raises(ValueError, match="checkpoint"):
        aggregate_candidate_summaries(mixed)
