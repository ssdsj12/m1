from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsl_rl.modules.normalizer import EmpiricalNormalization


def test_empirical_normalizer_count_round_trips_in_state_dict():
    normalizer = EmpiricalNormalization([2])
    normalizer.train()
    normalizer(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    state = normalizer.state_dict()
    assert state["_count"].dtype == torch.int64
    assert state["_count"].shape == torch.Size([])
    assert state["_count"].item() == 3

    restored = EmpiricalNormalization([2])
    restored.load_state_dict(state)
    assert restored.count == 3
    torch.testing.assert_close(restored.mean, normalizer.mean)
    torch.testing.assert_close(restored.std, normalizer.std)


def test_eval_forward_does_not_mutate_restored_count():
    normalizer = EmpiricalNormalization([2])
    normalizer.load_state_dict(
        {
            "_mean": torch.tensor([[2.0, 3.0]]),
            "_var": torch.tensor([[4.0, 9.0]]),
            "_std": torch.tensor([[2.0, 3.0]]),
            "_count": torch.tensor(204800, dtype=torch.int64),
        }
    )
    normalizer.eval()
    normalizer(torch.tensor([[4.0, 6.0]]))
    assert normalizer.count == 204800


def test_training_continues_from_restored_count_without_overwrite():
    normalizer = EmpiricalNormalization([1])
    normalizer.load_state_dict(
        {
            "_mean": torch.tensor([[10.0]]),
            "_var": torch.tensor([[4.0]]),
            "_std": torch.tensor([[2.0]]),
            "_count": torch.tensor(100, dtype=torch.int64),
        }
    )
    normalizer.train()
    normalizer(torch.tensor([[20.0]]))
    assert normalizer.count == 101
    assert normalizer.mean.item() == pytest.approx((1000.0 + 20.0) / 101.0)


def test_runner_normalizes_initial_observations_before_first_action():
    source = (
        Path(__file__).resolve().parents[1]
        / "rsl_rl/rsl_rl/runners/on_policy_runner.py"
    ).read_text(encoding="utf-8")
    train_mode = source.index("self.train_mode()")
    loop = source.index("for it in range(start_iter, tot_iter)")
    actor_normalization = source.index("obs = self.obs_normalizer(obs)", train_mode)
    critic_normalization = source.index(
        "critic_obs = self.critic_obs_normalizer(critic_obs)", train_mode
    )
    assert train_mode < actor_normalization < loop
    assert train_mode < critic_normalization < loop
