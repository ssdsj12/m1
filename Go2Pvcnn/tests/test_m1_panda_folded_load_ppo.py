from __future__ import annotations

import types

import pytest
import torch

from agent import get_m1_panda_folded_load_train_cfg
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def _actor(**overrides) -> ActorCritic:
    kwargs = {
        "num_actor_obs": 3,
        "num_critic_obs": 3,
        "num_actions": 2,
        "actor_hidden_dims": [4],
        "critic_hidden_dims": [4],
        "init_noise_std": 0.005,
        "noise_std_type": "scalar",
    }
    kwargs.update(overrides)
    return ActorCritic(**kwargs)


def _ppo(**overrides) -> PPO:
    kwargs = {
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-5,
        "min_learning_rate": 1.0e-6,
        "max_learning_rate": 1.0e-4,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "kl_abort_threshold": 0.015,
        "clip_min_std": 0.005,
        "clip_max_std": 0.02,
        "max_grad_norm": 0.5,
    }
    kwargs.update(overrides)
    return PPO(_actor(), **kwargs)


class _FakeStorage:
    def __init__(self, batches: int):
        self.batches = batches
        self.cleared = False

    @staticmethod
    def _batch():
        size = 4
        return (
            torch.zeros(size, 3),
            torch.zeros(size, 3),
            torch.zeros(size, 2),
            torch.zeros(size, 1),
            torch.ones(size, 1),
            torch.zeros(size, 1),
            torch.zeros(size, 1),
            torch.zeros(size, 2),
            torch.full((size, 2), 0.005),
            (None, None),
            None,
            None,
            None,
        )

    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        assert num_mini_batches * num_learning_epochs == self.batches
        for _ in range(self.batches):
            yield self._batch()

    def clear(self):
        self.cleared = True


def test_folded_load_ppo_configuration_is_exact_and_fresh():
    cfg = get_m1_panda_folded_load_train_cfg()
    assert cfg["num_steps_per_env"] == 256
    assert cfg["save_interval"] == 100
    assert cfg["algorithm"] == {
        "class_name": "PPO",
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-5,
        "min_learning_rate": 1.0e-6,
        "max_learning_rate": 1.0e-4,
        "clip_param": 0.2,
        "gamma": 0.9995,
        "lam": 0.995,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.0,
        "clip_min_std": 0.005,
        "clip_max_std": 0.02,
        "max_grad_norm": 0.5,
        "use_clipped_value_loss": True,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "kl_abort_threshold": 0.015,
    }
    assert cfg["policy"]["active_action_mask"] == [1] * 16 + [0] * 7
    assert cfg["policy"]["init_noise_std"] == 0.005
    assert cfg["policy"]["zero_actor_output"] is True
    assert get_m1_panda_folded_load_train_cfg() is not cfg


def test_zero_actor_output_initialization_zeros_all_output_rows():
    actor = _actor(zero_actor_output=True)
    final = [module for module in actor.actor if isinstance(module, torch.nn.Linear)][-1]
    assert final.weight.eq(0.0).all()
    assert final.bias.eq(0.0).all()
    assert actor.act_inference(torch.randn(3, 3)).eq(0.0).all()


def test_update_aborts_remaining_minibatches_after_kl_threshold():
    ppo = _ppo()
    ppo.storage = _FakeStorage(batches=8)
    kl_values = iter((0.0, 0.021))
    ppo._compute_kl_mean = types.MethodType(
        lambda self, old_mu, old_sigma, mu, sigma: next(kl_values), ppo
    )
    optimizer_steps = 0
    original_step = ppo.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    ppo.optimizer.step = counted_step

    value_loss, surrogate_loss = ppo.update()

    assert optimizer_steps == 1
    assert ppo.last_completed_mini_batches == 1
    assert ppo.last_kl_aborted is True
    assert ppo.last_kl_mean == pytest.approx(0.021)
    assert ppo.last_kl_max == pytest.approx(0.021)
    assert ppo.learning_rate < 1.0e-5
    assert ppo.storage.cleared is True
    assert torch.isfinite(torch.tensor([value_loss, surrogate_loss])).all()


def test_next_update_resets_abort_diagnostics_and_can_complete():
    ppo = _ppo(num_learning_epochs=1, num_mini_batches=2)
    first_storage = _FakeStorage(batches=2)
    ppo.storage = first_storage
    ppo._compute_kl_mean = types.MethodType(
        lambda self, old_mu, old_sigma, mu, sigma: 0.020, ppo
    )
    ppo.update()
    assert ppo.last_kl_aborted is True
    assert ppo.last_completed_mini_batches == 0

    ppo.storage = _FakeStorage(batches=2)
    ppo._compute_kl_mean = types.MethodType(
        lambda self, old_mu, old_sigma, mu, sigma: 0.005, ppo
    )
    ppo.update()
    assert ppo.last_kl_aborted is False
    assert ppo.last_completed_mini_batches == 2


def test_kl_uses_only_actor_active_dimensions():
    actor = _actor(active_action_mask=[1, 0])
    ppo = PPO(actor, desired_kl=0.01, schedule="adaptive")
    old_mu = torch.tensor([[0.0, 100.0]])
    new_mu = torch.tensor([[0.0, -100.0]])
    sigma = torch.ones_like(old_mu)
    assert ppo._compute_kl_mean(old_mu, sigma, new_mu, sigma) == pytest.approx(0.0, abs=2e-5)


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_kl_abort_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="kl_abort_threshold"):
        _ppo(kl_abort_threshold=threshold)


def test_none_kl_abort_threshold_preserves_legacy_behavior():
    ppo = _ppo(kl_abort_threshold=None, num_learning_epochs=1, num_mini_batches=2)
    ppo.storage = _FakeStorage(batches=2)
    ppo._compute_kl_mean = types.MethodType(
        lambda self, old_mu, old_sigma, mu, sigma: 100.0, ppo
    )
    ppo.update()
    assert ppo.last_kl_aborted is False
    assert ppo.last_completed_mini_batches == 2
