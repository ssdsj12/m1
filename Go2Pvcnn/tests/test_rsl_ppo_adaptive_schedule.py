from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import on_policy_runner


def _tiny_ppo(**overrides) -> PPO:
    actor = ActorCritic(
        3,
        3,
        2,
        actor_hidden_dims=[4, 4],
        critic_hidden_dims=[4, 4],
        init_noise_std=0.01,
        noise_std_type="scalar",
    )
    kwargs = {
        "learning_rate": 1.0e-4,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "min_learning_rate": 1.0e-6,
        "max_learning_rate": 3.0e-4,
        "clip_min_std": 0.005,
        "clip_max_std": 0.05,
    }
    kwargs.update(overrides)
    return PPO(actor, **kwargs)


def test_adaptive_lr_moves_toward_desired_kl():
    ppo = _tiny_ppo()

    assert ppo._adapt_learning_rate(0.03) == "decrease"
    assert ppo.learning_rate == pytest.approx(1.0e-4 / 1.5)
    assert ppo.last_kl_mean == pytest.approx(0.03)
    assert ppo.last_lr_adjustment == "decrease"

    assert ppo._adapt_learning_rate(0.001) == "increase"
    assert ppo.learning_rate == pytest.approx(1.0e-4)


def test_adaptive_lr_obeys_configured_bounds():
    ppo = _tiny_ppo()

    ppo.learning_rate = 1.0e-6
    assert ppo._adapt_learning_rate(0.03) == "hold"
    assert ppo.learning_rate == pytest.approx(1.0e-6)

    ppo.learning_rate = 3.0e-4
    assert ppo._adapt_learning_rate(0.001) == "hold"
    assert ppo.learning_rate == pytest.approx(3.0e-4)


def test_policy_std_is_clamped_in_physical_units():
    ppo = _tiny_ppo()
    with torch.no_grad():
        ppo.actor_critic.std.copy_(torch.tensor([0.001, 0.2]))

    ppo._clamp_policy_std()

    assert torch.equal(ppo.actor_critic.std, torch.tensor([0.005, 0.05]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_learning_rate": 0.0}, "min_learning_rate"),
        ({"max_learning_rate": 5.0e-5}, "learning-rate bounds"),
        ({"clip_max_std": 0.001}, "std bounds"),
    ],
)
def test_invalid_optimizer_bounds_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tiny_ppo(**overrides)


def test_runner_logs_kl_and_lr_adjustment_diagnostics():
    source = Path(on_policy_runner.__file__).read_text(encoding="utf-8")

    assert 'self.writer.add_scalar("Loss/kl"' in source
    assert 'self.writer.add_scalar("Loss/lr_adjustment"' in source
