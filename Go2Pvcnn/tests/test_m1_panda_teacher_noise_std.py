from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsl_rl.modules import ActorCritic
from rsl_rl.runners import on_policy_runner


def _actor(mode: str, init: float = 0.01) -> ActorCritic:
    return ActorCritic(
        3,
        3,
        2,
        actor_hidden_dims=[4, 4],
        critic_hidden_dims=[4, 4],
        init_noise_std=init,
        noise_std_type=mode,
    )


def test_scalar_std_is_used_directly_and_keeps_legacy_state_key():
    actor = _actor("scalar", 0.01)

    actor.update_distribution(torch.zeros(5, 3))

    assert set(name for name in actor.state_dict() if "std" in name) == {"std"}
    assert torch.allclose(actor.action_std, torch.full((5, 2), 0.01))


def test_log_std_is_exponentiated_and_has_log_state_key():
    actor = _actor("log", 0.01)

    actor.update_distribution(torch.zeros(5, 3))

    assert set(name for name in actor.state_dict() if "std" in name) == {
        "log_std"
    }
    assert torch.allclose(actor.action_std, torch.full((5, 2), 0.01))


@pytest.mark.parametrize("mode", ["scalar", "log"])
def test_clip_std_uses_effective_units_for_both_modes(mode):
    actor = _actor(mode, 1.0e-8)

    actor.clip_std(min=0.001)

    assert torch.allclose(
        actor.effective_action_std, torch.full((2,), 0.001)
    )


def test_unknown_noise_mode_fails():
    with pytest.raises(ValueError, match="noise_std_type"):
        _actor("softplus")


def test_runner_noise_diagnostics_preserve_legacy_direct_std_modules():
    class LegacyActor:
        std = torch.tensor([0.1, 0.2])

    raw, effective = on_policy_runner.policy_noise_diagnostics(LegacyActor())

    assert raw == pytest.approx(0.15)
    assert effective == pytest.approx(0.15)


def test_runner_console_uses_effective_action_std_name():
    source = Path(on_policy_runner.__file__).read_text(encoding="utf-8")

    assert "mean_std.item()" not in source
    assert source.count("mean_action_std.item()") >= 3
