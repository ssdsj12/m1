from __future__ import annotations

import pytest
import torch

from rsl_rl.modules import ActorCritic


def _model(*, mask=None, noise_std_type="scalar") -> ActorCritic:
    return ActorCritic(
        num_actor_obs=5,
        num_critic_obs=5,
        num_actions=4,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        activation="elu",
        init_noise_std=0.2,
        noise_std_type=noise_std_type,
        active_action_mask=mask,
    )


def _last_actor_linear(model: ActorCritic) -> torch.nn.Linear:
    return [module for module in model.actor.modules() if isinstance(module, torch.nn.Linear)][-1]


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_inactive_actions_means_log_prob_entropy_and_gradients_are_masked(noise_std_type):
    torch.manual_seed(5)
    model = _model(mask=[1, 1, 0, 0], noise_std_type=noise_std_type)
    observations = torch.randn(6, 5)

    actions = model.act(observations)

    assert actions[:, 2:].eq(0.0).all()
    assert model.action_mean[:, 2:].eq(0.0).all()
    expected_log_prob = model.distribution.log_prob(actions)[:, :2].sum(dim=-1)
    expected_entropy = model.distribution.entropy()[:, :2].sum(dim=-1)
    torch.testing.assert_close(model.get_actions_log_prob(actions), expected_log_prob)
    torch.testing.assert_close(model.entropy, expected_entropy)

    loss = -(model.get_actions_log_prob(actions).mean() + 0.01 * model.entropy.mean())
    loss.backward()
    final = _last_actor_linear(model)
    assert final.weight.grad is not None
    assert final.bias.grad is not None
    assert final.weight.grad[2:].eq(0.0).all()
    assert final.bias.grad[2:].eq(0.0).all()
    assert model.noise_parameter.grad is not None
    assert model.noise_parameter.grad[2:].eq(0.0).all()


def test_inactive_actor_rows_start_and_remain_exactly_zero_after_optimizer_step():
    model = _model(mask=[1, 1, 0, 0])
    final = _last_actor_linear(model)
    assert final.weight[2:].eq(0.0).all()
    assert final.bias[2:].eq(0.0).all()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randn(10, 5)
    actions = model.act(observations)
    loss = -model.get_actions_log_prob(actions).mean() + model.evaluate(observations).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert final.weight[2:].eq(0.0).all()
    assert final.bias[2:].eq(0.0).all()


def test_inference_and_checkpoint_round_trip_keep_inactive_rows_zero(tmp_path):
    source = _model(mask=[1, 1, 0, 0])
    observations = torch.randn(3, 5)
    assert source.act_inference(observations)[:, 2:].eq(0.0).all()
    checkpoint = tmp_path / "actor.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = _model(mask=[1, 1, 0, 0])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(
        source.act_inference(observations), restored.act_inference(observations)
    )
    final = _last_actor_linear(restored)
    assert final.weight[2:].eq(0.0).all()
    assert final.bias[2:].eq(0.0).all()


@pytest.mark.parametrize(
    "mask, message",
    [
        ([1, 1, 0], "num_actions"),
        ([0, 0, 0, 0], "at least one"),
        ([1, 2, 0, 0], "0 or 1"),
        ("1100", "sequence"),
    ],
)
def test_invalid_active_action_masks_are_rejected(mask, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _model(mask=mask)


def test_legacy_no_mask_behavior_and_checkpoint_keys_are_preserved():
    torch.manual_seed(11)
    model = _model(mask=None)
    observations = torch.randn(4, 5)
    actions = model.act(observations)
    assert model.active_action_mask.all()
    assert actions.shape == (4, 4)
    torch.testing.assert_close(
        model.get_actions_log_prob(actions), model.distribution.log_prob(actions).sum(dim=-1)
    )
    assert "active_action_mask" not in model.state_dict()

