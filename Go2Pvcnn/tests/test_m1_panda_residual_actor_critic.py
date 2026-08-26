import pytest
import torch
from torch import nn

from go2_pvcnn.control.m1_panda_coordination.residual_actor_critic import (
    ResidualActorCritic,
)


def _policy():
    return ResidualActorCritic(
        num_actor_obs=103,
        num_critic_obs=103,
        num_actions=8,
        init_noise_std=0.01,
    )


def _linears(module):
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def test_grouped_actor_and_independent_critic_freeze_exact_architecture():
    policy = _policy()

    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.actor_m1)] == [(59, 128)]
    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.actor_arm)] == [(20, 64)]
    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.actor_wrench)] == [(6, 32)]
    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.actor_context)] == [(18, 32)]
    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.actor_head)] == [(256, 128), (128, 8)]
    assert [(layer.in_features, layer.out_features) for layer in _linears(policy.critic_head)] == [(256, 128), (128, 1)]
    actor_ids = {id(parameter) for name, parameter in policy.named_parameters() if name.startswith("actor_")}
    critic_ids = {id(parameter) for name, parameter in policy.named_parameters() if name.startswith("critic_")}
    assert actor_ids.isdisjoint(critic_ids)


def test_policy_implements_rsl_action_value_and_distribution_contract():
    policy = _policy()
    observation = torch.randn(4, 103)

    inference = policy.act_inference(observation)
    sampled = policy.act(observation)
    log_prob = policy.get_actions_log_prob(sampled)
    value = policy.evaluate(observation)

    assert inference.shape == (4, 8)
    assert sampled.shape == (4, 8)
    assert log_prob.shape == (4,)
    assert value.shape == (4, 1)
    assert policy.action_mean.shape == (4, 8)
    assert policy.action_std.shape == (4, 8)
    assert policy.entropy.shape == (4,)
    assert torch.equal(inference, policy.act_inference(observation))


def test_policy_rejects_wrong_width_and_nonfinite_observations():
    policy = _policy()

    with pytest.raises(ValueError, match="width 103"):
        policy.act_inference(torch.zeros(2, 102))
    observation = torch.zeros(2, 103)
    observation[0, 7] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        policy.act_inference(observation)


def test_policy_state_dict_strict_round_trip_and_std_clamp():
    policy = _policy()
    clone = _policy()
    clone.load_state_dict(policy.state_dict(), strict=True)
    observation = torch.randn(3, 103)

    assert torch.equal(policy.act_inference(observation), clone.act_inference(observation))
    with torch.no_grad():
        clone.std[:] = torch.tensor([0.001, 0.2] * 4)
    clone.clip_std(min=0.005, max=0.05)
    assert torch.equal(clone.std, torch.tensor([0.005, 0.05] * 4))


def test_policy_rejects_any_non_frozen_public_dimensions():
    with pytest.raises(ValueError, match="num_actor_obs must be 103"):
        ResidualActorCritic(102, 103, 8)
    with pytest.raises(ValueError, match="num_critic_obs must be 103"):
        ResidualActorCritic(103, 104, 8)
    with pytest.raises(ValueError, match="num_actions must be 8"):
        ResidualActorCritic(103, 103, 23)
