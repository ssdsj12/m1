from __future__ import annotations

from pathlib import Path

import torch

from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_load_defaults_to_resetting_checkpoint_std() -> None:
    source = _source()

    assert "def load(self, path, load_optimizer=True, keep_std=False):" in source
    assert "if not keep_std:" in source
    assert 'state_dict.pop("std", None)' in source


def test_runner_load_can_keep_checkpoint_std_when_requested() -> None:
    source = _source()

    assert source.index("if not keep_std:") < source.index('state_dict.pop("std", None)')
    assert source.index('state_dict.pop("std", None)') < source.index(
        "self.alg.actor_critic.load_state_dict(state_dict, strict=False)"
    )


def test_runner_can_save_initial_policy_before_learn(tmp_path) -> None:
    class Env:
        num_actions = 2
        num_envs = 1
        cfg = object()

        def get_observations(self):
            return torch.zeros(1, 3), {"observations": {}}

    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 1,
        "empirical_normalization": False,
        "algorithm": {"class_name": "PPO"},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [4],
            "critic_hidden_dims": [4],
        },
    }
    runner = OnPolicyRunner(Env(), cfg, log_dir=str(tmp_path), device="cpu")
    checkpoint = tmp_path / "initial.pt"

    runner.save(str(checkpoint))

    assert checkpoint.is_file()
    assert torch.load(checkpoint, weights_only=False)["iter"] == 0


class _NormalizedEnv:
    num_actions = 2
    num_envs = 1
    cfg = object()

    def get_observations(self):
        observations = torch.zeros(1, 3)
        return observations, {"observations": {"critic": observations.clone()}}


def _normalized_cfg():
    return {
        "num_steps_per_env": 2,
        "save_interval": 1,
        "empirical_normalization": True,
        "algorithm": {"class_name": "PPO"},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [4],
            "critic_hidden_dims": [4],
        },
    }


def test_normalized_checkpoint_saves_and_requires_both_normalizer_states(tmp_path):
    checkpoint = tmp_path / "normalized.pt"
    runner = OnPolicyRunner(_NormalizedEnv(), _normalized_cfg(), device="cpu")
    runner.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)

    assert "obs_norm_state_dict" in payload
    assert "critic_obs_norm_state_dict" in payload
    for missing_key in ("obs_norm_state_dict", "critic_obs_norm_state_dict"):
        broken = tmp_path / f"missing_{missing_key}.pt"
        broken_payload = dict(payload)
        broken_payload.pop(missing_key)
        torch.save(broken_payload, broken)
        fresh = OnPolicyRunner(_NormalizedEnv(), _normalized_cfg(), device="cpu")
        try:
            fresh.load(str(broken), load_optimizer=False, keep_std=True)
        except KeyError as error:
            assert missing_key in str(error)
        else:
            raise AssertionError(f"missing {missing_key} must fail closed")


def test_inference_policy_freezes_actor_and_normalizer_statistics():
    runner = OnPolicyRunner(_NormalizedEnv(), _normalized_cfg(), device="cpu")
    runner.obs_normalizer.train()
    runner.obs_normalizer(torch.ones(2, 3))
    count_before = runner.obs_normalizer.count

    policy = runner.get_inference_policy(device="cpu")
    policy(torch.full((2, 3), 9.0))

    assert runner.alg.actor_critic.training is False
    assert runner.obs_normalizer.training is False
    assert runner.critic_obs_normalizer.training is False
    assert runner.obs_normalizer.count == count_before
