import torch
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_contracts import (
    WIRE_PPO_ACTION_DIM,
    WirePpoObservationSpec,
)
from go2_pvcnn.tasks.m1_wire_ppo_curriculum import WirePpoStage
from go2_pvcnn.tasks.m1_wire_ppo_env import M1WirePpoEnv


class _FakeWrapper:
    def __init__(self):
        self.step_count = 0
        self.last_latent = None
        self.last_snapshot = object()
        self.wire_acceptance_diagnostics = {
            "acceptance_phase": "approach",
            "success": False,
            "rejection_reason": None,
            "left_valid_contacts": 0,
            "right_valid_contacts": 0,
            "left_contact_mask": [False] * 5,
            "right_contact_mask": [False] * 5,
            "left_fingertip_forces_b": [[0.0, 0.0, 0.0]] * 5,
            "right_fingertip_forces_b": [[0.0, 0.0, 0.0]] * 5,
            "left_slip_m": 0.0,
            "right_slip_m": 0.0,
            "lift_elapsed_s": 0.0,
            "hold_elapsed_s": 0.0,
            "wire_center_height_m": 1.0,
            "feasibility": {"safety": True},
        }

    def reset(self, *, seed):
        self.seed = seed
        return self.last_snapshot

    def step_latent(self, action):
        self.last_latent = action.clone()
        self.step_count += 1
        return None

    def close(self):
        pass


def test_reset_and_step_have_contract_shape_and_diagnostics():
    env = M1WirePpoEnv(_FakeWrapper(), stage="approval", max_episode_steps=4)
    obs, info = env.reset(seed=42)
    assert obs.shape == (WirePpoObservationSpec().observation_dim,)
    assert obs.dtype == torch.float32
    assert torch.isfinite(obs).all()
    assert info["stage"] == "approval"
    next_obs, reward, terminated, truncated, step_info = env.step(torch.zeros(WIRE_PPO_ACTION_DIM))
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert step_info["episode_step"] == 1
    assert env.wrapper.last_latent.shape == (WIRE_PPO_ACTION_DIM,)


def test_invalid_action_is_rejected_without_wrapper_mutation():
    wrapper = _FakeWrapper()
    env = M1WirePpoEnv(wrapper)
    env.reset(seed=7)
    before = (wrapper.step_count, wrapper.last_latent)
    with pytest.raises((TypeError, ValueError)):
        env.step(torch.full((WIRE_PPO_ACTION_DIM,), 2.0))
    assert (wrapper.step_count, wrapper.last_latent) == before


def test_stage_string_is_normalized_and_timeout_truncates():
    env = M1WirePpoEnv(_FakeWrapper(), stage=WirePpoStage.ACTION, max_episode_steps=1)
    env.reset()
    _, _, terminated, truncated, info = env.step(torch.zeros(WIRE_PPO_ACTION_DIM))
    assert terminated is False
    assert truncated is True
    assert info["stage"] == "action"


def test_action_dimension_is_strict():
    env = M1WirePpoEnv(_FakeWrapper())
    env.reset()
    with pytest.raises(ValueError):
        env.step(torch.zeros(14))
