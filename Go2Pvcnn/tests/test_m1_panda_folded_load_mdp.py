from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "go2_pvcnn"
    / "mdp"
    / "m1_panda_folded_load.py"
)
SPEC = importlib.util.spec_from_file_location("m1_panda_folded_load_mdp_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mdp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mdp)


def test_tracking_rewards_have_exact_scales():
    vx_error = torch.tensor([0.0, 0.05])
    wz_error = torch.tensor([0.0, 0.15])
    torch.testing.assert_close(
        mdp.track_vx_error(vx_error), torch.exp(-vx_error.square() / 0.05**2)
    )
    torch.testing.assert_close(
        mdp.track_wz_error(wz_error), torch.exp(-wz_error.square() / 0.15**2)
    )


def test_active_action_costs_use_only_first_16_coordinates():
    actions = torch.zeros(2, 23)
    previous = torch.zeros(2, 23)
    actions[:, 16:] = 100.0
    previous[:, 16:] = -100.0
    assert mdp.active_action_l2_tensor(actions).eq(0.0).all()
    assert mdp.active_action_rate_l2_tensor(actions, previous).eq(0.0).all()
    actions[:, 3] = 2.0
    previous[:, 3] = 1.0
    torch.testing.assert_close(mdp.active_action_l2_tensor(actions), torch.full((2,), 4.0))
    torch.testing.assert_close(mdp.active_action_rate_l2_tensor(actions, previous), torch.ones(2))


@pytest.mark.parametrize("shape", [(2, 22), (2, 24), (23,), (2, 23, 1)])
def test_active_action_cost_rejects_noncanonical_action_shape(shape):
    with pytest.raises(ValueError, match=r"\[N, 23\]"):
        mdp.active_action_l2_tensor(torch.zeros(shape))


def test_nonfinite_reward_input_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        mdp.track_vx_error(torch.tensor([torch.nan]))


def test_desired_twist_maps_episode_command_into_existing_six_value_slot():
    env = SimpleNamespace(
        num_envs=2,
        folded_load_commands=torch.tensor([[0.1, 0.0, -0.2], [-0.1, 0.0, 0.3]]),
    )
    expected = torch.tensor(
        [[0.1, 0.0, 0.0, 0.0, 0.0, -0.2], [-0.1, 0.0, 0.0, 0.0, 0.0, 0.3]]
    )
    torch.testing.assert_close(mdp.folded_load_desired_twist_b(env), expected)


def test_environment_tracking_and_stability_terms_use_body_state():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.1, 0.2, 0.3]]),
            root_ang_vel_b=torch.tensor([[0.4, 0.5, -0.2]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        folded_load_commands=torch.tensor([[0.1, 0.0, -0.2]]),
        scene={"robot": robot},
    )
    torch.testing.assert_close(mdp.folded_load_track_vx(env), torch.ones(1))
    torch.testing.assert_close(mdp.folded_load_track_wz(env), torch.ones(1))
    torch.testing.assert_close(mdp.folded_load_lateral_velocity_l2(env), torch.tensor([0.04]))

