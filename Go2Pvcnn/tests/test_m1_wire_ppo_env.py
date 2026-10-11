import torch
import pytest
from dataclasses import replace
from types import SimpleNamespace

from tests.test_m1_bimanual_object_mpc import _snapshot

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
        self.last_snapshot = _snapshot()
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


def test_reset_height_comes_from_geometry_even_without_acceptance_diagnostics():
    wrapper = _FakeWrapper()
    wrapper.wire_acceptance_diagnostics = {}
    wrapper._wire_geometry_provider = lambda snapshot: SimpleNamespace(
        centerline=SimpleNamespace(positions=torch.tensor([[0., 0., .7], [0., .4, .7]])))
    env = M1WirePpoEnv(wrapper)
    env.reset()
    wrapper.wire_acceptance_diagnostics = {"wire_center_height_m": .7}
    _, _, _, _, info = env.step(torch.zeros(15))
    assert info["lift_height"] == pytest.approx(0.)


def test_mask_selects_actual_contact_fingers_not_first_n_forces():
    wrapper = _FakeWrapper()
    env = M1WirePpoEnv(wrapper)
    env.reset()
    for side in ("left", "right"):
        wrapper.wire_acceptance_diagnostics.update({
            f"{side}_contact_mask": [False, False, True, False, True],
            f"{side}_valid_contacts": 2,
            f"{side}_fingertip_forces_b": [[0., 0., x] for x in (0., 0., .3, 0., .4)],
        })
    state = env._wire_state(wrapper.last_snapshot, wrapper.wire_acceptance_diagnostics)
    assert state.contact_forces_left == (.3, .4)
    assert state.contact_forces_right == (.3, .4)
    wrapper.wire_acceptance_diagnostics["left_fingertip_forces_b"][2][2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        env._wire_state(wrapper.last_snapshot, wrapper.wire_acceptance_diagnostics)


def test_explicit_observation_layout_preserves_both_hands_and_arms():
    wrapper = _FakeWrapper()
    env = M1WirePpoEnv(wrapper)
    before, _ = env.reset()
    layout = env.observation_layout
    for name in ("left_arm", "right_arm", "left_hand", "right_hand"):
        original = getattr(wrapper.last_snapshot, name)
        wrapper.last_snapshot = replace(wrapper.last_snapshot, **{name: replace(original, q=original.q + .01)})
        after = env._build_observation(wrapper.last_snapshot, {})
        assert not torch.equal(before[layout[name]], after[layout[name]])
        before = after


def test_constant_action_has_zero_second_step_smoothness():
    env = M1WirePpoEnv(_FakeWrapper(), stage="action")
    env.reset()
    action = torch.full((15,), .3)
    env.step(action)
    assert env._previous_state.latent_delta_norm == pytest.approx(float(torch.linalg.vector_norm(action)))
    env.step(action)
    assert env._previous_state.latent_delta_norm == 0.


def test_stage_b_metrics_use_measured_geometry_and_explicit_target():
    wrapper = _FakeWrapper()
    measured = torch.tensor([[0., 0., .7], [0., .2, .72], [0., .4, .7]], dtype=torch.float64)
    target = measured.clone()
    target[:, 2] += torch.tensor([.05, .08, .05], dtype=torch.float64)
    wrapper.wire_acceptance_diagnostics.update({
        "left_contact_mask": [True, False, True, False, False],
        "right_contact_mask": [False, True, False, True, False],
        "left_fingertip_forces_b": [[0., 0., x] for x in (.3, 0., .4, 0., 0.)],
        "right_fingertip_forces_b": [[0., 0., x] for x in (0., .2, 0., .3, 0.)],
    })
    wrapper.runtime = SimpleNamespace(latest_reduced_wire_update=SimpleNamespace(state=SimpleNamespace(
        positions=measured, endpoint_velocities=torch.tensor([[0., 0., .1], [0., 0., 0.]]))))
    env = M1WirePpoEnv(wrapper, stage="action", target_centerline=target)
    env.reset()
    state = env._previous_state
    assert state.wire_centerline_error > 0.
    assert state.synchronization_error == pytest.approx(.1)
    assert state.tangent_error > 0.
    assert state.deformation_error > 0.
    assert state.force_balance_error == pytest.approx(.2)
    wrapper.runtime.latest_reduced_wire_update.state.positions = target
    env.step(torch.zeros(15))
    assert env._previous_state.wire_centerline_error == pytest.approx(0.)


def test_no_noarg_step_fallback_silently_discards_actions():
    wrapper = _FakeWrapper()
    wrapper.step_latent = None
    wrapper.step = lambda: None
    env = M1WirePpoEnv(wrapper)
    env.reset()
    with pytest.raises(TypeError, match="step_latent"):
        env.step(torch.zeros(15))


def test_raw_success_marker_cannot_bypass_measured_acceptance():
    wrapper = _FakeWrapper()
    wrapper.wire_acceptance_diagnostics["success"] = True
    env = M1WirePpoEnv(wrapper)
    env.reset()
    _, _, terminated, _, info = env.step(torch.zeros(15))
    assert not terminated and not info["success"]


def test_nonfinite_slip_is_rejected_instead_of_becoming_zero():
    wrapper = _FakeWrapper()
    env = M1WirePpoEnv(wrapper)
    env.reset()
    wrapper.wire_acceptance_diagnostics["left_slip_m"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        env.step(torch.zeros(15))
