import math

import pytest

from go2_pvcnn.tasks.m1_wire_ppo_curriculum import (
    WirePpoRewardCfg,
    WirePpoStage,
    WirePpoState,
    approval_episode,
    compute_wire_ppo_reward,
)


def accepted_state(**overrides):
    values = dict(
        approach_error=0.02,
        contacts_left=2,
        contacts_right=2,
        contact_forces_left=(0.3, 0.4),
        contact_forces_right=(0.25, 0.5),
        lift_height=0.05,
        hold_time=1.0,
        slip=0.02,
        safety_failure=False,
    )
    values.update(overrides)
    return WirePpoState(**values)


def test_approval_requires_all_acceptance_thresholds_and_strict_force():
    assert approval_episode(accepted_state())
    assert not approval_episode(accepted_state(contacts_left=1))
    assert not approval_episode(accepted_state(contact_forces_right=(0.2, 0.3)))
    assert not approval_episode(accepted_state(lift_height=0.049999))
    assert not approval_episode(accepted_state(hold_time=0.999999))
    assert not approval_episode(accepted_state(slip=0.020001))
    assert not approval_episode(accepted_state(safety_failure=True))


def test_approval_rejects_nonfinite_state_without_exception_or_success():
    assert not approval_episode(accepted_state(lift_height=math.nan))
    assert not approval_episode(accepted_state(contact_forces_left=(math.inf, 0.3)))


def test_stage_a_reward_contains_only_approval_terms():
    previous = accepted_state(approach_error=0.2, lift_height=0.0, hold_time=0.0)
    current = accepted_state(approach_error=0.1, lift_height=0.03, hold_time=0.4)
    breakdown = compute_wire_ppo_reward(previous, current, WirePpoStage.APPROVAL)
    assert breakdown.approach > 0
    assert breakdown.contact > 0
    assert breakdown.lift > 0
    assert breakdown.hold > 0
    assert breakdown.centerline == 0.0
    assert breakdown.tangent == 0.0
    assert breakdown.deformation == 0.0
    assert breakdown.total == pytest.approx(sum(breakdown.approval_terms().values()))


def test_stage_b_adds_wire_action_terms_and_latent_smoothness():
    previous = accepted_state(
        wire_centerline_error=0.2,
        tangent_error=0.3,
        deformation_error=0.4,
        synchronization_error=0.2,
        force_balance_error=0.3,
        latent_delta_norm=0.4,
    )
    current = accepted_state(
        wire_centerline_error=0.1,
        tangent_error=0.1,
        deformation_error=0.2,
        synchronization_error=0.1,
        force_balance_error=0.1,
        latent_delta_norm=0.1,
    )
    breakdown = compute_wire_ppo_reward(previous, current, WirePpoStage.ACTION)
    assert breakdown.centerline > 0
    assert breakdown.tangent > 0
    assert breakdown.deformation > 0
    assert breakdown.synchronization > 0
    assert breakdown.force_balance > 0
    assert breakdown.slip >= 0
    assert breakdown.latent_smoothness > 0
    assert breakdown.total == pytest.approx(sum(breakdown.action_terms().values()))


def test_failed_current_state_gets_safety_penalty_and_never_approval():
    previous = accepted_state()
    current = accepted_state(safety_failure=True)
    breakdown = compute_wire_ppo_reward(previous, current, WirePpoStage.ACTION)
    assert breakdown.safety < 0
    assert breakdown.total < 0
    assert not approval_episode(current)


@pytest.mark.parametrize("stage", [WirePpoStage.APPROVAL, WirePpoStage.ACTION])
def test_reward_rejects_unknown_stage_and_nonfinite_inputs(stage):
    with pytest.raises(ValueError):
        compute_wire_ppo_reward(accepted_state(approach_error=math.nan), accepted_state(), stage)


def test_reward_config_is_positive_and_customizable():
    cfg = WirePpoRewardCfg()
    assert cfg.safety_failure_penalty < 0
    assert cfg.contact_reward > 0
    assert cfg.action_centerline_weight > 0
