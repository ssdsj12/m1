from dataclasses import fields

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.standing_wbc import StandingWbcInput
from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    ResidualWbcCfg,
    WholeBodyResidualCommand,
    apply_residual_to_wbc,
)


def _selector(rows, dimension=31):
    result = torch.zeros(len(rows), dimension, dtype=torch.float64)
    for output, source in enumerate(rows):
        result[output, source] = 1.0
    return result


def _wbc_input():
    mount = torch.zeros(6, 31, dtype=torch.float64)
    mount[:, :6] = torch.eye(6, dtype=torch.float64)
    return StandingWbcInput(
        mass_matrix=torch.eye(31, dtype=torch.float64),
        bias_force=torch.zeros(31, dtype=torch.float64),
        contact_jacobian=torch.zeros(12, 31, dtype=torch.float64),
        contact_jacobian_dot_qd=torch.zeros(12, dtype=torch.float64),
        mount_wrench_jacobian=mount,
        external_wrench=torch.zeros(6, dtype=torch.float64),
        balance_jacobian=_selector((2, 3, 4)),
        balance_acceleration=torch.zeros(3, dtype=torch.float64),
        base_jacobian=_selector(tuple(range(6))),
        base_acceleration=torch.arange(6, dtype=torch.float64),
        leg_generalized_indices=torch.arange(6, 18, dtype=torch.long),
        wheel_generalized_indices=torch.arange(18, 22, dtype=torch.long),
        arm_generalized_indices=torch.arange(22, 29, dtype=torch.long),
        leg_acceleration=torch.arange(12, dtype=torch.float64),
        wheel_acceleration=torch.zeros(4, dtype=torch.float64),
        arm_acceleration=torch.zeros(7, dtype=torch.float64),
        qdd_lower=torch.full((31,), -100.0, dtype=torch.float64),
        qdd_upper=torch.full((31,), 100.0, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        friction_coefficient=0.7,
    )


def _command(values):
    physical = torch.tensor(values, dtype=torch.float64)
    return WholeBodyResidualCommand(
        physical=physical,
        wrench_b=physical[:6].clone(),
        delta_height=physical[6].clone(),
        delta_stance=physical[7].clone(),
    )


def _limits(lower=-1.0, upper=1.0):
    return torch.tensor([[lower, upper]] * 12, dtype=torch.float64)


def test_zero_residual_keeps_every_wbc_field_equal():
    state = _wbc_input()
    nominal = torch.linspace(-0.3, 0.3, 12, dtype=torch.float64)

    result, leg_target = apply_residual_to_wbc(
        state,
        _command([0.0] * 8),
        nominal_leg_target=nominal,
        leg_soft_limits=_limits(),
        safety_scale=1.0,
    )

    for field in fields(StandingWbcInput):
        before = getattr(state, field.name)
        after = getattr(result, field.name)
        if isinstance(before, torch.Tensor):
            assert torch.equal(before, after), field.name
        else:
            assert before == after, field.name
    assert torch.equal(leg_target, nominal)


def test_height_and_stance_modify_only_approved_wbc_targets():
    state = _wbc_input()
    nominal = torch.zeros(12, dtype=torch.float64)
    command = _command([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.04, 0.08])

    result, leg_target = apply_residual_to_wbc(
        state,
        command,
        nominal_leg_target=nominal,
        leg_soft_limits=_limits(),
        safety_scale=1.0,
    )

    expected_offset = torch.zeros(12, dtype=torch.float64)
    expected_offset[[0, 3, 6, 9]] = torch.tensor(
        [0.08, -0.08, 0.08, -0.08], dtype=torch.float64
    )
    assert torch.equal(result.external_wrench, command.wrench_b)
    assert result.base_acceleration[2].item() == pytest.approx(
        state.base_acceleration[2].item() + 40.0 * 0.04
    )
    assert result.balance_acceleration[0].item() == pytest.approx(
        state.balance_acceleration[0].item() + 40.0 * 0.04
    )
    assert torch.equal(
        result.leg_acceleration - state.leg_acceleration,
        80.0 * expected_offset,
    )
    assert torch.equal(leg_target, expected_offset)
    assert torch.equal(result.base_acceleration[[0, 1, 3, 4, 5]], state.base_acceleration[[0, 1, 3, 4, 5]])
    assert torch.equal(result.balance_acceleration[1:], state.balance_acceleration[1:])


def test_safety_scale_applies_to_wrench_height_and_stance_together():
    state = _wbc_input()
    command = _command([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.02, 0.04])

    full, full_leg = apply_residual_to_wbc(
        state, command, nominal_leg_target=torch.zeros(12, dtype=torch.float64), leg_soft_limits=_limits(), safety_scale=1.0
    )
    half, half_leg = apply_residual_to_wbc(
        state, command, nominal_leg_target=torch.zeros(12, dtype=torch.float64), leg_soft_limits=_limits(), safety_scale=0.5
    )

    assert torch.equal(half.external_wrench, 0.5 * full.external_wrench)
    assert half.base_acceleration[2] - state.base_acceleration[2] == pytest.approx(
        0.5 * (full.base_acceleration[2] - state.base_acceleration[2]).item()
    )
    assert torch.equal(half_leg, 0.5 * full_leg)


def test_stance_target_clips_to_each_soft_limit_before_acceleration_mapping():
    state = _wbc_input()
    nominal = torch.zeros(12, dtype=torch.float64)
    limits = _limits()
    limits[0] = torch.tensor([-0.01, 0.01], dtype=torch.float64)
    limits[3] = torch.tensor([-0.02, 0.02], dtype=torch.float64)

    result, leg_target = apply_residual_to_wbc(
        state,
        _command([0.0] * 7 + [0.08]),
        nominal_leg_target=nominal,
        leg_soft_limits=limits,
        safety_scale=1.0,
    )

    assert leg_target[0].item() == pytest.approx(0.01)
    assert leg_target[3].item() == pytest.approx(-0.02)
    assert result.leg_acceleration[0] - state.leg_acceleration[0] == pytest.approx(0.8)
    assert result.leg_acceleration[3] - state.leg_acceleration[3] == pytest.approx(-1.6)


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_wbc_transform_rejects_invalid_safety_scale_without_mutation(scale):
    state = _wbc_input()
    before = state.base_acceleration.clone()

    with pytest.raises(ValueError, match="safety_scale"):
        apply_residual_to_wbc(
            state,
            _command([0.0] * 8),
            nominal_leg_target=torch.zeros(12, dtype=torch.float64),
            leg_soft_limits=_limits(),
            safety_scale=scale,
        )

    assert torch.equal(state.base_acceleration, before)


def test_wbc_transform_rejects_bad_command_and_posture_contracts():
    state = _wbc_input()
    command = _command([0.0] * 8)
    command = WholeBodyResidualCommand(
        physical=command.physical.float(),
        wrench_b=command.wrench_b.float(),
        delta_height=command.delta_height.float(),
        delta_stance=command.delta_stance.float(),
    )
    with pytest.raises(TypeError, match="dtype"):
        apply_residual_to_wbc(
            state,
            command,
            nominal_leg_target=torch.zeros(12, dtype=torch.float64),
            leg_soft_limits=_limits(),
            safety_scale=1.0,
        )
    with pytest.raises(ValueError, match="leg_soft_limits"):
        apply_residual_to_wbc(
            state,
            _command([0.0] * 8),
            nominal_leg_target=torch.zeros(12, dtype=torch.float64),
            leg_soft_limits=torch.zeros(12, 3, dtype=torch.float64),
            safety_scale=1.0,
        )


def test_residual_wbc_configuration_is_frozen():
    cfg = ResidualWbcCfg()
    assert cfg.base_height_position_gain == pytest.approx(40.0)
    assert cfg.leg_position_gain == pytest.approx(80.0)
    assert cfg.abad_indices == (0, 3, 6, 9)
    assert cfg.stance_signs == (1.0, -1.0, 1.0, -1.0)
