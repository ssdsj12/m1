from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
    build_arm_mpc_input_from_teacher_state,
)


DTYPE = torch.float64


def _state():
    mass = torch.arange(31 * 31, dtype=DTYPE).reshape(31, 31)
    bias = torch.arange(31, dtype=DTYPE) + 1000.0
    arm_indices = torch.arange(24, 31, dtype=torch.long)
    panda_jacobian = torch.arange(6 * 7, dtype=DTYPE).reshape(6, 7) / 10.0
    coordinated_jacobian = torch.cat(
        (torch.zeros((6, 3), dtype=DTYPE), panda_jacobian), dim=1
    )
    wbc_input = SimpleNamespace(
        mass_matrix=mass,
        bias_force=bias,
        arm_generalized_indices=arm_indices,
        effort_limit=torch.arange(23, dtype=DTYPE) + 10.0,
    )
    return SimpleNamespace(
        coord_q=torch.arange(10, dtype=DTYPE) / 10.0,
        coord_qd=torch.arange(10, dtype=DTYPE) / -10.0,
        coord_q_min=-torch.arange(1, 11, dtype=DTYPE),
        coord_q_max=torch.arange(1, 11, dtype=DTYPE),
        coord_v_max=torch.arange(1, 11, dtype=DTYPE) + 1.0,
        coord_a_max=torch.arange(1, 11, dtype=DTYPE) + 2.0,
        ee_pose=torch.arange(6, dtype=DTYPE) / 100.0,
        coordinated_jacobian=coordinated_jacobian,
        wbc_input=wbc_input,
    )


def test_runtime_input_slices_arm_dynamics_from_one_teacher_snapshot():
    state = _state()
    targets = state.ee_pose.expand(20, -1).clone()
    twists = torch.zeros((20, 6), dtype=DTYPE)

    result = build_arm_mpc_input_from_teacher_state(state, targets, twists)

    arm = state.wbc_input.arm_generalized_indices
    assert torch.equal(result.q, state.coord_q[-7:])
    assert torch.equal(result.qd, state.coord_qd[-7:])
    assert torch.equal(result.jacobian_b, state.coordinated_jacobian[:, -7:])
    assert torch.equal(
        result.arm_mass_matrix,
        state.wbc_input.mass_matrix.index_select(0, arm).index_select(1, arm),
    )
    assert torch.equal(
        result.base_arm_coupling,
        state.wbc_input.mass_matrix[:6].index_select(1, arm),
    )
    assert torch.equal(result.arm_bias, state.wbc_input.bias_force.index_select(0, arm))
    assert torch.equal(result.effort_max, state.wbc_input.effort_limit[-7:])
    assert torch.equal(result.ee_twist_b, result.jacobian_b @ result.qd)


def test_runtime_input_is_cpu_float64_clone_and_cannot_alias_live_state():
    state = _state()
    targets = state.ee_pose.expand(20, -1).clone()
    twists = torch.zeros((20, 6), dtype=DTYPE)

    result = build_arm_mpc_input_from_teacher_state(state, targets, twists)
    state.coord_q[-1] = 999.0
    targets[0, 0] = 999.0

    assert result.q.device.type == "cpu"
    assert result.q.dtype == DTYPE
    assert result.q[-1].item() != 999.0
    assert result.target_pose_b[0, 0].item() != 999.0


def test_runtime_input_rejects_nonfinite_or_wrong_horizon_atomically():
    state = _state()
    targets = state.ee_pose.expand(20, -1).clone()
    twists = torch.zeros((20, 6), dtype=DTYPE)

    with pytest.raises(ValueError, match="target_pose_b must have shape"):
        build_arm_mpc_input_from_teacher_state(state, targets[:19], twists)
    state.wbc_input.mass_matrix[0, 24] = torch.nan
    with pytest.raises(ValueError, match="base_arm_coupling must contain only finite"):
        build_arm_mpc_input_from_teacher_state(state, targets, twists)
