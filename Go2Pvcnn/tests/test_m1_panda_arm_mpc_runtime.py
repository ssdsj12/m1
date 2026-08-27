from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
    arm_kinematics_in_base_frame,
    build_arm_mpc_input_from_teacher_state,
)


DTYPE = torch.float64


class _QuaternionMath:
    @staticmethod
    def quat_inv(quaternion):
        result = quaternion.clone()
        result[..., 1:] *= -1.0
        return result

    @staticmethod
    def quat_mul(first, second):
        aw, ax, ay, az = first.unbind(dim=-1)
        bw, bx, by, bz = second.unbind(dim=-1)
        return torch.stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ),
            dim=-1,
        )

    @classmethod
    def quat_apply_inverse(cls, quaternion, vector):
        pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
        return cls.quat_mul(
            cls.quat_mul(cls.quat_inv(quaternion), pure), quaternion
        )[..., 1:]

    @staticmethod
    def axis_angle_from_quat(quaternion):
        vector = quaternion[..., 1:]
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(norm, quaternion[..., :1])
        return torch.where(norm > 1.0e-9, vector * angle / norm, 2.0 * vector)


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


def test_arm_kinematics_are_expressed_at_base_origin_in_base_axes():
    half = torch.tensor(torch.pi / 4.0, dtype=DTYPE)
    root_quat = torch.stack((torch.cos(half), torch.zeros((), dtype=DTYPE), torch.zeros((), dtype=DTYPE), torch.sin(half)))
    root_position = torch.tensor([2.0, 3.0, 1.0], dtype=DTYPE)
    hand_position = torch.tensor([2.0, 4.0, 1.5], dtype=DTYPE)
    hand_quat = root_quat.clone()
    reference_root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    reference_hand_quat = reference_root_quat.clone()
    jacobian_w = torch.zeros((6, 7), dtype=DTYPE)
    jacobian_w[0, 0] = 1.0
    jacobian_w[4, 1] = 1.0

    pose_b, jacobian_b = arm_kinematics_in_base_frame(
        _QuaternionMath,
        root_position_w=root_position,
        root_quaternion_w=root_quat,
        hand_position_w=hand_position,
        hand_quaternion_w=hand_quat,
        reference_root_quaternion_w=reference_root_quat,
        reference_hand_quaternion_w=reference_hand_quat,
        arm_jacobian_w=jacobian_w,
    )

    torch.testing.assert_close(
        pose_b, torch.tensor([1.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        jacobian_b[:3, 0], torch.tensor([0.0, -1.0, 0.0], dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        jacobian_b[3:, 1], torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
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
