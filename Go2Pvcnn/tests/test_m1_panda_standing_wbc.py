from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.standing_wbc import (
    StandingWbcCfg,
    StandingWbcInput,
    build_standing_wbc_problem,
    solve_standing_wbc,
)


def _selector(rows, dimension=31, dtype=torch.float64):
    result = torch.zeros(len(rows), dimension, dtype=dtype)
    for output, source in enumerate(rows):
        result[output, source] = 1.0
    return result


def _input(dtype=torch.float64):
    leg_indices = torch.arange(6, 18, dtype=torch.long)
    wheel_indices = torch.arange(18, 22, dtype=torch.long)
    arm_indices = torch.arange(22, 29, dtype=torch.long)
    return StandingWbcInput(
        mass_matrix=torch.eye(31, dtype=dtype),
        bias_force=torch.zeros(31, dtype=dtype),
        contact_jacobian=torch.zeros(12, 31, dtype=dtype),
        contact_jacobian_dot_qd=torch.zeros(12, dtype=dtype),
        mount_wrench_jacobian=torch.zeros(6, 31, dtype=dtype),
        external_wrench=torch.zeros(6, dtype=dtype),
        balance_jacobian=_selector((2, 3, 4), dtype=dtype),
        balance_acceleration=torch.zeros(3, dtype=dtype),
        base_jacobian=_selector(tuple(range(6)), dtype=dtype),
        base_acceleration=torch.zeros(6, dtype=dtype),
        leg_generalized_indices=leg_indices,
        wheel_generalized_indices=wheel_indices,
        arm_generalized_indices=arm_indices,
        leg_acceleration=torch.zeros(12, dtype=dtype),
        wheel_acceleration=torch.zeros(4, dtype=dtype),
        arm_acceleration=torch.zeros(7, dtype=dtype),
        qdd_lower=torch.full((31,), -10.0, dtype=dtype),
        qdd_upper=torch.full((31,), 10.0, dtype=dtype),
        effort_limit=torch.full((23,), 100.0, dtype=dtype),
        friction_coefficient=0.7,
    )


def test_standing_wbc_default_priority_weights_are_frozen():
    cfg = StandingWbcCfg()

    assert cfg.balance_weight == pytest.approx(1.0e6)
    assert cfg.base_pose_weight == pytest.approx(1.0e5)
    assert cfg.leg_posture_weight == pytest.approx(1.0e4)
    assert cfg.arm_tracking_weight == pytest.approx(1.0e3)
    assert cfg.wheel_stop_weight == pytest.approx(1.0e3)
    assert cfg.force_equalization_weight == pytest.approx(10.0)
    assert cfg.regularization == pytest.approx(1.0e-6)
    assert cfg.balance_weight > cfg.base_pose_weight > cfg.leg_posture_weight
    assert cfg.leg_posture_weight > cfg.arm_tracking_weight


def test_problem_has_31_accelerations_12_contact_forces_and_hard_equalities():
    state = _input()

    assembled = build_standing_wbc_problem(state)

    qp = assembled.qp
    assert qp.hessian.shape == (43, 43)
    assert qp.gradient.shape == (43,)
    assert qp.equality_matrix.shape == (18, 43)
    assert qp.equality_rhs.shape == (18,)
    assert torch.equal(qp.equality_matrix[:6, :31], state.mass_matrix[:6])
    assert torch.equal(
        qp.equality_matrix[:6, 31:],
        -state.contact_jacobian.transpose(0, 1)[:6],
    )
    assert torch.equal(qp.equality_matrix[6:, :31], state.contact_jacobian)
    assert torch.count_nonzero(qp.equality_matrix[6:, 31:]) == 0
    assert torch.equal(
        qp.equality_rhs[6:], -state.contact_jacobian_dot_qd
    )


def test_external_wrench_maps_through_mount_jacobian_into_dynamics_and_torque():
    state = _input()
    mount = state.mount_wrench_jacobian.clone()
    mount[0, 0] = 1.5
    mount[1, 6] = -2.0
    bias = state.bias_force.clone()
    bias[0] = 4.0
    bias[6] = 5.0
    state = replace(
        state,
        mount_wrench_jacobian=mount,
        external_wrench=torch.tensor(
            [2.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
        ),
        bias_force=bias,
    )

    assembled = build_standing_wbc_problem(state)

    expected_external = mount.transpose(0, 1) @ state.external_wrench
    assert torch.equal(assembled.external_generalized_force, expected_external)
    assert assembled.qp.equality_rhs[0].item() == pytest.approx(-1.0)
    assert assembled.torque_offset[0].item() == pytest.approx(11.0)


def test_each_contact_has_positive_normal_and_four_sided_friction_pyramid():
    state = _input()
    assembled = build_standing_wbc_problem(state)
    inequalities = assembled.qp.inequality_matrix
    upper = assembled.qp.inequality_upper

    assert inequalities.shape == (66, 43)
    assert torch.count_nonzero(inequalities[:20, :31]) == 0
    assert torch.count_nonzero(upper[:20]) == 0
    expected = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -0.7],
            [-1.0, 0.0, -0.7],
            [0.0, 1.0, -0.7],
            [0.0, -1.0, -0.7],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(inequalities[:5, 31:34], expected)
    for contact in range(4):
        force_slice = slice(31 + 3 * contact, 31 + 3 * contact + 3)
        row_slice = slice(5 * contact, 5 * contact + 5)
        assert torch.equal(inequalities[row_slice, force_slice], expected)


def test_torque_recovery_matrix_and_limits_match_actuated_dynamics_rows():
    state = _input()
    mass = torch.arange(31 * 31, dtype=torch.float64).reshape(31, 31) / 100.0
    contact = torch.arange(12 * 31, dtype=torch.float64).reshape(12, 31) / 1000.0
    state = replace(state, mass_matrix=mass, contact_jacobian=contact)

    assembled = build_standing_wbc_problem(state)
    controlled = torch.cat(
        (
            state.leg_generalized_indices,
            state.wheel_generalized_indices,
            state.arm_generalized_indices,
        )
    )
    expected_matrix = torch.cat(
        (
            mass.index_select(0, controlled),
            -contact.transpose(0, 1).index_select(0, controlled),
        ),
        dim=1,
    )

    assert torch.equal(assembled.torque_matrix, expected_matrix)
    assert torch.equal(assembled.qp.inequality_matrix[20:43], expected_matrix)
    assert torch.equal(assembled.qp.inequality_matrix[43:66], -expected_matrix)
    assert torch.equal(
        assembled.qp.inequality_upper[20:43],
        state.effort_limit - assembled.torque_offset,
    )
    assert torch.equal(
        assembled.qp.inequality_upper[43:66],
        state.effort_limit + assembled.torque_offset,
    )


def test_balance_objective_has_greater_curvature_than_arm_tracking():
    state = _input()

    assembled = build_standing_wbc_problem(state)

    balance_diagonal = assembled.qp.hessian[2, 2]
    arm_diagonal = assembled.qp.hessian[22, 22]
    assert balance_diagonal > arm_diagonal * 100.0


def test_feasible_zero_state_solves_and_recovers_zero_effort():
    result = solve_standing_wbc(_input())

    assert result.qp_result.success
    assert result.effort is not None
    assert result.qdd.shape == (31,)
    assert result.contact_force.shape == (4, 3)
    assert result.effort.shape == (23,)
    assert torch.count_nonzero(result.qdd) == 0
    assert torch.count_nonzero(result.contact_force) == 0
    assert torch.count_nonzero(result.effort) == 0
    assert set(result.task_residuals) == {
        "balance",
        "base",
        "legs",
        "wheels",
        "arm",
        "contact",
    }


def test_effort_limit_is_enforced_when_arm_target_is_large():
    state = _input()
    state = replace(
        state,
        arm_acceleration=torch.full((7,), 5.0, dtype=torch.float64),
        effort_limit=torch.full((23,), 0.1, dtype=torch.float64),
    )

    result = solve_standing_wbc(state)

    assert result.qp_result.success
    assert result.effort is not None
    assert torch.max(torch.abs(result.effort)).item() <= 0.1 + 1.0e-8


def test_infeasible_qdd_bounds_return_no_new_effort_command():
    state = _input()
    lower = state.qdd_lower.clone()
    upper = state.qdd_upper.clone()
    lower[0] = 1.0
    upper[0] = 0.0
    state = replace(state, qdd_lower=lower, qdd_upper=upper)

    result = solve_standing_wbc(state)

    assert not result.qp_result.success
    assert result.effort is None
    assert torch.isfinite(result.qdd).all()
    assert torch.isfinite(result.contact_force).all()


def test_batch_size_greater_than_one_is_rejected_in_c0():
    state = _input()
    state = replace(state, mass_matrix=torch.stack((state.mass_matrix,) * 2))

    with pytest.raises(ValueError, match="C0 standing WBC supports one environment"):
        build_standing_wbc_problem(state)


def test_malformed_shape_duplicate_indices_and_non_finite_input_are_rejected():
    state = _input()
    with pytest.raises(
        ValueError, match=r"contact_jacobian must end with shape \(12, 31\)"
    ):
        build_standing_wbc_problem(
            replace(state, contact_jacobian=torch.zeros(31, 12))
        )

    duplicate_arm = state.arm_generalized_indices.clone()
    duplicate_arm[0] = state.leg_generalized_indices[0]
    with pytest.raises(ValueError, match="controlled generalized indices must be unique"):
        build_standing_wbc_problem(
            replace(state, arm_generalized_indices=duplicate_arm)
        )

    bad_bias = state.bias_force.clone()
    bad_bias[0] = float("nan")
    with pytest.raises(ValueError, match="bias_force must contain only finite values"):
        build_standing_wbc_problem(replace(state, bias_force=bad_bias))
