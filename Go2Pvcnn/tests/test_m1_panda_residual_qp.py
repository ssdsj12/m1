import torch

from go2_pvcnn.control.m1_panda_coordination.standing_wbc import (
    build_standing_wbc_problem,
)
from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    apply_residual_to_wbc,
)

from test_m1_panda_residual_wbc import _command, _limits, _wbc_input


def _assert_problem_equal(first, second):
    for name in ("hessian", "gradient", "equality_matrix", "equality_rhs", "inequality_matrix", "inequality_upper", "lower_bound", "upper_bound"):
        assert torch.equal(getattr(first.qp, name), getattr(second.qp, name)), name
    assert torch.equal(first.external_generalized_force, second.external_generalized_force)
    assert torch.equal(first.torque_matrix, second.torque_matrix)
    assert torch.equal(first.torque_offset, second.torque_offset)
    for name in first.task_matrices:
        assert torch.equal(first.task_matrices[name], second.task_matrices[name])
        assert torch.equal(first.task_targets[name], second.task_targets[name])


def test_six_axis_residual_enters_generalized_force_with_expected_sign():
    state = _wbc_input()
    command = _command([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.0, 0.0])
    transformed, _ = apply_residual_to_wbc(
        state,
        command,
        nominal_leg_target=torch.zeros(12, dtype=torch.float64),
        leg_soft_limits=_limits(),
        safety_scale=1.0,
    )

    problem = build_standing_wbc_problem(transformed)

    assert torch.equal(
        problem.external_generalized_force,
        transformed.mount_wrench_jacobian.transpose(0, 1) @ command.wrench_b,
    )
    assert torch.equal(problem.external_generalized_force[:6], command.wrench_b)


def test_zero_residual_qp_problem_is_exactly_equal_to_baseline():
    state = _wbc_input()
    transformed, _ = apply_residual_to_wbc(
        state,
        _command([0.0] * 8),
        nominal_leg_target=torch.zeros(12, dtype=torch.float64),
        leg_soft_limits=_limits(),
        safety_scale=1.0,
    )

    _assert_problem_equal(
        build_standing_wbc_problem(state),
        build_standing_wbc_problem(transformed),
    )


def test_positive_and_negative_wrench_problems_remain_finite():
    state = _wbc_input()
    for sign in (-1.0, 1.0):
        transformed, _ = apply_residual_to_wbc(
            state,
            _command([sign, sign, sign, sign, sign, sign, 0.0, 0.0]),
            nominal_leg_target=torch.zeros(12, dtype=torch.float64),
            leg_soft_limits=_limits(),
            safety_scale=1.0,
        )
        problem = build_standing_wbc_problem(transformed)
        tensors = (
            problem.qp.hessian,
            problem.qp.gradient,
            problem.qp.equality_matrix,
            problem.qp.equality_rhs,
            problem.qp.inequality_matrix,
            problem.qp.inequality_upper,
        )
        assert all(torch.isfinite(value).all() for value in tensors)
