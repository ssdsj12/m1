import torch

from go2_pvcnn.control.m1_panda_coordination.recursive_dynamics import (
    low_pass_wrench,
    recursive_newton_euler_terms,
    recursive_newton_euler_reaction,
)


DTYPE = torch.float64


def test_rne_wrench_filter_is_causal_and_uses_frozen_half_gain():
    previous = torch.tensor([2.0] * 6, dtype=DTYPE)
    current = torch.tensor([6.0] * 6, dtype=DTYPE)

    torch.testing.assert_close(
        low_pass_wrench(previous, current),
        torch.tensor([4.0] * 6, dtype=DTYPE),
    )


def test_recursive_reaction_sums_link_gravity_and_shifts_to_base():
    masses = torch.tensor([2.0, 3.0], dtype=DTYPE)
    com_pos_w = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=DTYPE)
    com_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=DTYPE)
    inertia_com_local = torch.eye(3, dtype=DTYPE).repeat(2, 1, 1)
    linear_acc_w = torch.zeros((2, 3), dtype=DTYPE)
    angular_vel_w = torch.zeros((2, 3), dtype=DTYPE)
    angular_acc_w = torch.zeros((2, 3), dtype=DTYPE)
    gravity_w = torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE)

    result = recursive_newton_euler_reaction(
        masses,
        com_pos_w,
        com_quat_w,
        inertia_com_local,
        linear_acc_w,
        angular_vel_w,
        angular_acc_w,
        gravity_w,
        base_pos_w=torch.zeros(3, dtype=DTYPE),
        base_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
    )

    # PhysX link-acceleration convention maps the translational reaction with
    # the opposite sign to the angular reaction at the fixed mount.
    torch.testing.assert_close(
        result,
        torch.tensor([0.0, 0.0, 49.05, 0.0, 29.43, 0.0], dtype=DTYPE),
    )


def test_recursive_reaction_rejects_mismatched_or_nonfinite_link_data():
    values = dict(
        masses=torch.ones(2, dtype=DTYPE),
        com_pos_w=torch.zeros((2, 3), dtype=DTYPE),
        inertia_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=DTYPE),
        inertia_com_local=torch.eye(3, dtype=DTYPE).repeat(2, 1, 1),
        linear_acc_w=torch.zeros((2, 3), dtype=DTYPE),
        angular_vel_w=torch.zeros((2, 3), dtype=DTYPE),
        angular_acc_w=torch.zeros((2, 3), dtype=DTYPE),
        gravity_w=torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE),
        base_pos_w=torch.zeros(3, dtype=DTYPE),
        base_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
    )
    with torch.no_grad():
        values["masses"][0] = torch.nan
    try:
        recursive_newton_euler_reaction(**values)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite link data must be rejected")


def test_recursive_reaction_rotates_local_com_inertia_to_world_axes():
    half_sqrt_two = 2.0**-0.5
    result = recursive_newton_euler_reaction(
        torch.ones(1, dtype=DTYPE),
        torch.zeros((1, 3), dtype=DTYPE),
        torch.tensor(
            [[half_sqrt_two, 0.0, 0.0, half_sqrt_two]], dtype=DTYPE
        ),
        torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)).unsqueeze(0),
        torch.tensor([[0.0, 0.0, -9.81]], dtype=DTYPE),
        torch.zeros((1, 3), dtype=DTYPE),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE),
        torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE),
        base_pos_w=torch.zeros(3, dtype=DTYPE),
        base_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
    )

    torch.testing.assert_close(
        result, torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=DTYPE)
    )


def test_recursive_terms_expose_force_angular_and_lever_components():
    terms = recursive_newton_euler_terms(
        masses=torch.tensor([2.0], dtype=DTYPE),
        com_pos_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
        inertia_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE),
        inertia_com_local=torch.eye(3, dtype=DTYPE).unsqueeze(0),
        linear_acc_w=torch.zeros((1, 3), dtype=DTYPE),
        angular_vel_w=torch.zeros((1, 3), dtype=DTYPE),
        angular_acc_w=torch.tensor([[0.0, 0.0, 3.0]], dtype=DTYPE),
        gravity_w=torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE),
        base_pos_w=torch.zeros(3, dtype=DTYPE),
    )

    torch.testing.assert_close(terms["required_force_w"], torch.tensor([0.0, 0.0, 19.62], dtype=DTYPE))
    torch.testing.assert_close(terms["angular_momentum_moment_w"], torch.tensor([0.0, 0.0, 3.0], dtype=DTYPE))
    torch.testing.assert_close(terms["lever_arm_moment_w"], torch.tensor([0.0, -19.62, 0.0], dtype=DTYPE))
    torch.testing.assert_close(terms["required_moment_w"], torch.tensor([0.0, -19.62, 3.0], dtype=DTYPE))
