import torch

from go2_pvcnn.tasks.mdp.m1_panda_arm_mpc_residual import (
    ResidualRewardSignals,
    SmallEeTrajectory,
    compute_residual_reward,
    stability_gate,
)


def _signals(**overrides):
    values = dict(
        roll=torch.zeros(2),
        pitch=torch.zeros(2),
        base_height_error=torch.zeros(2),
        support_margin=torch.full((2,), 0.08),
        wheel_contact_count=torch.full((2,), 4.0),
        joint_margin=torch.full((2,), 0.2),
        hard_failure=torch.zeros(2),
        ee_position_error=torch.full((2,), 0.01),
        ee_orientation_error=torch.full((2,), 0.04),
        wrench_error=torch.full((2,), 0.1),
        wheel_slip=torch.zeros(2),
        residual=torch.zeros((2, 8)),
        previous_residual=torch.zeros((2, 8)),
        intervention=torch.zeros(2),
    )
    values.update(overrides)
    return ResidualRewardSignals(**values)


def test_stability_gate_is_bounded_and_closes_on_contact_loss():
    safe = stability_gate(_signals())
    lost = stability_gate(_signals(wheel_contact_count=torch.zeros(2)))

    assert torch.all((safe >= 0.0) & (safe <= 1.0))
    assert torch.all((lost >= 0.0) & (lost <= 1.0))
    assert torch.equal(lost, torch.zeros(2))


def test_task_reward_is_suppressed_before_instability_can_be_profitable():
    safe = compute_residual_reward(_signals())
    unstable = compute_residual_reward(
        _signals(roll=torch.full((2,), 0.4), support_margin=torch.zeros(2))
    )

    assert torch.all(unstable.task < safe.task)
    assert torch.all(unstable.total < safe.total)


def test_hard_failure_dominates_perfect_task_tracking():
    nominal = compute_residual_reward(_signals())
    failed = compute_residual_reward(
        _signals(
            hard_failure=torch.ones(2),
            ee_position_error=torch.zeros(2),
            ee_orientation_error=torch.zeros(2),
        )
    )
    assert torch.all(failed.total < nominal.total - 10.0)


def test_residual_rate_magnitude_and_intervention_are_penalized():
    active = compute_residual_reward(
        _signals(
            residual=torch.ones((2, 8)),
            previous_residual=-torch.ones((2, 8)),
            intervention=torch.ones(2),
        )
    )
    quiet = compute_residual_reward(_signals())
    assert torch.all(active.regularization < quiet.regularization)


def test_small_ee_curriculum_is_seeded_bounded_and_has_zero_base_command():
    center = torch.tensor([0.4, 0.0, 0.55, 0.0, 0.0, 0.0])
    left = SmallEeTrajectory(seed=42, scale=1.0)
    right = SmallEeTrajectory(seed=42, scale=1.0)

    samples_left = torch.stack([left.sample(center, step / 50.0) for step in range(200)])
    samples_right = torch.stack([right.sample(center, step / 50.0) for step in range(200)])
    error = (samples_left - center).abs()
    assert torch.equal(samples_left, samples_right)
    assert error[:, :3].max() <= 0.03
    assert error[:, 3:].max() <= 0.08
    assert torch.equal(left.base_command, torch.zeros(3))
