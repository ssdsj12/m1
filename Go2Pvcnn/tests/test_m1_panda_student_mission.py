import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.student_mission import (
    StudentS1Mission,
)


def _reset(mission, *, seed=42):
    controlled_q = torch.arange(23, dtype=torch.float64) * 0.01
    center_pose = torch.tensor(
        [0.4, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    root = torch.zeros(3, dtype=torch.float64)
    mission.reset(
        center_pose,
        root,
        settled_controlled_q=controlled_q,
        seed=seed,
    )
    return controlled_q, root


def test_student_mission_has_five_phases_and_deployable_nominal_commands():
    mission = StudentS1Mission()
    settled_q, root = _reset(mission)
    root_velocity = torch.zeros(3, dtype=torch.float64)
    samples = []
    for step in range(4000):
        sample = mission.sample(step, root, root_velocity)
        if step in (0, 799, 800, 1599, 1600, 2399, 2400, 3199, 3200, 3999):
            samples.append((step, sample))

    assert [sample.phase for _, sample in samples] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    final = samples[-1][1]
    assert final.shaped_vx == pytest.approx(-0.05)
    assert final.nominal.position.shape == (1, 23)
    assert final.nominal.velocity.shape == (1, 23)
    torch.testing.assert_close(final.nominal.position[0, :12], settled_q[:12])
    torch.testing.assert_close(final.nominal.position[0, 16:], settled_q[16:])
    torch.testing.assert_close(
        final.nominal.velocity[0, 12:16],
        torch.full((4,), final.shaped_vx / 0.095, dtype=torch.float64),
    )


def test_student_mission_reset_is_seed_repeatable_and_instances_are_isolated():
    first = StudentS1Mission()
    second = StudentS1Mission()
    _, root = _reset(first, seed=7)
    _reset(second, seed=7)
    velocity = torch.zeros(3, dtype=torch.float64)

    first_samples = [first.sample(step, root, velocity) for step in range(12)]
    second_samples = [second.sample(step, root, velocity) for step in range(12)]
    for left, right in zip(first_samples, second_samples, strict=True):
        assert left.phase == right.phase
        assert left.shaped_vx == right.shaped_vx
        torch.testing.assert_close(left.target_pose, right.target_pose)
        torch.testing.assert_close(left.target_twist, right.target_twist)

    _reset(first, seed=11)
    restarted = first.sample(0, root, velocity)
    untouched = second.sample(12, root, velocity)
    assert restarted.phase == 0
    assert untouched.phase == 0
    assert not torch.equal(restarted.target_pose, untouched.target_pose)


def test_student_mission_rejects_wrong_reset_and_skipped_steps():
    mission = StudentS1Mission()
    _, root = _reset(mission)
    velocity = torch.zeros(3, dtype=torch.float64)
    mission.sample(0, root, velocity)
    with pytest.raises(ValueError, match="advance exactly once"):
        mission.sample(2, root, velocity)
    with pytest.raises(ValueError, match="shape"):
        mission.reset(
            torch.zeros(6, dtype=torch.float64),
            root,
            settled_controlled_q=torch.zeros(22, dtype=torch.float64),
            seed=1,
        )
