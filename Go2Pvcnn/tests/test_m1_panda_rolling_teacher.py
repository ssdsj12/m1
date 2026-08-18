import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import (
    LongitudinalCommandSchedule,
    LongitudinalScheduleCfg,
    PlanarBodyFrameTrajectory,
)
from go2_pvcnn.control.m1_panda_coordination.trajectory import (
    BandLimitedTrajectoryCfg,
)


def test_schedule_has_five_800_step_phases_and_rate_limits_boundaries():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()

    for step in range(800):
        command = schedule.sample(step)

    assert command.phase == 0
    assert command.raw_target_mps == pytest.approx(0.0)
    first_forward = schedule.sample(800)
    assert first_forward.phase == 1
    assert first_forward.raw_target_mps == pytest.approx(0.05)
    assert first_forward.shaped_target_mps == pytest.approx(0.0005)
    for step in range(801, 4000):
        command = schedule.sample(step)
    assert command.phase == 4
    assert command.raw_target_mps == pytest.approx(-0.05)


def test_hold_scale_requests_rate_limited_stop_instead_of_locking_wheels():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()
    for step in range(1000):
        command = schedule.sample(step)

    stopped = schedule.sample(1000, safety_scale=0.0)

    assert 0.0 < stopped.shaped_target_mps < command.shaped_target_mps
    assert command.shaped_target_mps - stopped.shaped_target_mps == pytest.approx(
        0.0005
    )


def test_schedule_rejects_skipped_or_repeated_steps():
    schedule = LongitudinalCommandSchedule()
    schedule.sample(0)
    with pytest.raises(ValueError, match="mission_step must advance exactly once"):
        schedule.sample(0)
    with pytest.raises(ValueError, match="mission_step must advance exactly once"):
        schedule.sample(2)


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_schedule_rejects_invalid_safety_scale(scale):
    schedule = LongitudinalCommandSchedule()
    with pytest.raises(ValueError, match="safety_scale must be finite and in"):
        schedule.sample(0, safety_scale=scale)


def test_schedule_reset_replays_the_same_commands():
    schedule = LongitudinalCommandSchedule()
    first = [schedule.sample(step) for step in range(900)]
    schedule.reset()
    second = [schedule.sample(step) for step in range(900)]
    assert first == second


def test_body_frame_center_advects_with_root_without_arm_extension():
    trajectory = PlanarBodyFrameTrajectory(
        BandLimitedTrajectoryCfg(
            position_amplitude=0.0,
            orientation_amplitude=0.0,
        )
    )
    ee = torch.tensor(
        [1.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    root = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    trajectory.reset(ee, root, seed=42)

    moved = trajectory.sample(
        1.0,
        torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
    )

    assert moved.pose[0].item() == pytest.approx(1.2)
    assert moved.pose[1].item() == pytest.approx(0.0)
    assert moved.twist[0].item() == pytest.approx(0.1)


def test_body_frame_center_rotates_with_heading_and_includes_yaw_twist():
    trajectory = PlanarBodyFrameTrajectory(
        BandLimitedTrajectoryCfg(
            position_amplitude=0.0,
            orientation_amplitude=0.0,
        )
    )
    trajectory.reset(
        torch.tensor([1.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        seed=7,
    )

    sample = trajectory.sample(
        0.0,
        torch.tensor([0.0, 0.0, torch.pi / 2.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64),
    )

    assert sample.pose[0].item() == pytest.approx(0.0, abs=1.0e-12)
    assert sample.pose[1].item() == pytest.approx(1.0)
    assert sample.pose[5].item() == pytest.approx(torch.pi / 2.0)
    assert sample.twist[0].item() == pytest.approx(-0.2)
    assert sample.twist[1].item() == pytest.approx(0.0, abs=1.0e-12)
    assert sample.twist[5].item() == pytest.approx(0.2)


def test_body_frame_trajectory_is_seed_repeatable_after_reset():
    cfg = BandLimitedTrajectoryCfg(
        position_amplitude=0.005,
        orientation_amplitude=0.01,
    )
    trajectory = PlanarBodyFrameTrajectory(cfg)
    ee = torch.tensor(
        [0.4, -0.1, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    root = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    root_velocity = torch.zeros(3, dtype=torch.float64)

    trajectory.reset(ee, root, seed=21)
    first = trajectory.sample(1.25, root, root_velocity)
    trajectory.reset(ee, root, seed=21)
    second = trajectory.sample(1.25, root, root_velocity)

    assert torch.equal(first.pose, second.pose)
    assert torch.equal(first.twist, second.twist)
    assert torch.equal(first.acceleration, second.acceleration)
