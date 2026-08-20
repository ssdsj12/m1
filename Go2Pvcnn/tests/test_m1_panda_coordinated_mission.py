import math

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.coordinated_mission import (
    CoordinatedMission,
    CoordinatedMissionCfg,
    MissionPhase,
)


def _mission(settled_steps=2):
    return CoordinatedMission(
        CoordinatedMissionCfg(
            physics_dt=0.1,
            settled_steps=settled_steps,
            unfold_duration_s=0.2,
            folded_arm_target=torch.zeros(7),
        )
    )


def test_mission_progresses_fold_arrive_unfold_and_track():
    mission = _mission()
    mission.reset(
        base_pose=torch.zeros(3),
        ee_pose=torch.zeros(6),
        target_base_pose=torch.tensor([1.0, 0.0, 0.2]),
        ee_target_pose=torch.ones(6),
        seed=7,
    )

    first = mission.step(
        base_pose=torch.zeros(3),
        ee_pose=torch.zeros(6),
        ee_target_pose=torch.ones(6),
        ee_target_twist=torch.zeros(6),
    )
    assert first.phase is MissionPhase.FOLD_AND_NAVIGATE
    assert torch.equal(first.arm_target, torch.zeros(7))

    arrived = None
    for _ in range(2):
        arrived = mission.step(
            base_pose=torch.tensor([1.0, 0.0, 0.2]),
            ee_pose=torch.zeros(6),
            ee_target_pose=torch.ones(6),
            ee_target_twist=torch.zeros(6),
        )
    assert arrived is not None
    assert arrived.phase is MissionPhase.ARRIVE_HOLD

    transitioning = mission.step(
        base_pose=torch.tensor([1.0, 0.0, 0.2]),
        ee_pose=torch.zeros(6),
        ee_target_pose=torch.ones(6),
        ee_target_twist=torch.zeros(6),
    )
    assert transitioning.phase is MissionPhase.UNFOLD_AND_TRACK
    assert transitioning.arm_target.shape == (7,)

    tracked = transitioning
    for _ in range(2):
        tracked = mission.step(
            base_pose=torch.tensor([1.0, 0.0, 0.2]),
            ee_pose=torch.ones(6),
            ee_target_pose=torch.ones(6),
            ee_target_twist=torch.zeros(6),
        )
    assert tracked.phase is MissionPhase.COORDINATED_TRACK


def test_mission_reset_isolated_and_rejects_nonfinite_targets():
    mission = _mission(settled_steps=1)
    with pytest.raises(ValueError, match="finite"):
        mission.reset(
            base_pose=torch.zeros(3),
            ee_pose=torch.zeros(6),
            target_base_pose=torch.tensor([math.nan, 0.0, 0.0]),
            ee_target_pose=torch.zeros(6),
            seed=0,
        )

    mission.reset(
        base_pose=torch.zeros(3),
        ee_pose=torch.zeros(6),
        target_base_pose=torch.tensor([0.5, 0.0, 0.0]),
        ee_target_pose=torch.zeros(6),
        seed=3,
    )
    state = mission.step(
        base_pose=torch.zeros(3),
        ee_pose=torch.zeros(6),
        ee_target_pose=torch.zeros(6),
        ee_target_twist=torch.zeros(6),
    )
    assert state.step == 0
    assert state.phase is MissionPhase.FOLD_AND_NAVIGATE
