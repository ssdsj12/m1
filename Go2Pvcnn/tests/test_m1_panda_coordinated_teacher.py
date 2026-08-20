import torch

from go2_pvcnn.control.m1_panda_coordination.coordinated_mission import (
    CoordinatedMission,
    CoordinatedMissionCfg,
    MissionPhase,
)
from go2_pvcnn.control.m1_panda_coordination.coordinated_teacher import (
    CoordinatedTeacherAdapter,
)


def test_adapter_keeps_base_still_until_tracking_then_assists():
    adapter = CoordinatedTeacherAdapter(
        mission=CoordinatedMission(
            CoordinatedMissionCfg(
                physics_dt=0.1,
                settled_steps=1,
                unfold_duration_s=0.1,
                folded_arm_target=torch.zeros(7),
            )
        )
    )
    zero = torch.zeros(3)
    adapter.reset(zero, torch.zeros(6), torch.tensor([0.1, 0.0, 0.0]), torch.ones(6), seed=1)
    first = adapter.step(zero, torch.zeros(6), torch.ones(6), torch.zeros(6), torch.tensor(0.1), torch.tensor(0.2))
    assert first.phase is MissionPhase.FOLD_AND_NAVIGATE
    assert not first.base_assist_active
    adapter.step(torch.tensor([0.1, 0.0, 0.0]), torch.zeros(6), torch.ones(6), torch.zeros(6), torch.tensor(0.1), torch.tensor(0.3))
    tracking = adapter.step(torch.tensor([0.1, 0.0, 0.0]), torch.ones(6), torch.ones(6), torch.zeros(6), torch.tensor(0.05), torch.tensor(0.3))
    assert tracking.phase in (MissionPhase.UNFOLD_AND_TRACK, MissionPhase.COORDINATED_TRACK)
    tracking = adapter.step(torch.tensor([0.1, 0.0, 0.0]), torch.ones(6), torch.ones(6), torch.zeros(6), torch.tensor(0.05), torch.tensor(0.3))
    assert tracking.phase is MissionPhase.COORDINATED_TRACK
    assert tracking.base_assist_active
    assert tracking.base_velocity.shape == (3,)


def test_adapter_returns_safe_zero_assistance_for_invalid_sigma():
    adapter = CoordinatedTeacherAdapter()
    zero = torch.zeros(3)
    adapter.reset(zero, torch.zeros(6), zero, torch.zeros(6), seed=0)
    decision = adapter.step(zero, torch.zeros(6), torch.zeros(6), torch.zeros(6), torch.tensor(0.0), torch.tensor(float("nan")))
    assert not decision.base_assist_active
    assert torch.equal(decision.base_velocity, zero)
