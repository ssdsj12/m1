import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.base_assist import (
    BaseAssistCfg,
    compute_base_assist,
)


def test_base_assist_is_bounded_and_improves_when_margin_is_low():
    cfg = BaseAssistCfg(
        max_speed_xy=0.2,
        max_yaw_rate=0.3,
        max_accel_xy=0.1,
        max_yaw_accel=0.2,
        max_displacement_xy=0.5,
        enable_margin=0.2,
        disable_margin=0.3,
    )
    decision = compute_base_assist(
        base_pose=torch.zeros(3),
        arrived_base_pose=torch.zeros(3),
        target_base_pose=torch.tensor([1.0, 0.0, 0.0]),
        arm_margin_before=torch.tensor(0.1),
        arm_margin_after=torch.tensor(0.4),
        sigma_min=torch.tensor(0.05),
        previous_velocity=torch.zeros(3),
        dt=0.1,
        cfg=cfg,
    )
    assert decision.active
    assert torch.linalg.vector_norm(decision.base_velocity[:2]) <= 0.01 + 1e-6
    assert decision.base_velocity[2].abs() <= 0.02 + 1e-6
    assert decision.arm_margin_after > decision.arm_margin_before


def test_base_assist_hysteresis_and_nonfinite_fallback():
    cfg = BaseAssistCfg(enable_margin=0.2, disable_margin=0.3)
    common = dict(
        base_pose=torch.zeros(3),
        arrived_base_pose=torch.zeros(3),
        target_base_pose=torch.ones(3),
        arm_margin_before=torch.tensor(0.25),
        arm_margin_after=torch.tensor(0.2),
        sigma_min=torch.tensor(0.2),
        previous_velocity=torch.zeros(3),
        dt=0.01,
        cfg=cfg,
    )
    inactive = compute_base_assist(**common)
    assert not inactive.active
    invalid = compute_base_assist(**{**common, "sigma_min": torch.tensor(float("nan"))})
    assert not invalid.active
    assert torch.equal(invalid.base_velocity, torch.zeros(3))


def test_base_assist_rejects_invalid_dt():
    with pytest.raises(ValueError, match="dt"):
        compute_base_assist(
            base_pose=torch.zeros(3),
            arrived_base_pose=torch.zeros(3),
            target_base_pose=torch.zeros(3),
            arm_margin_before=torch.tensor(0.0),
            arm_margin_after=torch.tensor(0.0),
            sigma_min=torch.tensor(0.1),
            previous_velocity=torch.zeros(3),
            dt=0.0,
        )
