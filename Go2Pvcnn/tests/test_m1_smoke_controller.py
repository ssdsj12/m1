import math

import pytest
import torch


def test_rolling_action_keeps_legs_neutral_and_drives_wheels():
    from go2_pvcnn.tasks.m1_smoke_controller import M1SmokeControllerCfg, build_m1_smoke_action

    cfg = M1SmokeControllerCfg(rolling_wheel_velocity=0.5, wheel_action_scale=8.0)

    action = build_m1_smoke_action(num_envs=2, time_s=0.0, mode="rolling", cfg=cfg, device="cpu")

    assert action.shape == (2, 16)
    assert torch.allclose(action[:, :12], torch.zeros(2, 12))
    assert torch.allclose(action[:, 12:], torch.full((2, 4), 0.5 / 8.0))


def test_wave_action_keeps_legs_locked_by_default_and_drives_equal_wheels():
    from go2_pvcnn.tasks.m1_smoke_controller import M1SmokeControllerCfg, build_m1_smoke_action

    cfg = M1SmokeControllerCfg(
        leg_action_scale=0.25,
        wheel_action_scale=8.0,
        wave_wheel_velocity=1.5,
        wave_amplitude=0.0,
        wave_knee_ratio=1.5,
        wave_frequency=1.0,
        wave_phase_offsets=(0.0, 0.5, 0.5, 0.0),
    )

    action = build_m1_smoke_action(num_envs=1, time_s=0.25, mode="wave", cfg=cfg, device="cpu")

    assert action.shape == (1, 16)
    assert torch.allclose(action[:, :12], torch.zeros(1, 12))
    assert torch.allclose(action[:, 12:], torch.full((1, 4), 1.5 / 8.0))


def test_controller_rejects_unknown_mode():
    from go2_pvcnn.tasks.m1_smoke_controller import build_m1_smoke_action

    with pytest.raises(ValueError, match="Unsupported M1 smoke control mode"):
        build_m1_smoke_action(num_envs=1, time_s=0.0, mode="hop", device="cpu")


def test_wave_signal_is_periodic():
    from go2_pvcnn.tasks.m1_smoke_controller import M1SmokeControllerCfg, _positive_sine

    cfg = M1SmokeControllerCfg(wave_frequency=1.0)

    assert _positive_sine(0.25, 0.0, cfg) == pytest.approx(1.0, abs=1e-6)
    assert _positive_sine(0.75, 0.0, cfg) == pytest.approx(0.0, abs=1e-6)
    assert _positive_sine(1.25, 0.0, cfg) == pytest.approx(1.0, abs=1e-6)
    assert _positive_sine(0.25, 0.5, cfg) == pytest.approx(0.0, abs=1e-6)
