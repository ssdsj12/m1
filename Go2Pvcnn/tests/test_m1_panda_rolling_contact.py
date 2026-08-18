import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.rolling_contact import (
    RollingContactCfg,
    build_wheel_contact_jacobian,
    rolling_contact_metrics,
    wheel_speed_from_base_velocity,
)


def test_forward_base_velocity_maps_to_four_canonical_wheel_speeds():
    cfg = RollingContactCfg()
    vx = torch.tensor(0.095, dtype=torch.float64)

    target = wheel_speed_from_base_velocity(vx, cfg)

    assert cfg.wheel_radius_m == pytest.approx(0.095)
    assert cfg.wheel_signs == (1.0, 1.0, 1.0, 1.0)
    assert torch.equal(target, torch.ones(4, dtype=torch.float64))


def test_bottom_point_jacobian_encodes_pure_rolling_cancellation():
    cfg = RollingContactCfg()
    body = torch.zeros(4, 6, 31, dtype=torch.float64)
    body[:, 0, 0] = 1.0
    for wheel, column in enumerate((18, 19, 20, 21)):
        body[wheel, 4, column] = 1.0
    contact = build_wheel_contact_jacobian(body, cfg)
    qd = torch.zeros(31, dtype=torch.float64)
    qd[0] = 0.095
    qd[18:22] = 1.0

    metrics = rolling_contact_metrics(contact, qd, yaw=0.0)

    assert contact.shape == (12, 31)
    assert metrics.max_longitudinal_residual_mps == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert metrics.max_lateral_slip_mps == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize("radius", [0.0, -0.1, float("nan"), float("inf")])
def test_rolling_contact_cfg_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="wheel_radius_m must be finite and positive"):
        RollingContactCfg(wheel_radius_m=radius)


@pytest.mark.parametrize(
    "signs",
    [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 2.0),
    ],
)
def test_rolling_contact_cfg_rejects_invalid_wheel_signs(signs):
    with pytest.raises(
        ValueError, match=r"wheel_signs must contain four values in \{-1.0, 1.0\}"
    ):
        RollingContactCfg(wheel_signs=signs)


def test_wheel_mapping_rejects_non_finite_or_non_scalar_velocity():
    cfg = RollingContactCfg()
    with pytest.raises(ValueError, match="vx must contain only finite values"):
        wheel_speed_from_base_velocity(torch.tensor(float("nan")), cfg)
    with pytest.raises(ValueError, match="vx must be one scalar tensor"):
        wheel_speed_from_base_velocity(torch.zeros(2), cfg)


def test_contact_helpers_reject_wrong_shapes():
    cfg = RollingContactCfg()
    with pytest.raises(
        ValueError, match=r"body_jacobians must end with shape \(4, 6, 31\)"
    ):
        build_wheel_contact_jacobian(torch.zeros(4, 6, 30), cfg)
    with pytest.raises(
        ValueError, match=r"contact_jacobian must end with shape \(12, 31\)"
    ):
        rolling_contact_metrics(
            torch.zeros(4, 3, 31), torch.zeros(31), yaw=0.0
        )


def test_metrics_rotate_world_velocity_into_heading_frame():
    contact = torch.zeros(12, 31, dtype=torch.float64)
    contact[1::3, 0] = 1.0
    qd = torch.zeros(31, dtype=torch.float64)
    qd[0] = 0.03

    metrics = rolling_contact_metrics(contact, qd, yaw=torch.pi / 2.0)

    assert metrics.max_longitudinal_residual_mps == pytest.approx(0.03)
    assert metrics.max_lateral_slip_mps == pytest.approx(0.0, abs=1.0e-12)
