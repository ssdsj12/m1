from dataclasses import replace

import pytest
import torch

import go2_pvcnn.control.m1_panda_coordination.rolling_wbc as rolling_wbc_module
from go2_pvcnn.control.m1_panda_coordination.rolling_wbc import (
    RollingWbcCfg,
    build_rolling_wbc_problem,
    solve_rolling_wbc,
)
from go2_pvcnn.control.m1_panda_coordination.standing_wbc import StandingWbcInput


def _selector(rows, dimension=31):
    result = torch.zeros(len(rows), dimension, dtype=torch.float64)
    for output, source in enumerate(rows):
        result[output, source] = 1.0
    return result


def _input():
    return StandingWbcInput(
        mass_matrix=torch.eye(31, dtype=torch.float64),
        bias_force=torch.zeros(31, dtype=torch.float64),
        contact_jacobian=torch.zeros(12, 31, dtype=torch.float64),
        contact_jacobian_dot_qd=torch.zeros(12, dtype=torch.float64),
        mount_wrench_jacobian=torch.zeros(6, 31, dtype=torch.float64),
        external_wrench=torch.zeros(6, dtype=torch.float64),
        balance_jacobian=_selector((2, 3, 4)),
        balance_acceleration=torch.zeros(3, dtype=torch.float64),
        base_jacobian=_selector(tuple(range(6))),
        base_acceleration=torch.zeros(6, dtype=torch.float64),
        leg_generalized_indices=torch.arange(6, 18, dtype=torch.long),
        wheel_generalized_indices=torch.arange(18, 22, dtype=torch.long),
        arm_generalized_indices=torch.arange(22, 29, dtype=torch.long),
        leg_acceleration=torch.zeros(12, dtype=torch.float64),
        wheel_acceleration=torch.zeros(4, dtype=torch.float64),
        arm_acceleration=torch.zeros(7, dtype=torch.float64),
        qdd_lower=torch.full((31,), -10.0, dtype=torch.float64),
        qdd_upper=torch.full((31,), 10.0, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        friction_coefficient=0.7,
    )


def test_rolling_wbc_keeps_balance_above_velocity_and_arm_tracking():
    cfg = RollingWbcCfg()

    assert cfg.balance_weight > cfg.base_velocity_weight
    assert cfg.base_velocity_weight > cfg.wheel_tracking_weight
    assert cfg.wheel_tracking_weight > cfg.leg_posture_weight
    assert cfg.leg_posture_weight > cfg.arm_tracking_weight


def test_rolling_wbc_carries_nonzero_base_and_wheel_targets_into_qp():
    state = _input()
    base = state.base_acceleration.clone()
    base[0] = 0.5
    wheels = torch.full((4,), 2.0, dtype=torch.float64)

    assembled = build_rolling_wbc_problem(
        replace(
            state,
            base_acceleration=base,
            wheel_acceleration=wheels,
        )
    )

    assert assembled.task_targets["base"][0].item() == pytest.approx(0.5)
    assert torch.equal(assembled.task_targets["wheels"], wheels)
    assert assembled.qp.equality_matrix.shape == (18, 43)
    assert torch.equal(
        assembled.qp.equality_matrix[6:, :31], state.contact_jacobian
    )


def test_rolling_wbc_maps_every_weight_to_the_shared_problem():
    cfg = RollingWbcCfg(
        balance_weight=101.0,
        base_velocity_weight=53.0,
        wheel_tracking_weight=31.0,
        leg_posture_weight=17.0,
        arm_tracking_weight=11.0,
        force_equalization_weight=7.0,
        tangential_force_weight=5.0,
        regularization=3.0e-6,
        qp_tolerance=2.0e-7,
    )

    mapped = cfg.standing_cfg()

    assert mapped.balance_weight == pytest.approx(101.0)
    assert mapped.base_pose_weight == pytest.approx(53.0)
    assert mapped.wheel_stop_weight == pytest.approx(31.0)
    assert mapped.leg_posture_weight == pytest.approx(17.0)
    assert mapped.arm_tracking_weight == pytest.approx(11.0)
    assert mapped.force_equalization_weight == pytest.approx(7.0)
    assert mapped.tangential_force_weight == pytest.approx(5.0)
    assert mapped.regularization == pytest.approx(3.0e-6)
    assert mapped.qp_tolerance == pytest.approx(2.0e-7)


def test_rolling_solver_forwards_qp_tolerance(monkeypatch):
    captured = {}

    def fake_solver(state, cfg):
        captured["state"] = state
        captured["cfg"] = cfg
        return "result"

    monkeypatch.setattr(rolling_wbc_module, "solve_standing_wbc", fake_solver)

    result = solve_rolling_wbc(_input(), RollingWbcCfg(qp_tolerance=4.0e-8))

    assert result == "result"
    assert captured["state"] is not None
    assert captured["cfg"].qp_tolerance == pytest.approx(4.0e-8)
