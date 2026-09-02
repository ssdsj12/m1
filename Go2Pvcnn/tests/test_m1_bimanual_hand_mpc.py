from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import (
    HandMpcCfg,
    HandMpcInput,
    O6HandMpc,
)


DTYPE = torch.float64


def _input(**overrides) -> HandMpcInput:
    forces = torch.zeros(5, 3, dtype=DTYPE)
    forces[:, 2] = 1.0
    jacobian = torch.zeros(15, 6, dtype=DTYPE)
    for fingertip in range(5):
        jacobian[3 * fingertip + 2, fingertip] = 1.0
    jacobian[0, 5] = 0.2
    wrench_map = torch.zeros(6, 15, dtype=DTYPE)
    for fingertip in range(5):
        wrench_map[2, 3 * fingertip + 2] = 1.0
    values = {
        "q": 0.1 * torch.ones(6, dtype=DTYPE),
        "qd": torch.zeros(6, dtype=DTYPE),
        "fingertip_forces_b": forces,
        "contact_mask": torch.ones(5, dtype=torch.bool),
        "contact_jacobian": jacobian,
        "wrench_map": wrench_map,
        "target_wrench_b": torch.tensor([0.0, 0.0, 7.0, 0.0, 0.0, 0.0], dtype=DTYPE),
        "q_min": torch.zeros(6, dtype=DTYPE),
        "q_max": torch.ones(6, dtype=DTYPE),
        "qd_max": torch.ones(6, dtype=DTYPE),
    }
    values.update(overrides)
    return HandMpcInput(**values)


def test_default_cfg_freezes_100_hz_twenty_node_contract():
    cfg = HandMpcCfg()
    assert cfg.dt == pytest.approx(0.01)
    assert cfg.horizon_steps == 20
    assert cfg.horizon_seconds == pytest.approx(0.2)
    assert cfg.normal_force_max == pytest.approx(10.0)
    assert cfg.friction_coefficient == pytest.approx(0.8)


def test_hand_mpc_tracks_wrench_with_six_active_axes_only():
    sample = _input()
    solution = O6HandMpc().plan(sample)

    assert solution.diagnostics.feasible
    assert solution.q_ref.shape == (6,)
    assert solution.qd_ref.shape == (6,)
    assert solution.predicted_forces_b.shape == (5, 3)
    assert solution.predicted_wrench_b.shape == (6,)
    assert torch.all(solution.qd_ref.abs() <= sample.qd_max + 1.0e-12)
    assert solution.predicted_wrench_b[2].item() > 5.0
    assert solution.diagnostics.wrench_error_norm < 2.0


def test_mimic_joint_shape_is_rejected_from_active_input():
    with pytest.raises(ValueError, match="six active"):
        _input(q=torch.zeros(11, dtype=DTYPE))


def test_contact_forces_obey_normal_and_friction_limits():
    planner = O6HandMpc()
    solution = planner.plan(_input())
    force = solution.predicted_forces_b
    normal = force[:, 2]
    assert torch.all(normal >= -1.0e-8)
    assert torch.all(normal <= planner.cfg.normal_force_max + 1.0e-8)
    assert torch.all(force[:, 0].abs() <= planner.cfg.friction_coefficient * normal + 1.0e-8)
    assert torch.all(force[:, 1].abs() <= planner.cfg.friction_coefficient * normal + 1.0e-8)
    assert solution.diagnostics.slip_margin >= -1.0e-8


def test_inactive_fingertips_predict_zero_force():
    mask = torch.tensor([True, False, True, False, True])
    solution = O6HandMpc().plan(_input(contact_mask=mask))
    assert solution.diagnostics.feasible
    assert torch.count_nonzero(solution.predicted_forces_b[~mask]) == 0


def test_infeasible_joint_recovery_holds_last_safe_reference():
    planner = O6HandMpc()
    safe = planner.plan(_input())
    impossible = _input(q=5.0 * torch.ones(6, dtype=DTYPE))
    fallback = planner.plan(impossible)

    assert fallback.diagnostics.fallback_used
    assert not fallback.diagnostics.feasible
    assert fallback.diagnostics.fallback_reason == "qp_infeasible"
    assert torch.equal(fallback.q_ref, safe.q_ref)
    assert torch.equal(fallback.predicted_forces_b, safe.predicted_forces_b)


def test_input_requires_exact_cpu_float64_bool_and_positive_limits():
    with pytest.raises(TypeError, match="float64"):
        _input(q=torch.zeros(6, dtype=torch.float32))
    with pytest.raises(TypeError, match="bool"):
        _input(contact_mask=torch.ones(5, dtype=DTYPE))
    with pytest.raises(ValueError, match="positive"):
        _input(qd_max=torch.zeros(6, dtype=DTYPE))
    matrix = torch.zeros(6, 15, dtype=DTYPE)
    matrix[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _input(wrench_map=matrix)


def test_repeated_plans_are_deterministic_and_inputs_are_caller_isolated():
    source = torch.zeros(6, dtype=DTYPE)
    sample = _input(qd=source)
    source.add_(10.0)
    assert torch.count_nonzero(sample.qd) == 0
    left = O6HandMpc().plan(sample)
    right = O6HandMpc().plan(sample)
    assert torch.equal(left.q_ref, right.q_ref)
    assert torch.equal(left.predicted_forces_b, right.predicted_forces_b)
    assert left.diagnostics == right.diagnostics
