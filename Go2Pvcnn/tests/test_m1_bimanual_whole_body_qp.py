from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import DualArmMpcSolution
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
from go2_pvcnn.control.m1_bimanual_coordination.object_mpc import BimanualObjectMpc
from go2_pvcnn.control.m1_bimanual_coordination.whole_body_qp import (
    BimanualWbcRequest,
    BimanualWholeBodyQp,
    M1_STANDING_Q,
    _nominal_effort,
    build_bimanual_constraints,
)
from tests.test_m1_bimanual_dual_arm_mpc import _arm_input, _solution as _arm_solution
from tests.test_m1_bimanual_hand_mpc import _input as _hand_input
from tests.test_m1_bimanual_object_mpc import _input as _object_input


DTYPE = torch.float64


def _request(**overrides) -> BimanualWbcRequest:
    object_sample = _object_input()
    object_solution = BimanualObjectMpc().plan(object_sample)
    left_arm_input = _arm_input()
    right_arm_input = _arm_input()
    arm_solution = DualArmMpcSolution(
        left=_arm_solution(left_arm_input, True),
        right=_arm_solution(right_arm_input, True),
        both_feasible=True,
        synchronized_fallback=False,
    )
    left_hand = O6HandMpc().plan(_hand_input())
    right_hand = O6HandMpc().plan(_hand_input())
    values = {
        "snapshot": object_sample.snapshot,
        "object_solution": object_solution,
        "arm_solution": arm_solution,
        "left_hand_solution": left_hand,
        "right_hand_solution": right_hand,
        "collision_distances": torch.tensor([0.10, 0.12], dtype=DTYPE),
        "collision_jacobian": torch.zeros(2, 43, dtype=DTYPE),
    }
    values.update(overrides)
    return BimanualWbcRequest(**values)


def test_wbc_returns_exact_43_efforts_and_zero_wheels_for_stationary_task():
    solution = BimanualWholeBodyQp().solve(_request())
    assert solution.feasible
    assert not solution.fallback_used
    assert solution.effort.shape == (43,)
    assert torch.allclose(solution.effort[12:16], torch.zeros(4, dtype=DTYPE))
    assert torch.isfinite(solution.effort).all()


def test_one_invalid_subsolution_rejects_the_entire_new_command():
    controller = BimanualWholeBodyQp()
    safe = controller.solve(_request())
    bad_hand = replace(
        _request().left_hand_solution,
        diagnostics=replace(
            _request().left_hand_solution.diagnostics,
            feasible=False,
            fallback_used=True,
            fallback_reason="hand_failed",
        ),
    )
    bad = controller.solve(_request(left_hand_solution=bad_hand))
    assert bad.fallback_used
    assert not bad.feasible
    assert torch.equal(bad.effort, safe.effort)
    assert bad.diagnostics.fallback_reason == "left_hand_mpc_infeasible"


def test_collision_linearization_is_a_hard_constraint():
    jacobian = torch.zeros(1, 43, dtype=DTYPE)
    jacobian[0, 0] = 1.0
    request = _request(
        collision_distances=torch.tensor([0.019], dtype=DTYPE),
        collision_jacobian=jacobian,
    )
    constraints = build_bimanual_constraints(request)
    solution = BimanualWholeBodyQp().solve(request)
    assert solution.feasible
    assert torch.all(
        constraints.inequality_matrix @ solution.effort
        <= constraints.inequality_upper + 1.0e-7
    )
    assert solution.effort[0] > 0.0


def test_unrecoverable_collision_uses_last_safe_command():
    controller = BimanualWholeBodyQp()
    safe = controller.solve(_request())
    bad = controller.solve(
        _request(
            collision_distances=torch.tensor([0.0], dtype=DTYPE),
            collision_jacobian=torch.zeros(1, 43, dtype=DTYPE),
        )
    )
    assert bad.fallback_used
    assert torch.equal(bad.effort, safe.effort)
    assert bad.diagnostics.fallback_reason == "qp_infeasible"


def test_constraint_set_covers_effort_collision_and_force_closure():
    constraints = build_bimanual_constraints(_request())
    assert constraints.lower_effort.shape == constraints.upper_effort.shape == (43,)
    assert torch.all(constraints.lower_effort < constraints.upper_effort)
    assert constraints.inequality_matrix.shape[1] == 43
    assert constraints.min_collision_distance == pytest.approx(0.10)
    assert constraints.force_closure_margin > 0.0


def test_approach_zero_wrench_does_not_require_force_closure():
    request = _request()
    approach_object = replace(
        request.object_solution,
        left_wrench=torch.zeros_like(request.object_solution.left_wrench),
        right_wrench=torch.zeros_like(request.object_solution.right_wrench),
        diagnostics=replace(request.object_solution.diagnostics, force_closure_margin=0.0),
    )
    solution = BimanualWholeBodyQp().solve(
        _request(object_solution=approach_object)
    )
    assert solution.feasible


def test_nominal_effort_actively_holds_m1_standing_posture():
    request = _request()
    displaced = replace(
        request.snapshot,
        m1_q=M1_STANDING_Q + 0.1,
        m1_qd=torch.full((16,), 0.2, dtype=DTYPE),
    )
    effort = _nominal_effort(_request(snapshot=displaced), BimanualWholeBodyQp().cfg)
    assert torch.all(effort[:12] < 0.0)
    assert torch.allclose(effort[12:16], torch.full((4,), -6.0, dtype=DTYPE))


def test_request_rejects_nonfinite_or_wrong_collision_contract():
    with pytest.raises(ValueError, match="finite"):
        _request(collision_distances=torch.tensor([float("nan")], dtype=DTYPE), collision_jacobian=torch.zeros(1, 43, dtype=DTYPE))
    with pytest.raises(ValueError, match="shape"):
        _request(collision_distances=torch.ones(2, dtype=DTYPE), collision_jacobian=torch.zeros(2, 42, dtype=DTYPE))
