from __future__ import annotations

from dataclasses import replace

import torch

from go2_pvcnn.control.m1_bimanual_coordination import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import (
    SafetyInput,
    SafetyProjection,
)
from tests.test_m1_bimanual_full_action_teacher import _dynamics


DTYPE = torch.float64


def _input() -> SafetyInput:
    return SafetyInput(
        candidate_effort=torch.zeros(43, dtype=DTYPE),
        safe_effort=torch.zeros(43, dtype=DTYPE),
        dynamics=_dynamics(),
        active_generalized_ids=torch.arange(6, 49),
        active_q=torch.zeros(43, dtype=DTYPE),
        active_qd=torch.zeros(43, dtype=DTYPE),
        q_min=-2.0 * torch.ones(43, dtype=DTYPE),
        q_max=2.0 * torch.ones(43, dtype=DTYPE),
        qd_max=5.0 * torch.ones(43, dtype=DTYPE),
        effort_limits=100.0 * torch.ones(43, dtype=DTYPE),
        collision_distances=torch.ones(1, dtype=DTYPE),
        collision_jacobian=torch.zeros(1, 43, dtype=DTYPE),
        base_error=torch.zeros(6, dtype=DTYPE),
        force_closure_margin=1.0,
        phase=BimanualPhase.APPROACH,
    )


def test_projection_locks_wheels_and_respects_effort_limits() -> None:
    sample = replace(
        _input(), candidate_effort=1.0e6 * torch.ones(43, dtype=DTYPE)
    )

    result = SafetyProjection().project(sample)

    assert result.feasible
    assert torch.all(result.effort.abs() <= sample.effort_limits + 1.0e-10)
    assert torch.all(result.effort[12:16] == 0.0)
    assert "wheel_lock" in result.active_constraints


def test_projection_rejects_nonfinite_without_returning_candidate() -> None:
    candidate = torch.zeros(43, dtype=DTYPE)
    candidate[0] = torch.nan

    result = SafetyProjection().project(
        replace(_input(), candidate_effort=candidate)
    )

    assert not result.feasible
    assert result.fallback_reason == "nonfinite_candidate"
    assert torch.isfinite(result.effort).all()
    assert torch.all(result.effort[12:16] == 0.0)


def test_projection_enforces_one_step_joint_position_barrier() -> None:
    sample = _input()
    q = sample.active_q.clone()
    q[6] = sample.q_max[6] - 1.0e-8
    candidate = sample.candidate_effort.clone()
    candidate[6] = 100.0

    result = SafetyProjection().project(
        replace(sample, active_q=q, candidate_effort=candidate)
    )

    assert result.feasible
    assert result.effort[6] < candidate[6]


def test_force_closure_is_hard_only_after_grasp() -> None:
    lost = replace(_input(), force_closure_margin=-0.1)

    approach = SafetyProjection().project(lost)
    grasp = SafetyProjection().project(replace(lost, phase=BimanualPhase.GRASP))

    assert approach.feasible
    assert not grasp.feasible
    assert grasp.fallback_reason == "force_closure_lost"
