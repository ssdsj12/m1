from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import (
    SafetyInput,
    SafetyProjection,
)
from go2_pvcnn.control.m1_bimanual_coordination.reduced_dynamics import (
    condense_constrained_dynamics,
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


def test_projection_keeps_a_predictive_joint_limit_margin() -> None:
    sample = _input()
    q = sample.active_q.clone()
    q[6] = sample.q_min[6] + 0.025
    candidate = sample.candidate_effort.clone()
    candidate[6] = -100.0
    projection = SafetyProjection(joint_position_margin_rad=0.02)

    result = projection.project(
        replace(sample, active_q=q, candidate_effort=candidate)
    )

    reduced = condense_constrained_dynamics(sample.dynamics)
    ids = sample.active_generalized_ids
    predicted_q = (
        q
        + projection.dt * sample.active_qd
        + 0.5
        * projection.dt**2
        * (reduced.qdd_offset[ids] + reduced.qdd_from_effort[ids] @ result.effort)
    )

    assert result.feasible
    assert predicted_q[6] >= sample.q_min[6] + 0.02 - 1.0e-9


def test_projection_keeps_a_predictive_velocity_margin() -> None:
    sample = _input()
    qd = sample.active_qd.clone()
    qd[6] = 4.49
    candidate = sample.candidate_effort.clone()
    candidate[6] = 100.0
    projection = SafetyProjection(joint_velocity_margin_fraction=0.1)

    result = projection.project(
        replace(sample, active_qd=qd, candidate_effort=candidate)
    )

    reduced = condense_constrained_dynamics(sample.dynamics)
    ids = sample.active_generalized_ids
    predicted_qd = qd + projection.dt * (
        reduced.qdd_offset[ids] + reduced.qdd_from_effort[ids] @ result.effort
    )
    assert result.feasible
    assert predicted_qd[6] <= 0.9 * sample.qd_max[6] + 1.0e-9


def test_force_closure_is_hard_only_after_grasp() -> None:
    lost = replace(_input(), force_closure_margin=-0.1)

    approach = SafetyProjection().project(lost)
    grasp = SafetyProjection().project(replace(lost, phase=BimanualPhase.GRASP))

    assert approach.feasible
    assert not grasp.feasible
    assert grasp.fallback_reason == "force_closure_lost"


def test_base_and_wheel_safety_constraints_cannot_use_o6_reaction_motion() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py"
    ).read_text(encoding="utf-8")

    assert "contact_map[:, 31:43] = 0.0" in source
    assert "base_map[:, 31:43] = 0.0" in source


def test_external_o6_servo_owns_hand_position_and_velocity_barriers() -> None:
    sample = _input()
    q = sample.active_q.clone()
    qd = sample.active_qd.clone()
    q[31] = sample.q_max[31] + 1.0
    qd[31] = 2.0 * sample.qd_max[31]

    result = SafetyProjection(externally_servoed_hand=True).project(
        replace(sample, active_q=q, active_qd=qd)
    )

    assert result.feasible


def test_projection_accepts_an_already_safe_candidate_without_calling_qp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("QP should not run for an already feasible candidate")

    monkeypatch.setattr(
        "go2_pvcnn.control.m1_bimanual_coordination.safety_projection.solve_reference_qp",
        fail_if_called,
    )

    result = SafetyProjection().project(_input())

    assert result.feasible
    assert result.fallback_reason is None
    assert torch.equal(result.effort, torch.zeros(43, dtype=DTYPE))
