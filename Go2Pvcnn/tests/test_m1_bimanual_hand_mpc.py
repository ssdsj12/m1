from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FingertipPriorDiagnostics,
    FingertipPriorTarget,
    PriorQueryResult,
)
import go2_pvcnn.control.m1_bimanual_coordination.hand_mpc as hand_mpc_module
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import (
    HandMpcCfg,
    HandMpcInput,
    O6HandMpc,
    build_hand_contact_qp,
)
from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import (
    rotvec_to_matrix,
)
from go2_pvcnn.control.m1_panda_coordination.qp_backend import DenseQpResult


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
        "fingertip_positions_b": torch.arange(15, dtype=DTYPE).reshape(5, 3) / 100.0,
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


def _target(*, mean: torch.Tensor | None = None, precision: torch.Tensor | None = None):
    return FingertipPriorTarget(
        mean_velocity=torch.ones(15, dtype=DTYPE) if mean is None else mean,
        precision=torch.ones(15, dtype=DTYPE) if precision is None else precision,
        component=2,
        probability=0.75,
    )


class _FixedPrior:
    def __init__(self, target=_target(), *, reason: str | None = None, error=None):
        self.value = target
        self.reason = reason
        self.error = error
        self.calls = []

    def target(self, sample, baseline_qd):
        self.calls.append((sample, baseline_qd.clone()))
        if self.error is not None:
            raise self.error
        if self.reason is not None:
            return PriorQueryResult(
                target=None,
                diagnostics=FingertipPriorDiagnostics(False, self.reason, 1.25),
            )
        return PriorQueryResult(
            target=self.value,
            diagnostics=FingertipPriorDiagnostics(True, None, 1.25),
        )


class _ExplodingTarget:
    @property
    def mean_velocity(self):
        raise RuntimeError("malicious target property")


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


def test_prior_disabled_is_exactly_existing_hand_solution():
    sample = _input(fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE))
    before = O6HandMpc().plan(sample)
    after = O6HandMpc(expert_prior=None).plan(sample)
    assert torch.equal(before.q_ref, after.q_ref)
    assert torch.equal(before.qd_ref, after.qd_ref)
    assert torch.equal(before.predicted_forces_b, after.predicted_forces_b)
    assert torch.equal(before.predicted_wrench_b, after.predicted_wrench_b)
    assert before.diagnostics == after.diagnostics


def test_prior_disabled_runs_only_the_existing_single_contact_qp(monkeypatch):
    calls = 0
    real_solve = hand_mpc_module.solve_reference_qp

    def count_solve(problem, **kwargs):
        nonlocal calls
        calls += 1
        return real_solve(problem, **kwargs)

    monkeypatch.setattr(hand_mpc_module, "solve_reference_qp", count_solve)
    O6HandMpc(expert_prior=None).plan(_input())
    assert calls == 1


def test_contact_prior_adds_only_a_psd_soft_term_and_preserves_hard_constraints():
    sample = _input(
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    baseline = build_hand_contact_qp(sample, HandMpcCfg())
    regularized = build_hand_contact_qp(sample, HandMpcCfg(), prior_target=_target())
    delta = regularized.hessian - baseline.hessian
    assert torch.linalg.eigvalsh(0.5 * (delta + delta.T)).min().item() >= -1.0e-10
    assert torch.equal(regularized.equality_matrix, baseline.equality_matrix)
    assert torch.equal(regularized.equality_rhs, baseline.equality_rhs)
    assert torch.equal(regularized.inequality_matrix, baseline.inequality_matrix)
    assert torch.equal(regularized.inequality_upper, baseline.inequality_upper)
    assert torch.equal(regularized.lower_bound, baseline.lower_bound)
    assert torch.equal(regularized.upper_bound, baseline.upper_bound)


def test_prior_precontact_projection_keeps_latched_joint_axes_and_bounds():
    sample = _input(
        phase=BimanualPhase.PRELOAD,
        contact_mask=torch.tensor([True, False, False, False, False]),
        q=torch.full((6,), 0.2, dtype=DTYPE),
    )
    result = O6HandMpc(expert_prior=_FixedPrior()).plan(sample)
    assert torch.equal(result.qd_ref[:2], torch.zeros(2, dtype=DTYPE))
    # Thumb contact masks three Cartesian target rows, while its two active
    # joint axes (not arbitrary Jacobian columns such as axis 5) are latched.
    assert result.qd_ref[5].item() > 0.0
    assert torch.all(result.qd_ref.abs() <= sample.qd_max)
    assert torch.all(result.q_ref >= sample.q_min)
    assert torch.all(result.q_ref <= sample.q_max)
    assert result.diagnostics.prior_qp_accepted


def test_precontact_projection_preserves_a_latch_after_contact_mask_dropout():
    prior = _FixedPrior()
    planner = O6HandMpc(expert_prior=prior)
    contacted = _input(
        phase=BimanualPhase.PRELOAD,
        contact_mask=torch.tensor([False, True, False, False, False]),
        q=torch.full((6,), 0.2, dtype=DTYPE),
    )
    planner.plan(contacted)
    dropped = replace(contacted, contact_mask=torch.zeros(5, dtype=torch.bool))
    result = planner.plan(dropped)
    assert result.qd_ref[2].item() == 0.0


def test_precontact_latch_never_overrides_an_incompatible_position_bound():
    sample = _input(
        phase=BimanualPhase.PRELOAD,
        contact_mask=torch.tensor([True, False, False, False, False]),
        q=torch.tensor([1.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=DTYPE),
    )
    expected = O6HandMpc().plan(sample)
    result = O6HandMpc(expert_prior=_FixedPrior()).plan(sample)
    assert torch.equal(result.q_ref, expected.q_ref)
    assert result.diagnostics.prior_fallback_reason == "prior_qp_rejected"


def test_contact_prior_queries_real_geometry_after_baseline_and_regularizes_rate():
    prior = _FixedPrior()
    sample = _input(
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    baseline = O6HandMpc().plan(sample)
    result = O6HandMpc(expert_prior=prior).plan(sample)
    assert len(prior.calls) == 1
    prior_sample, selected_from = prior.calls[0]
    assert torch.equal(prior_sample.fingertip_positions_b, sample.fingertip_positions_b)
    assert torch.equal(prior_sample.contact_jacobian, sample.contact_jacobian)
    assert torch.equal(prior_sample.qd, sample.qd)
    assert torch.equal(prior_sample.contact_mask, sample.contact_mask)
    assert prior_sample.phase is sample.phase
    assert torch.equal(selected_from, baseline.qd_ref)
    assert not torch.equal(result.qd_ref, baseline.qd_ref)
    assert result.diagnostics.prior_enabled
    assert result.diagnostics.prior_qp_accepted
    assert result.diagnostics.prior_component == 2
    assert result.diagnostics.prior_probability == pytest.approx(0.75)


def test_prior_query_and_soft_qp_use_palm_frame_geometry_without_changing_base_jacobian():
    prior = _FixedPrior()
    palm_pose = torch.tensor(
        [0.4, -0.2, 0.7, 0.0, 0.0, torch.pi / 2.0], dtype=DTYPE
    )
    rotation = rotvec_to_matrix(palm_pose[3:])
    positions_p = torch.arange(15, dtype=DTYPE).reshape(5, 3) / 100.0
    jacobian_p = torch.arange(90, dtype=DTYPE).reshape(5, 3, 6) / 100.0
    positions_b = positions_p @ rotation.T + palm_pose[:3]
    jacobian_b = (rotation @ jacobian_p).reshape(15, 6)
    sample = _input(
        fingertip_positions_b=positions_b,
        contact_jacobian=jacobian_b,
        palm_pose_b=palm_pose,
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    O6HandMpc(expert_prior=prior).plan(sample)
    queried = prior.calls[0][0]
    assert torch.allclose(queried.fingertip_positions_b, positions_p, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(queried.contact_jacobian, jacobian_p.reshape(15, 6), atol=1.0e-12, rtol=0.0)
    assert torch.equal(sample.contact_jacobian, jacobian_b)


def test_prior_query_failure_and_exception_return_exact_same_cycle_baseline():
    sample = _input(
        phase=BimanualPhase.PRELOAD,
        contact_mask=torch.zeros(5, dtype=torch.bool),
        q=torch.full((6,), 0.2, dtype=DTYPE),
    )
    expected = O6HandMpc().plan(sample)
    for prior, reason in (
        (_FixedPrior(reason="timeout"), "timeout"),
        (_FixedPrior(error=RuntimeError("bad prior")), "prior_exception"),
    ):
        result = O6HandMpc(expert_prior=prior).plan(sample)
        assert torch.equal(result.q_ref, expected.q_ref)
        assert torch.equal(result.qd_ref, expected.qd_ref)
        assert result.diagnostics.feasible
        assert not result.diagnostics.fallback_used
        assert result.diagnostics.prior_fallback_reason == reason


@pytest.mark.parametrize(
    "bad_target",
    [
        SimpleNamespace(
            mean_velocity=torch.zeros(15, dtype=DTYPE),
            precision=-torch.ones(15, dtype=DTYPE),
            component=0,
            probability=1.0,
        ),
        SimpleNamespace(
            mean_velocity=torch.full((15,), torch.nan, dtype=DTYPE),
            precision=torch.ones(15, dtype=DTYPE),
            component=0,
            probability=1.0,
        ),
    ],
)
def test_malicious_negative_or_nonfinite_prior_target_is_rejected(bad_target):
    sample = _input(
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    expected = O6HandMpc().plan(sample)
    result = O6HandMpc(expert_prior=_FixedPrior(target=bad_target)).plan(sample)
    assert torch.equal(result.q_ref, expected.q_ref)
    assert torch.equal(result.predicted_forces_b, expected.predicted_forces_b)
    assert result.diagnostics.prior_fallback_reason == "invalid_prior"


def test_malicious_target_property_exception_is_fail_closed():
    sample = _input(
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    expected = O6HandMpc().plan(sample)
    result = O6HandMpc(expert_prior=_FixedPrior(target=_ExplodingTarget())).plan(sample)
    assert torch.equal(result.q_ref, expected.q_ref)
    assert result.diagnostics.prior_fallback_reason == "invalid_prior"


def test_second_qp_failure_returns_same_cycle_baseline_without_pollution(monkeypatch):
    prior = _FixedPrior()
    planner = O6HandMpc(expert_prior=prior)
    first_sample = _input(
        contact_mask=torch.zeros(5, dtype=torch.bool),
        target_wrench_b=torch.zeros(6, dtype=DTYPE),
    )
    planner.plan(first_sample)
    sample = replace(first_sample, q=0.2 * torch.ones(6, dtype=DTYPE))
    expected = O6HandMpc().plan(sample)
    real_solve = hand_mpc_module.solve_reference_qp
    calls = 0

    def reject_second(problem, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_solve(problem, **kwargs)
        return DenseQpResult(
            solution=torch.zeros(21, dtype=DTYPE),
            success=False,
            iterations=1,
            max_equality_residual=1.0,
            max_inequality_violation=0.0,
            active_set=(),
        )

    monkeypatch.setattr(hand_mpc_module, "solve_reference_qp", reject_second)
    result = planner.plan(sample)
    assert calls == 2
    assert result.diagnostics.prior_fallback_reason == "prior_qp_rejected"
    assert result.diagnostics.feasible
    assert not result.diagnostics.fallback_used
    assert torch.equal(result.q_ref, expected.q_ref)
    assert torch.equal(planner._last_safe.q_ref, expected.q_ref)


def test_baseline_infeasible_uses_old_fallback_without_querying_prior():
    prior = _FixedPrior()
    planner = O6HandMpc(expert_prior=prior)
    fallback = planner.plan(_input(q=5.0 * torch.ones(6, dtype=DTYPE)))
    assert not fallback.diagnostics.feasible
    assert fallback.diagnostics.fallback_reason == "qp_infeasible"
    assert prior.calls == []
