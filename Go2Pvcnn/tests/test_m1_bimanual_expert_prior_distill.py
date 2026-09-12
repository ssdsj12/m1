from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MixtureDistribution,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import FingertipMixtureNet


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_distill_fingertip_prior.py"
EVAL_SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_eval_fingertip_prior.py"


def _module():
    spec = importlib.util.spec_from_file_location("task7_distill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _eval_module():
    spec = importlib.util.spec_from_file_location("task7_eval", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _teacher() -> MixtureDistribution:
    logits = torch.tensor([[0.4, -0.7, 1.2, -0.2]], dtype=torch.float32)
    offsets = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    return MixtureDistribution(
        logits=logits,
        mean=offsets.expand(1, 4, 20, 5, 3).clone(),
        log_std=torch.full((1, 4, 20, 5, 3), -1.5, dtype=torch.float32),
    )


def _permuted(distribution: MixtureDistribution, permutation: torch.Tensor) -> MixtureDistribution:
    return MixtureDistribution(
        logits=distribution.logits[:, permutation],
        mean=distribution.mean[:, permutation],
        log_std=distribution.log_std[:, permutation],
    )


def test_distillation_scores_fixed_teacher_samples_without_component_alignment():
    module = _module()
    teacher = _teacher()
    permuted = _permuted(teacher, torch.tensor([2, 0, 3, 1]))
    samples = module.sample_mixture(teacher, samples_per_state=8, seed=42)

    assert samples.shape == (1, 8, 20, 5, 3)
    assert not torch.equal(teacher.logits, permuted.logits)
    torch.testing.assert_close(
        module.distribution_distillation_loss(teacher, samples),
        module.distribution_distillation_loss(permuted, samples),
    )


def test_distillation_loss_rejects_samples_with_wrong_frozen_shape():
    module = _module()
    with pytest.raises(ValueError, match="teacher_samples"):
        module.distribution_distillation_loss(_teacher(), torch.zeros(1, 8, 20, 3, 5))


def test_ensemble_teacher_aggregation_keeps_each_member_four_component_contract():
    module = _module()
    torch.manual_seed(3)
    members = (FingertipMixtureNet(hidden=(16, 16)), FingertipMixtureNet(hidden=(16, 16)))
    inputs = torch.zeros(2, 42, dtype=torch.float32)
    target = torch.zeros(2, 20, 5, 3, dtype=torch.float32)

    log_prob = module._ensemble_log_prob(members, inputs, target)
    samples = module._sample_ensemble(members, inputs, samples_per_state=3, seed=7)

    assert log_prob.shape == (2,)
    assert samples.shape == (2, 3, 20, 5, 3)
    assert torch.isfinite(log_prob).all() and torch.isfinite(samples).all()


def test_student_objective_has_nonzero_temporal_acceleration_and_jerk_terms():
    module = _module()
    torch.manual_seed(9)
    model = FingertipMixtureNet(hidden=(16, 16))
    inputs = torch.zeros(2, 42, dtype=torch.float32)
    target = torch.zeros(2, 20, 5, 3, dtype=torch.float32)
    samples = torch.zeros(2, 2, 20, 5, 3, dtype=torch.float32)

    total, label, sampled, acceleration, jerk = module.student_distillation_objective(
        model(inputs), target, samples
    )

    assert acceleration > 0.0 and jerk > 0.0
    torch.testing.assert_close(
        total,
        0.5 * label + 0.5 * sampled
        + module.DISTILLATION_CONFIG["acceleration_weight"] * acceleration
        + module.DISTILLATION_CONFIG["jerk_weight"] * jerk,
    )


def test_comparison_gate_rejects_negative_or_unversioned_report_values():
    result = _eval_module().accept_prior_comparison(
        {
            "nll_improvement_fraction": 0.2,
            "prior_off_jerk_p95": -1.0,
            "prior_on_jerk_p95": -2.0,
            "prior_off_task_success": 0.9,
            "prior_on_task_success": 0.9,
            "prior_off_safety_rejections": 0,
            "prior_on_safety_rejections": 0,
        }
    )

    assert not result.accepted
    assert result.reason == "invalid_metrics"


def test_distill_help_is_offline_and_does_not_create_output(tmp_path: Path):
    import subprocess
    import sys
    import os

    output = tmp_path / "must-stay-missing"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])},
    )

    assert completed.returncode == 0
    assert "offline" in completed.stdout.lower()
    assert not output.exists()
