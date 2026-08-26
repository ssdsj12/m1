from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.residual_observation import (
    ARM_OBSERVATION_DIM,
    M1_OBSERVATION_DIM,
    RESIDUAL_OBSERVATION_DIM,
    STABILITY_OBSERVATION_DIM,
    TASK_OBSERVATION_DIM,
    ResidualObservationParts,
    build_residual_observation,
)


def _parts(batch=2, dtype=torch.float64):
    return ResidualObservationParts(
        m1_state=torch.arange(batch * M1_OBSERVATION_DIM, dtype=dtype).reshape(batch, M1_OBSERVATION_DIM),
        arm_state=torch.arange(batch * ARM_OBSERVATION_DIM, dtype=dtype).reshape(batch, ARM_OBSERVATION_DIM),
        filtered_mount_wrench=torch.arange(batch * 6, dtype=dtype).reshape(batch, 6),
        task_state=torch.arange(batch * TASK_OBSERVATION_DIM, dtype=dtype).reshape(batch, TASK_OBSERVATION_DIM),
        sigma_min=torch.full((batch, 1), 0.2, dtype=dtype),
        joint_limit_margin_min=torch.full((batch, 1), 0.1, dtype=dtype),
        joint_limit_margin_mean=torch.full((batch, 1), 0.3, dtype=dtype),
        support_margin=torch.full((batch, 1), 0.05, dtype=dtype),
        previous_residual=torch.arange(batch * 8, dtype=dtype).reshape(batch, 8),
    )


def test_residual_observation_has_deterministic_group_and_flatten_order():
    parts = _parts()

    result = build_residual_observation(parts)

    expected_stability = torch.cat(
        (
            parts.sigma_min,
            parts.joint_limit_margin_min,
            parts.joint_limit_margin_mean,
            parts.support_margin,
        ),
        dim=-1,
    )
    expected = torch.cat(
        (
            parts.m1_state,
            parts.arm_state,
            parts.filtered_mount_wrench,
            parts.task_state,
            expected_stability,
            parts.previous_residual,
        ),
        dim=-1,
    )
    assert result.groups == (
        "m1_state",
        "arm_state",
        "filtered_mount_wrench",
        "task_state",
        "stability",
        "previous_residual",
    )
    assert result.stability.shape == (2, STABILITY_OBSERVATION_DIM)
    assert result.flat.shape == (2, RESIDUAL_OBSERVATION_DIM)
    assert torch.equal(result.flat, expected)


def test_residual_observation_returns_clones():
    parts = _parts()
    result = build_residual_observation(parts)
    before = result.flat.clone()

    parts.m1_state.zero_()
    result.previous_residual.zero_()

    assert torch.equal(result.flat, before)


def test_residual_observation_supports_arbitrary_batch_prefix():
    parts = _parts(batch=6)
    parts = replace(
        parts,
        m1_state=parts.m1_state.reshape(2, 3, M1_OBSERVATION_DIM),
        arm_state=parts.arm_state.reshape(2, 3, ARM_OBSERVATION_DIM),
        filtered_mount_wrench=parts.filtered_mount_wrench.reshape(2, 3, 6),
        task_state=parts.task_state.reshape(2, 3, TASK_OBSERVATION_DIM),
        sigma_min=parts.sigma_min.reshape(2, 3, 1),
        joint_limit_margin_min=parts.joint_limit_margin_min.reshape(2, 3, 1),
        joint_limit_margin_mean=parts.joint_limit_margin_mean.reshape(2, 3, 1),
        support_margin=parts.support_margin.reshape(2, 3, 1),
        previous_residual=parts.previous_residual.reshape(2, 3, 8),
    )

    result = build_residual_observation(parts)

    assert result.flat.shape == (2, 3, RESIDUAL_OBSERVATION_DIM)


def test_residual_observation_rejects_batch_dtype_and_nonfinite_mismatch():
    parts = _parts()
    with pytest.raises(ValueError, match="batch"):
        build_residual_observation(replace(parts, arm_state=parts.arm_state[:1]))
    with pytest.raises(TypeError, match="dtype"):
        build_residual_observation(replace(parts, task_state=parts.task_state.float()))
    invalid = parts.filtered_mount_wrench.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_residual_observation(replace(parts, filtered_mount_wrench=invalid))
