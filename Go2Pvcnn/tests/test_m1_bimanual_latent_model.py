from __future__ import annotations

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import (
    ACTION_DIM,
    FEATURE_ORDER,
    HORIZON,
    LATENT_DIM,
    LATENT_GROUPS,
    MODEL_FORMAT_VERSION,
    STATE_DIM,
    TASK_FEATURE_DIM,
    LatentArtifactMetadata,
    LatentNormalizer,
    pack_state_features,
)
from go2_pvcnn.control.m1_bimanual_coordination.latent_model import (
    LatentActionModel,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _metadata() -> LatentArtifactMetadata:
    return LatentArtifactMetadata(
        format_version=MODEL_FORMAT_VERSION,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        task_feature_dim=TASK_FEATURE_DIM,
        latent_dim=LATENT_DIM,
        action_order=tuple(f"joint_{index}" for index in range(ACTION_DIM)),
        feature_order=FEATURE_ORDER,
        normalization_sha256="a" * 64,
        dataset_sha256="b" * 64,
        training_seed=42,
    )


def test_latent_model_shapes_and_group_contract() -> None:
    model = LatentActionModel()
    state = torch.zeros((2, STATE_DIM), dtype=torch.float32)
    teacher_action = torch.zeros((2, HORIZON, ACTION_DIM), dtype=torch.float32)
    teacher_task = torch.zeros(
        (2, HORIZON, TASK_FEATURE_DIM), dtype=torch.float32
    )

    z = model.encode(state, teacher_action, teacher_task)

    assert z.shape == (2, 16)
    assert model.decode_trajectory(state, z).shape == (2, 25, 43)
    assert model.body_action(
        state,
        z,
        torch.zeros(2, 1),
        torch.zeros(2, 43),
    ).shape == (2, 43)
    assert LATENT_GROUPS == {
        "support": (0, 4),
        "palms": (4, 10),
        "hands": (10, 14),
        "platform": (14, 16),
    }
    assert torch.all(z.abs() <= 1.0)


def test_pack_state_features_has_frozen_111_value_order() -> None:
    snapshot = _snapshot()
    snapshot.base_state.copy_(torch.arange(13, dtype=torch.float64))

    features = pack_state_features(snapshot)

    assert features.shape == (111,)
    assert features.dtype == torch.float32
    assert torch.equal(features[:13], torch.arange(13, dtype=torch.float32))
    assert len(FEATURE_ORDER) == 111
    assert FEATURE_ORDER[0] == "base_state.0"
    assert FEATURE_ORDER[-1] == "box.twist_b.5"


def test_normalizer_round_trip_and_rejects_zero_scale() -> None:
    mean = torch.linspace(-1.0, 1.0, STATE_DIM)
    scale = torch.linspace(0.1, 2.0, STATE_DIM)
    normalizer = LatentNormalizer(mean=mean, scale=scale)
    value = torch.randn(3, STATE_DIM)

    assert torch.allclose(
        normalizer.denormalize(normalizer.normalize(value)), value, atol=1.0e-6
    )
    with pytest.raises(ValueError, match="positive"):
        LatentNormalizer(mean=mean, scale=torch.zeros(STATE_DIM))


def test_artifact_rejects_wrong_action_order_and_dimensions() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="action_order"):
        metadata.validate_runtime(
            action_order=tuple(reversed(metadata.action_order)),
            state_dim=STATE_DIM,
            normalization_sha256=metadata.normalization_sha256,
        )
    with pytest.raises(ValueError, match="state_dim"):
        metadata.validate_runtime(
            action_order=metadata.action_order,
            state_dim=110,
            normalization_sha256=metadata.normalization_sha256,
        )


def test_fixed_seed_reproduces_model_initialization() -> None:
    torch.manual_seed(42)
    first = LatentActionModel()
    torch.manual_seed(42)
    second = LatentActionModel()

    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )

