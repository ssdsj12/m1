from __future__ import annotations

import math

import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MixtureDistribution,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import (
    FingertipMixtureNet,
    mixture_log_prob,
    mixture_nll,
    temporal_regularizer,
)


def test_mixture_network_outputs_normalized_finite_distribution():
    model = FingertipMixtureNet(hidden=(128, 128, 128))

    dist = model(torch.zeros(3, 42))

    assert dist.logits.shape == (3, 4)
    assert dist.mean.shape == dist.log_std.shape == (3, 4, 20, 5, 3)
    assert torch.isfinite(dist.mean).all()
    torch.testing.assert_close(dist.logits.softmax(-1).sum(-1), torch.ones(3))


def test_mixture_nll_matches_single_standard_normal_density():
    dist = MixtureDistribution(
        logits=torch.zeros(1, 4, dtype=torch.float32),
        mean=torch.zeros(1, 4, 20, 5, 3, dtype=torch.float32),
        log_std=torch.zeros(1, 4, 20, 5, 3, dtype=torch.float32),
    )
    target = torch.zeros(1, 20, 5, 3, dtype=torch.float32)

    log_prob = mixture_log_prob(dist, target)

    torch.testing.assert_close(log_prob, torch.tensor([-300 * 0.5 * math.log(2.0 * math.pi)]))
    torch.testing.assert_close(mixture_nll(dist, target), -log_prob.mean())


def test_temporal_regularizer_is_finite_and_zero_for_constant_velocity():
    dist = MixtureDistribution(
        logits=torch.zeros(2, 4, dtype=torch.float32),
        mean=torch.ones(2, 4, 20, 5, 3, dtype=torch.float32),
        log_std=torch.zeros(2, 4, 20, 5, 3, dtype=torch.float32),
    )

    acceleration, jerk = temporal_regularizer(dist)

    torch.testing.assert_close(acceleration, torch.zeros(()))
    torch.testing.assert_close(jerk, torch.zeros(()))
    assert torch.isfinite(acceleration + jerk)
