from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    DEXMANIPNET_REVISION,
    FINGER_ORDER,
    MANIPTRANS_COMMIT,
    MIXTURE_COMPONENTS,
    PRIOR_HORIZON,
    ExpertWindow,
    MixtureDistribution,
    PriorPhase,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import (
    SOURCE_HANDS,
)


def test_expert_window_freezes_geometry_only_contract():
    window = ExpertWindow(
        fingertip_position_palm=torch.zeros(5, 3),
        fingertip_velocity_palm=torch.zeros(5, 3),
        contact_mask=torch.zeros(5, dtype=torch.bool),
        phase=PriorPhase.APPROACH,
        future_fingertip_velocity_palm=torch.zeros(20, 5, 3),
        source_group="favor/seq/rh",
        source_sha256="0" * 64,
    )
    assert window.network_input().shape == (42,)
    assert window.target.shape == (20, 5, 3)


def test_source_pins_and_five_finger_order_are_frozen():
    assert DEXMANIPNET_REVISION == "3933fae5fe83498fb314a0924aa21d5038fba5a5"
    assert MANIPTRANS_COMMIT == "a3d08cfe3c3a5868a7f057533bcaf759c5af4705"
    assert FINGER_ORDER == ("thumb", "index", "middle", "ring", "pinky")
    assert all(len(set(spec.fingertip_links)) == 5 for spec in SOURCE_HANDS.values())


def test_distribution_and_window_reject_invalid_frozen_shapes():
    window = ExpertWindow(
        fingertip_position_palm=torch.zeros(5, 3),
        fingertip_velocity_palm=torch.zeros(5, 3),
        contact_mask=torch.zeros(5, dtype=torch.bool),
        phase=PriorPhase.HOLD,
        future_fingertip_velocity_palm=torch.zeros(PRIOR_HORIZON, 5, 3),
        source_group="oakink/sequence/rh",
        source_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="shape"):
        replace(window, fingertip_position_palm=torch.zeros(4, 3))
    with pytest.raises(TypeError, match="PriorPhase"):
        replace(window, phase=PriorPhase.HOLD.value)

    with pytest.raises(ValueError, match="shape"):
        MixtureDistribution(
            logits=torch.zeros(MIXTURE_COMPONENTS),
            mean=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3),
            log_std=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 2),
        )
