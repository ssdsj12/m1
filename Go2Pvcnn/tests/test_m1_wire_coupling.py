from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np
import torch

from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
    DloCenterlineState,
)
from go2_pvcnn.control.m1_bimanual_coordination.wire_coupling import (
    WireCouplingPhase,
    WireCouplingState,
    propose_wire_coupling,
)
from go2_pvcnn.control.m1_bimanual_coordination.wire_scene_geometry import (
    WireSceneGeometry,
)
from tests.test_m1_bimanual_object_mpc import _snapshot


def _geometry() -> WireSceneGeometry:
    left = torch.tensor([0.40, 0.10, 1.20], dtype=torch.float64)
    right = torch.tensor([0.60, -0.10, 1.20], dtype=torch.float64)
    positions = torch.stack((left, 0.5 * (left + right), right))
    return WireSceneGeometry(
        centerline=DloCenterlineState(positions.numpy(), radius_m=0.004),
        center_b=positions[1].clone(),
        tangent_b=(right - left) / torch.linalg.vector_norm(right - left),
        radius_m=0.004,
        left_station_b=left,
        right_station_b=right,
    )


def _contact_snapshot(
    *,
    left: bool,
    right: bool,
    left_consistent: bool = True,
    right_consistent: bool = True,
    thumb_only: bool = False,
):
    geometry = _geometry()
    snapshot = _snapshot()

    def hand(original, station, contact, consistent):
        positions = original.fingertip_positions_b.clone()
        positions[0] = station + torch.tensor([0.001, 0.0, 0.0])
        positions[1] = station - torch.tensor([0.001, 0.0, 0.0])
        mask = torch.zeros(5, dtype=torch.bool)
        if contact:
            mask[: 1 if thumb_only else 2] = True
        return replace(
            original,
            fingertip_positions_b=positions,
            contact_mask=mask,
            contact_consistent=consistent,
        )

    return replace(
        snapshot,
        left_hand=hand(
            snapshot.left_hand,
            geometry.left_station_b,
            left,
            left_consistent,
        ),
        right_hand=hand(
            snapshot.right_hand,
            geometry.right_station_b,
            right,
            right_consistent,
        ),
    )


def test_fifth_bilateral_contact_tick_requests_bind_without_committing_bound_phase():
    geometry = _geometry()
    state = WireCouplingState.initial()

    for accepted_tick in range(4):
        proposal = propose_wire_coupling(
            state,
            _contact_snapshot(left=True, right=True),
            geometry,
        )
        state = proposal.next_state
        assert state.phase is WireCouplingPhase.CONTACT_LATCHING
        assert state.left_contact_ticks == accepted_tick + 1
        assert state.right_contact_ticks == accepted_tick + 1
        assert not proposal.bind_requested

    proposal = propose_wire_coupling(
        state,
        _contact_snapshot(left=True, right=True),
        geometry,
    )

    assert proposal.next_state.phase is WireCouplingPhase.CONTACT_LATCHING
    assert proposal.next_state.left_latched
    assert proposal.next_state.right_latched
    assert proposal.next_state.bind_count == 0
    assert proposal.bind_requested
    assert proposal.left_pinch_error_m == 0.0
    assert proposal.right_pinch_error_m == 0.0


def test_thumb_only_and_inconsistent_contact_do_not_advance_debounce():
    geometry = _geometry()
    state = WireCouplingState.initial()
    state = propose_wire_coupling(
        state,
        _contact_snapshot(left=True, right=True),
        geometry,
    ).next_state

    thumb_only = propose_wire_coupling(
        state,
        _contact_snapshot(left=True, right=True, thumb_only=True),
        geometry,
    ).next_state
    assert thumb_only.left_contact_ticks == 0
    assert thumb_only.right_contact_ticks == 0

    inconsistent = propose_wire_coupling(
        state,
        _contact_snapshot(
            left=True,
            right=True,
            left_consistent=False,
            right_consistent=False,
        ),
        geometry,
    ).next_state
    assert inconsistent.left_contact_ticks == 0
    assert inconsistent.right_contact_ticks == 0
    assert not inconsistent.left_latched
    assert not inconsistent.right_latched


def test_unilateral_latch_holds_while_other_side_continues_and_reset_is_clean():
    geometry = _geometry()
    state = WireCouplingState.initial()
    for _ in range(5):
        state = propose_wire_coupling(
            state,
            _contact_snapshot(left=True, right=False),
            geometry,
        ).next_state

    assert state.left_latched
    assert state.left_contact_ticks == 5
    assert not state.right_latched
    assert state.right_contact_ticks == 0

    progressed = propose_wire_coupling(
        state,
        _contact_snapshot(left=False, right=True),
        geometry,
    ).next_state
    assert progressed.left_latched
    assert progressed.left_contact_ticks == 5
    assert progressed.right_contact_ticks == 1

    reset = WireCouplingState.initial()
    assert reset.phase is WireCouplingPhase.SCENE_TRACKED
    assert reset.left_contact_ticks == 0
    assert reset.right_contact_ticks == 0
    assert not reset.left_latched
    assert not reset.right_latched


def test_proposal_is_pure_so_rejection_can_leave_accepted_state_byte_equivalent():
    geometry = _geometry()
    state = WireCouplingState.initial()
    before = pickle.dumps(state)

    proposal = propose_wire_coupling(
        state,
        _contact_snapshot(left=True, right=True),
        geometry,
    )

    assert pickle.dumps(state) == before
    assert proposal.next_state is not state
    assert state == WireCouplingState.initial()


def test_bound_contact_loss_enters_sticky_safe_hold_without_rebind():
    geometry = _geometry()
    bound = replace(
        WireCouplingState.initial(),
        phase=WireCouplingPhase.COSSERAT_BOUND,
        left_contact_ticks=5,
        right_contact_ticks=5,
        left_latched=True,
        right_latched=True,
        bind_count=1,
    )

    lost = propose_wire_coupling(
        bound,
        _contact_snapshot(left=True, right=False),
        geometry,
    )
    assert lost.next_state.phase is WireCouplingPhase.SAFE_HOLD
    assert lost.next_state.bind_count == 1
    assert lost.next_state.failure_reason is not None
    assert not lost.bind_requested

    safe = propose_wire_coupling(
        lost.next_state,
        _contact_snapshot(left=True, right=True),
        geometry,
    )
    assert safe.next_state == lost.next_state
    assert not safe.bind_requested


def test_station_error_above_five_millimetres_never_requests_bind():
    geometry = _geometry()
    snapshot = _contact_snapshot(left=True, right=True)
    shifted = snapshot.left_hand.fingertip_positions_b.clone()
    shifted[:2, 2] += 0.006
    snapshot = replace(
        snapshot,
        left_hand=replace(snapshot.left_hand, fingertip_positions_b=shifted),
    )
    state = WireCouplingState.initial()
    for _ in range(5):
        proposal = propose_wire_coupling(state, snapshot, geometry)
        state = proposal.next_state

    assert np.isclose(proposal.left_pinch_error_m, 0.006, atol=1.0e-12)
    assert not proposal.bind_requested
