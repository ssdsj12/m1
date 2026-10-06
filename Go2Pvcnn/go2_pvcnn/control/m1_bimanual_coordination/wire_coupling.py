"""Pure bilateral contact-latch proposals for staged wire coupling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import torch

from .contracts import BimanualSnapshot, SideHandState
from .wire_scene_geometry import WireSceneGeometry


_LATCH_TICKS = 5
_MAX_BIND_ERROR_M = 0.005


class WireCouplingPhase(Enum):
    SCENE_TRACKED = auto()
    APPROACH_ENDPOINTS = auto()
    CONTACT_LATCHING = auto()
    COSSERAT_BOUND = auto()
    SAFE_HOLD = auto()


@dataclass(frozen=True)
class WireCouplingState:
    phase: WireCouplingPhase
    left_contact_ticks: int
    right_contact_ticks: int
    left_latched: bool
    right_latched: bool
    bind_count: int
    failure_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, WireCouplingPhase):
            raise TypeError("phase must be WireCouplingPhase")
        for name in ("left_contact_ticks", "right_contact_ticks", "bind_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("left_latched", "right_latched"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise TypeError("failure_reason must be str or None")

    @classmethod
    def initial(cls) -> "WireCouplingState":
        return cls(
            phase=WireCouplingPhase.SCENE_TRACKED,
            left_contact_ticks=0,
            right_contact_ticks=0,
            left_latched=False,
            right_latched=False,
            bind_count=0,
            failure_reason=None,
        )


@dataclass(frozen=True)
class WireCouplingProposal:
    next_state: WireCouplingState
    bind_requested: bool
    left_pinch_error_m: float
    right_pinch_error_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.next_state, WireCouplingState):
            raise TypeError("next_state must be WireCouplingState")
        if not isinstance(self.bind_requested, bool):
            raise TypeError("bind_requested must be bool")
        for name in ("left_pinch_error_m", "right_pinch_error_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")


def _pinch_error(hand: SideHandState, station_b: torch.Tensor) -> float:
    pinch_b = hand.fingertip_positions_b[:2].mean(dim=0)
    return float(torch.linalg.vector_norm(pinch_b - station_b).item())


def _contact_qualifies(hand: SideHandState) -> bool:
    return hand.contact_consistent and bool(hand.contact_mask[:2].all().item())


def _advance_side(
    ticks: int,
    latched: bool,
    qualifies: bool,
) -> tuple[int, bool]:
    if latched:
        return _LATCH_TICKS, True
    next_ticks = min(ticks + 1, _LATCH_TICKS) if qualifies else 0
    return next_ticks, next_ticks >= _LATCH_TICKS


def propose_wire_coupling(
    state: WireCouplingState,
    snapshot: BimanualSnapshot,
    geometry: WireSceneGeometry,
) -> WireCouplingProposal:
    """Return the next candidate latch state without mutating accepted state."""

    if not isinstance(state, WireCouplingState):
        raise TypeError("state must be WireCouplingState")
    if not isinstance(snapshot, BimanualSnapshot):
        raise TypeError("snapshot must be BimanualSnapshot")
    if not isinstance(geometry, WireSceneGeometry):
        raise TypeError("geometry must be WireSceneGeometry")

    left_error = _pinch_error(snapshot.left_hand, geometry.left_station_b)
    right_error = _pinch_error(snapshot.right_hand, geometry.right_station_b)

    if state.phase is WireCouplingPhase.SAFE_HOLD:
        return WireCouplingProposal(state, False, left_error, right_error)

    left_contact = _contact_qualifies(snapshot.left_hand)
    right_contact = _contact_qualifies(snapshot.right_hand)

    if state.phase is WireCouplingPhase.COSSERAT_BOUND:
        if left_contact and right_contact:
            return WireCouplingProposal(state, False, left_error, right_error)
        failed = WireCouplingState(
            phase=WireCouplingPhase.SAFE_HOLD,
            left_contact_ticks=state.left_contact_ticks,
            right_contact_ticks=state.right_contact_ticks,
            left_latched=state.left_latched,
            right_latched=state.right_latched,
            bind_count=state.bind_count,
            failure_reason="post_bind_thumb_index_contact_loss",
        )
        return WireCouplingProposal(failed, False, left_error, right_error)

    left_ticks, left_latched = _advance_side(
        state.left_contact_ticks,
        state.left_latched,
        left_contact,
    )
    right_ticks, right_latched = _advance_side(
        state.right_contact_ticks,
        state.right_latched,
        right_contact,
    )
    has_latch_progress = any(
        (left_ticks, right_ticks, left_latched, right_latched)
    )
    phase = (
        WireCouplingPhase.CONTACT_LATCHING
        if has_latch_progress or state.phase is WireCouplingPhase.CONTACT_LATCHING
        else WireCouplingPhase.APPROACH_ENDPOINTS
    )
    both_latched = left_latched and right_latched
    within_bind_tolerance = (
        left_error <= _MAX_BIND_ERROR_M and right_error <= _MAX_BIND_ERROR_M
    )
    failure_reason = None
    if both_latched and not within_bind_tolerance:
        phase = WireCouplingPhase.SAFE_HOLD
        failure_reason = "pinch_station_error_exceeds_0.005_m"

    next_state = WireCouplingState(
        phase=phase,
        left_contact_ticks=left_ticks,
        right_contact_ticks=right_ticks,
        left_latched=left_latched,
        right_latched=right_latched,
        bind_count=state.bind_count,
        failure_reason=failure_reason,
    )
    return WireCouplingProposal(
        next_state=next_state,
        bind_requested=both_latched and within_bind_tolerance,
        left_pinch_error_m=left_error,
        right_pinch_error_m=right_error,
    )


__all__ = [
    "WireCouplingPhase",
    "WireCouplingProposal",
    "WireCouplingState",
    "propose_wire_coupling",
]
