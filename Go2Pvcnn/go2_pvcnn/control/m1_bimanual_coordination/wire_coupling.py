"""Pure bilateral contact-latch proposals for staged wire coupling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
import math
from typing import Any

import numpy as np
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
    RECLAMPING = auto()


@dataclass(frozen=True)
class WireReclampStatus:
    """Opt-in: current contacts and the last tick's final acceptance decision."""

    bilateral_contact: bool
    dwell_complete: bool
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bilateral_contact, bool) or not isinstance(self.dwell_complete, bool):
            raise TypeError("wire contact and dwell status must be bool")
        if self.terminal_reason is not None and not isinstance(self.terminal_reason, str):
            raise TypeError("terminal_reason must be str or None")


def _finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real values") from exc
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return tuple(float(item) for item in array)


def _contact_mask(value: Any, name: str) -> tuple[bool, bool, bool, bool, bool]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (5,):
        raise ValueError(f"{name} must have shape (5,)")
    if array.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    return tuple(bool(item) for item in array)


@dataclass(frozen=True)
class WireCouplingDiagnostics:
    """Immutable, JSON-safe observation of one staged wire control tick."""

    object_id: str
    target_prim_path: str
    state: str
    transition_reason: str
    centerline_midpoint_b: tuple[float, float, float]
    tangent_b: tuple[float, float, float]
    radius_m: float
    left_station_b: tuple[float, float, float]
    right_station_b: tuple[float, float, float]
    left_pinch_error_m: float
    right_pinch_error_m: float
    left_contact_mask: tuple[bool, bool, bool, bool, bool]
    right_contact_mask: tuple[bool, bool, bool, bool, bool]
    left_contact_ticks: int
    right_contact_ticks: int
    bind_count: int
    deformx_elapsed_time: float
    wbc_feasible: bool
    wire_goal_committed: bool

    def __post_init__(self) -> None:
        for name in ("object_id", "target_prim_path", "state", "transition_reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "centerline_midpoint_b",
            "tangent_b",
            "left_station_b",
            "right_station_b",
        ):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        tangent_norm = math.sqrt(sum(value * value for value in self.tangent_b))
        if not math.isclose(tangent_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("tangent_b must be a unit vector")
        for name in (
            "radius_m",
            "left_pinch_error_m",
            "right_pinch_error_m",
            "deformx_elapsed_time",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a real scalar")
            try:
                normalized = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"{name} must be a real scalar") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            if normalized < 0.0 or (name == "radius_m" and normalized == 0.0):
                qualifier = "positive" if name == "radius_m" else "non-negative"
                raise ValueError(f"{name} must be {qualifier}")
            object.__setattr__(self, name, normalized)
        for name in ("left_contact_mask", "right_contact_mask"):
            object.__setattr__(self, name, _contact_mask(getattr(self, name), name))
        for name in ("left_contact_ticks", "right_contact_ticks", "bind_count"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        for name in ("wbc_feasible", "wire_goal_committed"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
            object.__setattr__(self, name, bool(getattr(self, name)))

    def to_json(self) -> dict[str, object]:
        """Return a detached mapping accepted by ``json.dumps(..., allow_nan=False)``."""

        return {
            "object_id": self.object_id,
            "target_prim_path": self.target_prim_path,
            "state": self.state,
            "transition_reason": self.transition_reason,
            "centerline_midpoint_b": list(self.centerline_midpoint_b),
            "tangent_b": list(self.tangent_b),
            "radius_m": self.radius_m,
            "left_station_b": list(self.left_station_b),
            "right_station_b": list(self.right_station_b),
            "left_pinch_error_m": self.left_pinch_error_m,
            "right_pinch_error_m": self.right_pinch_error_m,
            "left_contact_mask": list(self.left_contact_mask),
            "right_contact_mask": list(self.right_contact_mask),
            "left_contact_ticks": self.left_contact_ticks,
            "right_contact_ticks": self.right_contact_ticks,
            "bind_count": self.bind_count,
            "deformx_elapsed_time": self.deformx_elapsed_time,
            "wbc_feasible": self.wbc_feasible,
            "wire_goal_committed": self.wire_goal_committed,
        }


# Prebuilt so even a first-tick observer construction failure has a finite,
# explicitly unavailable record. Its identity/counters cannot validate as a bind.
UNAVAILABLE_WIRE_DIAGNOSTICS = WireCouplingDiagnostics(
    object_id="unavailable",
    target_prim_path="unavailable",
    state="SAFE_HOLD",
    transition_reason="wire_diagnostics_unavailable",
    centerline_midpoint_b=(0.0, 0.0, 0.0),
    tangent_b=(1.0, 0.0, 0.0),
    radius_m=1.0,
    left_station_b=(0.0, 0.0, 0.0),
    right_station_b=(0.0, 0.0, 0.0),
    left_pinch_error_m=1.0,
    right_pinch_error_m=1.0,
    left_contact_mask=(False,) * 5,
    right_contact_mask=(False,) * 5,
    left_contact_ticks=0,
    right_contact_ticks=0,
    bind_count=0,
    deformx_elapsed_time=0.0,
    wbc_feasible=False,
    wire_goal_committed=False,
)


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
    *,
    reclamp_status: WireReclampStatus | None = None,
) -> WireCouplingProposal:
    """Return the next candidate latch state without mutating accepted state."""

    if not isinstance(state, WireCouplingState):
        raise TypeError("state must be WireCouplingState")
    if not isinstance(snapshot, BimanualSnapshot):
        raise TypeError("snapshot must be BimanualSnapshot")
    if not isinstance(geometry, WireSceneGeometry):
        raise TypeError("geometry must be WireSceneGeometry")
    if reclamp_status is not None and not isinstance(reclamp_status, WireReclampStatus):
        raise TypeError("reclamp_status must be WireReclampStatus or None")

    left_error = _pinch_error(snapshot.left_hand, geometry.left_station_b)
    right_error = _pinch_error(snapshot.right_hand, geometry.right_station_b)

    if state.phase is WireCouplingPhase.SAFE_HOLD:
        return WireCouplingProposal(state, False, left_error, right_error)
    if reclamp_status is not None and reclamp_status.terminal_reason is not None:
        failed = replace(state, phase=WireCouplingPhase.SAFE_HOLD,
                         failure_reason=reclamp_status.terminal_reason)
        return WireCouplingProposal(failed, False, left_error, right_error)

    left_contact = _contact_qualifies(snapshot.left_hand)
    right_contact = _contact_qualifies(snapshot.right_hand)

    if state.phase in (WireCouplingPhase.COSSERAT_BOUND, WireCouplingPhase.RECLAMPING):
        qualified = (left_contact and right_contact if reclamp_status is None else
                     reclamp_status.bilateral_contact)
        ready = state.phase is WireCouplingPhase.COSSERAT_BOUND or (
            reclamp_status is not None and reclamp_status.dwell_complete
        )
        if qualified and ready:
            restored = (state if state.phase is WireCouplingPhase.COSSERAT_BOUND else
                        replace(state, phase=WireCouplingPhase.COSSERAT_BOUND))
            return WireCouplingProposal(restored, False, left_error, right_error)
        if reclamp_status is not None:
            paused = replace(state, phase=WireCouplingPhase.RECLAMPING)
            return WireCouplingProposal(paused, False, left_error, right_error)
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
    "WireCouplingDiagnostics",
    "WireCouplingPhase",
    "WireCouplingProposal",
    "WireCouplingState",
    "WireReclampStatus",
    "propose_wire_coupling",
]
