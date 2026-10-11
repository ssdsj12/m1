"""DLO centerline state and wire-specific four-finger contact targets.

This is the perception/MPC boundary described by DeformX.  It deliberately
does not integrate Cosserat dynamics; a future DLO backend can update the
centerline state and contact wrench while the existing MPC consumes this
stable representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grasp_goal import ContactTarget


@dataclass(frozen=True)
class DloCenterlineState:
    """Discrete DLO centerline sampled in the robot base frame."""

    positions: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        points = np.asarray(self.positions, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("positions must contain at least two XYZ centerline points")
        if not np.isfinite(points).all():
            raise ValueError("positions must be finite")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1.0e-8):
            raise ValueError("centerline points must not be coincident")
        points.setflags(write=False)
        object.__setattr__(self, "positions", points)

    @property
    def tangents(self) -> np.ndarray:
        deltas = np.gradient(self.positions, axis=0)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        return deltas / np.maximum(norms, 1.0e-12)

    @property
    def midpoint(self) -> np.ndarray:
        return self.positions[len(self.positions) // 2].copy()

    @property
    def midpoint_tangent(self) -> np.ndarray:
        tangent = self.tangents[len(self.positions) // 2].copy()
        return tangent / max(float(np.linalg.norm(tangent)), 1.0e-12)


@dataclass(frozen=True)
class WireLineContactGoal:
    object_class: str
    center: tuple[float, float, float]
    tangent: tuple[float, float, float]
    contact_half_gap_m: float
    left_contact_targets: tuple[ContactTarget, ...]
    right_contact_targets: tuple[ContactTarget, ...]
    left_palm_approach_normal: tuple[float, float, float] | None = None
    right_palm_approach_normal: tuple[float, float, float] | None = None


def _orthogonal_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, tangent))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, reference)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    binormal = np.cross(tangent, normal)
    binormal /= max(float(np.linalg.norm(binormal)), 1.0e-12)
    return normal, binormal


def _fingertip_offset(radius_m: float, ratio: float) -> float:
    """Return the thumb/index cross-section offset for the O6 pinch."""
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.5:
        raise ValueError("fingertip_offset_ratio must be in (0, 1.5]")
    return float(radius_m) * float(ratio)


def build_wire_line_contact_goal(
    state: DloCenterlineState,
    *,
    clearance_m: float = 0.003,
    fingertip_offset_ratio: float = 1.5,
) -> WireLineContactGoal:
    """Create opposed thumb/index targets around the DLO midpoint.

    The two fingers on each hand are offset along the binormal, while the
    hands oppose across the normal.  This is intentionally contact-only; lift
    and Cosserat force feedback remain downstream responsibilities.
    """

    if not isinstance(state, DloCenterlineState):
        raise TypeError("state must be DloCenterlineState")
    if not np.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("clearance_m must be positive and finite")
    tangent = state.midpoint_tangent
    normal, binormal = _orthogonal_frame(tangent)
    center = state.midpoint
    gap = state.radius_m + float(clearance_m)
    finger_offset = _fingertip_offset(state.radius_m, fingertip_offset_ratio)

    def make_side(sign: float) -> tuple[ContactTarget, ...]:
        inward = -sign * normal
        positions = (center + sign * gap * normal + finger_offset * binormal,
                     center + sign * gap * normal - finger_offset * binormal)
        return tuple(
            ContactTarget(
                position=tuple(float(v) for v in position),
                normal=tuple(float(v) for v in inward),
                tangent=tuple(float(v) for v in tangent),
            )
            for position in positions
        )

    return WireLineContactGoal(
        object_class="wire",
        center=tuple(float(v) for v in center),
        tangent=tuple(float(v) for v in tangent),
        contact_half_gap_m=float(gap),
        left_contact_targets=make_side(-1.0),
        right_contact_targets=make_side(1.0),
        left_palm_approach_normal=tuple(float(v) for v in normal),
        right_palm_approach_normal=tuple(float(v) for v in -normal),
    )


def build_wire_station_contact_goal(
    state: DloCenterlineState,
    *,
    clearance_m: float = 0.003,
    fingertip_offset_ratio: float = 1.5,
) -> WireLineContactGoal:
    """Place opposed thumb/index pinches at the held-span endpoints.

    Palm approach normals are independent of the opposed finger normals,
    whose mean is zero. Each pinch center remains exactly on its station.
    """
    if not isinstance(state, DloCenterlineState):
        raise TypeError("state must be DloCenterlineState")
    if not np.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("clearance_m must be positive and finite")
    tangent = state.midpoint_tangent
    normal, binormal = _orthogonal_frame(tangent)
    gap = state.radius_m + float(clearance_m)
    finger_offset = _fingertip_offset(state.radius_m, fingertip_offset_ratio)

    def make_side(center, sign):
        offset = sign * gap * normal + finger_offset * binormal
        return tuple(
            ContactTarget(
                position=tuple(float(v) for v in center + finger_sign * offset),
                # Isaac reports the reaction force acting on the fingertip,
                # which points away from the wire surface at this target.
                normal=tuple(float(v) for v in finger_sign * offset / np.linalg.norm(offset)),
                tangent=tuple(float(v) for v in tangent),
            )
            for finger_sign in (1.0, -1.0)
        )

    return WireLineContactGoal(
        object_class="wire",
        center=tuple(float(v) for v in state.midpoint),
        tangent=tuple(float(v) for v in tangent),
        contact_half_gap_m=float(gap),
        left_contact_targets=make_side(state.positions[0], -1.0),
        right_contact_targets=make_side(state.positions[-1], 1.0),
        left_palm_approach_normal=tuple(float(v) for v in normal),
        right_palm_approach_normal=tuple(float(v) for v in -normal),
    )


__all__ = ["DloCenterlineState", "WireLineContactGoal", "build_wire_line_contact_goal", "build_wire_station_contact_goal"]
