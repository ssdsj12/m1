"""Deterministic diagnostics for the reduced-wire grasp/lift boundary.

The wire path deliberately keeps this contract independent of Isaac/DeformX.
Callers provide already-filtered physical contact observations; this module
only validates, latches, and serializes the values used by the safety gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def _mask(name: str, value: Any) -> tuple[bool, ...]:
    try:
        values = tuple(bool(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an iterable of booleans") from exc
    if len(values) != 5:
        raise ValueError(f"{name} must contain five fingertip values")
    return values


def _number(name: str, value: Any, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class WirePipelineDiagnostics:
    """One complete, JSON-safe reduced-wire gate observation."""

    phase: str
    bilateral_contact: bool
    left_contact_mask: tuple[bool, ...]
    right_contact_mask: tuple[bool, ...]
    left_clamp_force_n: float
    right_clamp_force_n: float
    force_closure_margin: float
    slip_m: float
    wire_feasible: bool
    arm_recoverability: bool
    command_accepted: bool
    terminal_reason: str | None = None
    wire_failure_reason: str | None = None
    left_latched: bool = False
    right_latched: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("phase must be a non-empty string")
        object.__setattr__(self, "phase", self.phase.upper())
        for name in ("bilateral_contact", "wire_feasible", "arm_recoverability", "command_accepted"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("left_contact_mask", "right_contact_mask"):
            object.__setattr__(self, name, _mask(name, getattr(self, name)))
        for name in ("left_clamp_force_n", "right_clamp_force_n", "force_closure_margin", "slip_m"):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        for name in ("left_latched", "right_latched"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("terminal_reason", "wire_failure_reason"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string")

    @property
    def left_contact_count(self) -> int:
        return sum(self.left_contact_mask)

    @property
    def right_contact_count(self) -> int:
        return sum(self.right_contact_mask)

    def with_latches(self, left: bool, right: bool, *, terminal_reason: str | None = None) -> "WirePipelineDiagnostics":
        return WirePipelineDiagnostics(
            **{**self.__dict__, "left_latched": bool(left), "right_latched": bool(right),
               "terminal_reason": terminal_reason if terminal_reason is not None else self.terminal_reason}
        )

    def to_json(self) -> dict[str, object]:
        """Return deterministic JSON primitives (never tensors or NaN)."""

        return {
            "phase": self.phase,
            "bilateral_contact": self.bilateral_contact,
            "left_contact_mask": list(self.left_contact_mask),
            "right_contact_mask": list(self.right_contact_mask),
            "left_contact_count": self.left_contact_count,
            "right_contact_count": self.right_contact_count,
            "left_clamp_force_n": self.left_clamp_force_n,
            "right_clamp_force_n": self.right_clamp_force_n,
            "force_closure_margin": self.force_closure_margin,
            "slip_m": self.slip_m,
            "wire_feasible": self.wire_feasible,
            "arm_recoverability": self.arm_recoverability,
            "command_accepted": self.command_accepted,
            "left_latched": self.left_latched,
            "right_latched": self.right_latched,
            "wire_failure_reason": self.wire_failure_reason,
            "terminal_reason": self.terminal_reason,
        }


WireDiagnostics = WirePipelineDiagnostics

__all__ = ["WirePipelineDiagnostics", "WireDiagnostics"]
