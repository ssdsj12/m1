"""Deterministic normal and safe phase transitions for bimanual manipulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import BimanualPhase, BimanualSnapshot


@dataclass(frozen=True)
class BimanualMissionCfg:
    physics_dt: float = 0.005
    approach_dwell_steps: int = 20
    preload_dwell_steps: int = 20
    grasp_dwell_steps: int = 20
    hold_steps: int = 600
    safe_hold_steps: int = 20
    max_consecutive_failures: int = 3
    lift_height_m: float = 0.10
    max_relative_palm_slip_m: float = 0.005
    supported_twist_tolerance: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "physics_dt",
            "lift_height_m",
            "max_relative_palm_slip_m",
            "supported_twist_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "approach_dwell_steps",
            "preload_dwell_steps",
            "grasp_dwell_steps",
            "hold_steps",
            "safe_hold_steps",
            "max_consecutive_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BimanualMissionDiagnostics:
    command_accepted: bool
    palms_reached: bool
    bilateral_contact: bool
    force_closure_margin: float
    relative_palm_slip_m: float
    box_twist_norm: float
    box_supported: bool
    hands_open: bool
    collision_margin_m: float
    support_margin_m: float
    subsystem_failure: str | None

    def __post_init__(self) -> None:
        for name in (
            "command_accepted",
            "palms_reached",
            "bilateral_contact",
            "box_supported",
            "hands_open",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "force_closure_margin",
            "relative_palm_slip_m",
            "box_twist_norm",
            "collision_margin_m",
            "support_margin_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.relative_palm_slip_m < 0.0 or self.box_twist_norm < 0.0:
            raise ValueError("slip and box twist norms must be non-negative")
        if self.subsystem_failure is not None and (
            not isinstance(self.subsystem_failure, str)
            or not self.subsystem_failure.strip()
        ):
            raise ValueError("subsystem_failure must be None or a non-empty string")


@dataclass(frozen=True)
class BimanualMissionState:
    phase: BimanualPhase
    step: int
    phase_steps: int
    consecutive_failures: int
    lift_height_m: float
    fallback_reason: str | None


class BimanualMission:
    """Own phase progression; normal progress requires an accepted command."""

    def __init__(self, cfg: BimanualMissionCfg | None = None) -> None:
        self.cfg = BimanualMissionCfg() if cfg is None else cfg
        if not isinstance(self.cfg, BimanualMissionCfg):
            raise TypeError("cfg must be BimanualMissionCfg")
        self.phase = BimanualPhase.APPROACH
        self._step = 0
        self._phase_steps = 0
        self._dwell_steps = 0
        self._consecutive_failures = 0
        self._initial_box_height: float | None = None
        self._fallback_reason: str | None = None

    def _transition(self, phase: BimanualPhase) -> None:
        self.phase = phase
        self._phase_steps = 0
        self._dwell_steps = 0

    def _supported(self, diagnostics: BimanualMissionDiagnostics) -> bool:
        return diagnostics.box_supported and (
            diagnostics.box_twist_norm <= self.cfg.supported_twist_tolerance
        )

    def _enter_safe(self, diagnostics: BimanualMissionDiagnostics, reason: str) -> None:
        self._fallback_reason = reason
        if self._supported(diagnostics):
            self._transition(BimanualPhase.SAFE_RELEASE)
        else:
            self._transition(BimanualPhase.HOLD_SAFE)

    def _critical_reason(
        self, diagnostics: BimanualMissionDiagnostics
    ) -> str | None:
        if diagnostics.relative_palm_slip_m > self.cfg.max_relative_palm_slip_m:
            return "palm_slip"
        if diagnostics.collision_margin_m < 0.0:
            return "collision_margin"
        if diagnostics.support_margin_m < 0.0:
            return "balance_margin"
        if self._consecutive_failures >= self.cfg.max_consecutive_failures:
            return diagnostics.subsystem_failure or "repeated_infeasibility"
        return None

    def update(
        self,
        snapshot: BimanualSnapshot,
        diagnostics: BimanualMissionDiagnostics,
    ) -> BimanualMissionState:
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if not isinstance(diagnostics, BimanualMissionDiagnostics):
            raise TypeError("diagnostics must be BimanualMissionDiagnostics")
        if self._initial_box_height is None:
            self._initial_box_height = float(snapshot.box.pose_b[2].item())
        self._step += 1
        self._phase_steps += 1
        if diagnostics.subsystem_failure is not None or not diagnostics.command_accepted:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        if self.phase not in {
            BimanualPhase.HOLD_SAFE,
            BimanualPhase.LOWER_SAFE,
            BimanualPhase.SAFE_RELEASE,
            BimanualPhase.TERMINATED,
        }:
            reason = self._critical_reason(diagnostics)
            if reason is not None:
                self._enter_safe(diagnostics, reason)

        if self.phase is BimanualPhase.HOLD_SAFE:
            if self._phase_steps >= self.cfg.safe_hold_steps:
                self._transition(BimanualPhase.LOWER_SAFE)
        elif self.phase is BimanualPhase.LOWER_SAFE:
            if self._supported(diagnostics):
                self._transition(BimanualPhase.SAFE_RELEASE)
        elif self.phase is BimanualPhase.SAFE_RELEASE:
            if self._supported(diagnostics) and diagnostics.hands_open:
                self._transition(BimanualPhase.TERMINATED)
        elif self.phase not in {BimanualPhase.DONE, BimanualPhase.TERMINATED} and diagnostics.command_accepted:
            if self.phase is BimanualPhase.APPROACH:
                self._dwell_steps = self._dwell_steps + 1 if diagnostics.palms_reached else 0
                if self._dwell_steps >= self.cfg.approach_dwell_steps:
                    self._transition(BimanualPhase.PRELOAD)
            elif self.phase is BimanualPhase.PRELOAD:
                self._dwell_steps = self._dwell_steps + 1 if diagnostics.bilateral_contact else 0
                if self._dwell_steps >= self.cfg.preload_dwell_steps:
                    self._transition(BimanualPhase.GRASP)
            elif self.phase is BimanualPhase.GRASP:
                closed = diagnostics.bilateral_contact and diagnostics.force_closure_margin > 0.0
                self._dwell_steps = self._dwell_steps + 1 if closed else 0
                if self._dwell_steps >= self.cfg.grasp_dwell_steps:
                    self._transition(BimanualPhase.LIFT)
            elif self.phase is BimanualPhase.LIFT:
                if self.lift_height(snapshot) >= self.cfg.lift_height_m - 1.0e-9:
                    self._transition(BimanualPhase.HOLD)
            elif self.phase is BimanualPhase.HOLD:
                if self._phase_steps >= self.cfg.hold_steps:
                    self._transition(BimanualPhase.LOWER)
            elif self.phase is BimanualPhase.LOWER:
                if self._supported(diagnostics):
                    self._transition(BimanualPhase.RELEASE)
            elif self.phase is BimanualPhase.RELEASE:
                if self._supported(diagnostics) and diagnostics.hands_open:
                    self._transition(BimanualPhase.DONE)
        return BimanualMissionState(
            phase=self.phase,
            step=self._step,
            phase_steps=self._phase_steps,
            consecutive_failures=self._consecutive_failures,
            lift_height_m=self.lift_height(snapshot),
            fallback_reason=self._fallback_reason,
        )

    def lift_height(self, snapshot: BimanualSnapshot) -> float:
        if self._initial_box_height is None:
            return 0.0
        return float(snapshot.box.pose_b[2].item()) - self._initial_box_height
