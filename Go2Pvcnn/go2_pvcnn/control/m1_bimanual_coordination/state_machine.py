"""Deterministic normal and safe phase transitions for bimanual manipulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import BimanualPhase, BimanualSnapshot
from .grasp_goal import BimanualGraspGoal


@dataclass(frozen=True)
class BimanualMissionCfg:
    physics_dt: float = 0.005
    approach_dwell_steps: int = 4
    preload_dwell_steps: int = 20
    grasp_dwell_steps: int = 20
    hold_duration_s: float = 3.0
    safe_hold_steps: int = 20
    max_consecutive_failures: int = 3
    palm_position_tolerance_m: float = 0.03
    lift_height_m: float = 0.10
    max_relative_palm_slip_m: float = 0.005
    supported_twist_tolerance: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "physics_dt",
            "palm_position_tolerance_m",
            "lift_height_m",
            "max_relative_palm_slip_m",
            "supported_twist_tolerance",
            "hold_duration_s",
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
    left_palm_reached: bool
    right_palm_reached: bool
    left_contact: bool
    right_contact: bool
    contact_consistent: bool
    bilateral_contact: bool
    force_closure_margin: float
    relative_palm_slip_m: float
    box_twist_norm: float
    box_supported: bool
    hands_open: bool
    collision_margin_m: float
    support_margin_m: float
    subsystem_failure: str | None
    # These measurements are optional for the legacy Box path.  Catalog goals
    # use them to make the clamp/lift gate explicit instead of treating a
    # contact bit as proof of a successful grasp.
    left_contact_count: int = 0
    right_contact_count: int = 0
    left_normal_force_n: float = 0.0
    right_normal_force_n: float = 0.0
    left_normal_alignment: float = 0.0
    right_normal_alignment: float = 0.0
    relative_palm_slip_speed_m_s: float = 0.0
    vertical_force_n: float = 0.0
    object_tilt_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "command_accepted",
            "palms_reached",
            "left_palm_reached",
            "right_palm_reached",
            "left_contact",
            "right_contact",
            "contact_consistent",
            "bilateral_contact",
            "box_supported",
            "hands_open",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "force_closure_margin",
            "relative_palm_slip_m",
            "relative_palm_slip_speed_m_s",
            "box_twist_norm",
            "collision_margin_m",
            "support_margin_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if (
            self.relative_palm_slip_m < 0.0
            or self.relative_palm_slip_speed_m_s < 0.0
            or self.box_twist_norm < 0.0
        ):
            raise ValueError("slip and box twist norms must be non-negative")
        if self.subsystem_failure is not None and (
            not isinstance(self.subsystem_failure, str)
            or not self.subsystem_failure.strip()
        ):
            raise ValueError("subsystem_failure must be None or a non-empty string")
        for name in ("left_contact_count", "right_contact_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "left_normal_force_n",
            "right_normal_force_n",
            "left_normal_alignment",
            "right_normal_alignment",
            "vertical_force_n",
            "object_tilt_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


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

    def __init__(
        self,
        cfg: BimanualMissionCfg | None = None,
        *,
        grasp_goal: BimanualGraspGoal | None = None,
    ) -> None:
        self.cfg = BimanualMissionCfg() if cfg is None else cfg
        if not isinstance(self.cfg, BimanualMissionCfg):
            raise TypeError("cfg must be BimanualMissionCfg")
        if grasp_goal is not None and not isinstance(grasp_goal, BimanualGraspGoal):
            raise TypeError("grasp_goal must be BimanualGraspGoal or None")
        self._grasp_goal = grasp_goal
        self.phase = BimanualPhase.APPROACH
        self._step = 0
        self._phase_steps = 0
        self._dwell_steps = 0
        self._approach_side_dwell = [0, 0]
        self._approach_side_ready = [False, False]
        self._consecutive_failures = 0
        self._initial_box_height: float | None = None
        self._fallback_reason: str | None = None
        self._previous_timestamp_ns: int | None = None
        self._hold_elapsed_ns = 0

    @property
    def grasp_goal(self) -> BimanualGraspGoal | None:
        return self._grasp_goal

    def set_grasp_goal(self, grasp_goal: BimanualGraspGoal | None) -> None:
        """Update the immutable catalog criteria used by subsequent samples."""

        if grasp_goal is not None and not isinstance(grasp_goal, BimanualGraspGoal):
            raise TypeError("grasp_goal must be BimanualGraspGoal or None")
        self._grasp_goal = grasp_goal

    def _transition(self, phase: BimanualPhase) -> None:
        self.phase = phase
        self._phase_steps = 0
        self._dwell_steps = 0
        if phase is BimanualPhase.HOLD:
            self._hold_elapsed_ns = 0

    def _supported(self, diagnostics: BimanualMissionDiagnostics) -> bool:
        return diagnostics.box_supported and (
            diagnostics.box_twist_norm <= self.cfg.supported_twist_tolerance
        )

    def _clamp_ready(self, diagnostics: BimanualMissionDiagnostics) -> bool:
        """Return the contact/force gate for a catalog target.

        The legacy Box path intentionally retains its historical force-closure
        gate.  Catalog goals additionally require the configured minimum
        contact count and per-hand normal force.
        """

        if self._grasp_goal is None:
            return (
                diagnostics.left_contact
                and diagnostics.right_contact
                and diagnostics.contact_consistent
                and diagnostics.bilateral_contact
                and diagnostics.force_closure_margin > 0.0
                and diagnostics.relative_palm_slip_m <= self.cfg.max_relative_palm_slip_m
            )
        criteria = self._grasp_goal.clamp_criteria
        return (
            diagnostics.left_contact
            and diagnostics.right_contact
            and diagnostics.contact_consistent
            and diagnostics.bilateral_contact
            and diagnostics.force_closure_margin > 0.0
            and diagnostics.left_contact_count >= criteria.min_contact_count_per_hand
            and diagnostics.right_contact_count >= criteria.min_contact_count_per_hand
            and diagnostics.left_normal_force_n >= criteria.min_normal_force_n
            and diagnostics.right_normal_force_n >= criteria.min_normal_force_n
            and diagnostics.left_normal_force_n <= criteria.max_normal_force_n
            and diagnostics.right_normal_force_n <= criteria.max_normal_force_n
            and diagnostics.left_normal_alignment >= criteria.min_normal_alignment
            and diagnostics.right_normal_alignment >= criteria.min_normal_alignment
            and diagnostics.relative_palm_slip_speed_m_s
            <= criteria.max_slip_speed_m_s
        )

    def _lift_ready(
        self, snapshot: BimanualSnapshot, diagnostics: BimanualMissionDiagnostics
    ) -> bool:
        if self._grasp_goal is None:
            return self.lift_height(snapshot) >= self.cfg.lift_height_m - 1.0e-9
        criteria = self._grasp_goal.lift_criteria
        return (
            self.lift_height(snapshot) >= criteria.height_m - 1.0e-9
            and self._lift_measurements_ready(diagnostics)
            and self._clamp_ready(diagnostics)
        )

    def _lift_measurements_ready(
        self, diagnostics: BimanualMissionDiagnostics
    ) -> bool:
        """Check catalog lift force/tilt health independently of clamp health."""

        if self._grasp_goal is None:
            return True
        criteria = self._grasp_goal.lift_criteria
        return (
            diagnostics.vertical_force_n >= criteria.min_vertical_force_n
            and diagnostics.object_tilt_rad <= criteria.max_tilt_rad
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
        slip_exceeded = (
            diagnostics.relative_palm_slip_speed_m_s
            > self._grasp_goal.clamp_criteria.max_slip_speed_m_s
            if self._grasp_goal is not None
            else diagnostics.relative_palm_slip_m > self.cfg.max_relative_palm_slip_m
        )
        if slip_exceeded and not (
            self.phase in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD}
            and diagnostics.box_supported
        ):
            return "palm_slip"
        if diagnostics.collision_margin_m < 0.0:
            return "collision_margin"
        if diagnostics.support_margin_m < 0.0:
            return "balance_margin"
        if self._grasp_goal is not None and self.phase in {
            BimanualPhase.LIFT,
            BimanualPhase.HOLD,
        }:
            if self.phase is BimanualPhase.HOLD and not self._lift_measurements_ready(
                diagnostics
            ):
                return "lift_criteria_failed"
            if not self._clamp_ready(diagnostics):
                return "clamp_criteria_failed"
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
        elapsed_ns = 0
        if self._previous_timestamp_ns is not None:
            elapsed_ns = max(0, snapshot.timestamp_ns - self._previous_timestamp_ns)
        if self._previous_timestamp_ns is None or snapshot.timestamp_ns > self._previous_timestamp_ns:
            self._previous_timestamp_ns = snapshot.timestamp_ns
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
                for index, reached in enumerate(
                    (diagnostics.left_palm_reached, diagnostics.right_palm_reached)
                ):
                    self._approach_side_dwell[index] = (
                        self._approach_side_dwell[index] + 1 if reached else 0
                    )
                    if self._approach_side_dwell[index] >= self.cfg.approach_dwell_steps:
                        self._approach_side_ready[index] = True
                if all(self._approach_side_ready):
                    self._transition(BimanualPhase.PRELOAD)
            elif self.phase is BimanualPhase.PRELOAD:
                contact_ready = self._clamp_ready(diagnostics)
                self._dwell_steps = self._dwell_steps + 1 if contact_ready else 0
                if self._dwell_steps >= self.cfg.preload_dwell_steps:
                    self._transition(BimanualPhase.GRASP)
            elif self.phase is BimanualPhase.GRASP:
                closed = self._clamp_ready(diagnostics)
                self._dwell_steps = self._dwell_steps + 1 if closed else 0
                if self._dwell_steps >= self.cfg.grasp_dwell_steps:
                    self._transition(BimanualPhase.LIFT)
            elif self.phase is BimanualPhase.LIFT:
                if self._lift_ready(snapshot, diagnostics):
                    self._transition(BimanualPhase.HOLD)
            elif self.phase is BimanualPhase.HOLD:
                stable_grasp = self._clamp_ready(diagnostics)
                self._hold_elapsed_ns = (
                    self._hold_elapsed_ns + elapsed_ns if stable_grasp else 0
                )
                hold_time_s = (
                    self.cfg.hold_duration_s
                    if self._grasp_goal is None
                    else self._grasp_goal.lift_criteria.hold_time_s
                )
                if self._hold_elapsed_ns >= round(hold_time_s * 1e9):
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
