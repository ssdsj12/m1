"""Balance-first degradation state machine for the C0 deterministic Teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from .contracts import PANDA_ARM_JOINT_NAMES, require_tensor


PANDA_ARM_DOF = len(PANDA_ARM_JOINT_NAMES)


class SafetyState(IntEnum):
    TRACK = 0
    SCALE = 1
    HOLD = 2
    RETRACT = 3
    TERMINATE = 4


def residual_scale_for_safety(
    state: SafetyState, scale_factor: float
) -> float:
    """Project all residual channels through the current balance state."""

    if not isinstance(state, SafetyState):
        raise TypeError("state must be a SafetyState")
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, (int, float))
        or not math.isfinite(float(scale_factor))
        or not 0.0 <= float(scale_factor) <= 1.0
    ):
        raise ValueError("scale_factor must be finite and in [0,1]")
    if state is SafetyState.TRACK:
        return 1.0
    if state is SafetyState.SCALE:
        return float(scale_factor)
    return 0.0


@dataclass(frozen=True)
class SafetyCfg:
    warning_angle_rad: float = math.radians(7.0)
    critical_angle_rad: float = math.radians(10.0)
    required_wheel_contacts: int = 4
    max_lateral_slip: float = 0.05
    max_rolling_residual: float = 0.05
    retract_base_speed: float = 0.02
    unsafe_samples_to_advance: int = 2
    safe_samples_to_recover: int = 20
    scaled_twist_factor: float = 0.5
    base_scaled_velocity_factor: float = 0.5
    retract_rate_rad_per_step: float = 0.02


@dataclass(frozen=True)
class SafetyDecision:
    state: SafetyState
    twist_scale: float
    base_velocity_scale: float
    arm_target: torch.Tensor
    stop_wheels: bool
    terminate: bool
    reason: str


class BalanceSafetySupervisor:
    """Escalate balance failures and recover conservatively with hysteresis."""

    def __init__(self, cfg: SafetyCfg, safe_arm_target: torch.Tensor):
        if not isinstance(cfg, SafetyCfg):
            raise TypeError("cfg must be a SafetyCfg")
        require_tensor(
            "safe_arm_target", safe_arm_target, trailing_shape=(PANDA_ARM_DOF,)
        )
        if safe_arm_target.ndim != 1 or not safe_arm_target.is_floating_point():
            raise ValueError("safe_arm_target must be one floating 7-vector")
        self.cfg = cfg
        self._safe_arm_target = safe_arm_target.detach().clone()
        self._state = SafetyState.TRACK
        self._unsafe_count = 0
        self._safe_count = 0
        self._last_finite_target = self._safe_arm_target.clone()
        self._hold_target = self._safe_arm_target.clone()
        self._retract_target = self._safe_arm_target.clone()

    @property
    def state(self) -> SafetyState:
        return self._state

    def _validate_arm_target_contract(self, target: torch.Tensor) -> None:
        if not isinstance(target, torch.Tensor):
            raise TypeError("current_arm_target must be a torch.Tensor")
        if target.shape != (PANDA_ARM_DOF,):
            raise ValueError("current_arm_target must end with shape (7,)")
        if target.dtype != self._safe_arm_target.dtype:
            raise TypeError("current_arm_target dtype must match safe_arm_target")
        if target.device != self._safe_arm_target.device:
            raise ValueError("current_arm_target device must match safe_arm_target")

    def reset(self, current_arm_target: torch.Tensor | None = None) -> None:
        target = self._safe_arm_target if current_arm_target is None else current_arm_target
        self._validate_arm_target_contract(target)
        require_tensor(
            "current_arm_target", target, trailing_shape=(PANDA_ARM_DOF,)
        )
        self._state = SafetyState.TRACK
        self._unsafe_count = 0
        self._safe_count = 0
        self._last_finite_target = target.detach().clone()
        self._hold_target = target.detach().clone()
        self._retract_target = target.detach().clone()

    def _advance(self) -> None:
        if self._state == SafetyState.TERMINATE:
            return
        self._state = SafetyState(self._state + 1)
        if self._state == SafetyState.HOLD:
            self._hold_target = self._last_finite_target.clone()
            self._retract_target = self._hold_target.clone()

    def _recover(self) -> None:
        if self._state in (SafetyState.TRACK, SafetyState.TERMINATE):
            return
        if self._state == SafetyState.RETRACT:
            self._hold_target = self._retract_target.clone()
        self._state = SafetyState(self._state - 1)

    def _step_retract(self) -> torch.Tensor:
        delta = self._safe_arm_target - self._retract_target
        step = torch.clamp(
            delta,
            min=-self.cfg.retract_rate_rad_per_step,
            max=self.cfg.retract_rate_rad_per_step,
        )
        self._retract_target = self._retract_target + step
        return self._retract_target.clone()

    def _decision(self, current_arm_target: torch.Tensor, reason: str) -> SafetyDecision:
        if self._state in (SafetyState.TRACK, SafetyState.SCALE):
            arm_target = current_arm_target.detach().clone()
        elif self._state == SafetyState.HOLD:
            arm_target = self._hold_target.clone()
        elif self._state == SafetyState.RETRACT:
            arm_target = self._step_retract()
        else:
            arm_target = self._retract_target.clone()

        if self._state == SafetyState.TRACK:
            twist_scale = 1.0
            base_velocity_scale = 1.0
        elif self._state == SafetyState.SCALE:
            twist_scale = self.cfg.scaled_twist_factor
            base_velocity_scale = self.cfg.base_scaled_velocity_factor
        else:
            twist_scale = 0.0
            base_velocity_scale = 0.0
        return SafetyDecision(
            state=self._state,
            twist_scale=twist_scale,
            base_velocity_scale=base_velocity_scale,
            arm_target=arm_target,
            stop_wheels=self._state >= SafetyState.HOLD,
            terminate=self._state == SafetyState.TERMINATE,
            reason=reason,
        )

    def update(
        self,
        *,
        roll: float,
        pitch: float,
        wheel_contact_count: int,
        max_lateral_slip: float,
        qp_success: bool,
        signals_finite: bool,
        current_arm_target: torch.Tensor,
        max_rolling_residual: float = 0.0,
        base_speed: float = 0.0,
    ) -> SafetyDecision:
        self._validate_arm_target_contract(current_arm_target)
        scalar_finite = all(
            math.isfinite(float(value))
            for value in (
                roll,
                pitch,
                max_lateral_slip,
                max_rolling_residual,
                base_speed,
            )
        )
        target_finite = bool(torch.isfinite(current_arm_target).all().item())
        all_finite = bool(signals_finite) and scalar_finite and target_finite
        if not all_finite:
            self._state = SafetyState.TERMINATE
            self._unsafe_count = 0
            self._safe_count = 0
            self._retract_target = self._last_finite_target.clone()
            return self._decision(self._last_finite_target, "non_finite")

        if not isinstance(wheel_contact_count, int) or isinstance(
            wheel_contact_count, bool
        ):
            raise TypeError("wheel_contact_count must be an integer")
        if not isinstance(qp_success, bool):
            raise TypeError("qp_success must be a bool")
        if not isinstance(signals_finite, bool):
            raise TypeError("signals_finite must be a bool")

        if self._state <= SafetyState.SCALE:
            self._last_finite_target = current_arm_target.detach().clone()

        reason = "safe"
        unsafe = False
        if abs(float(roll)) >= self.cfg.critical_angle_rad or abs(
            float(pitch)
        ) >= self.cfg.critical_angle_rad:
            unsafe = True
            reason = "critical_orientation"
        elif abs(float(roll)) >= self.cfg.warning_angle_rad or abs(
            float(pitch)
        ) >= self.cfg.warning_angle_rad:
            unsafe = True
            reason = "warning_orientation"
        elif wheel_contact_count < self.cfg.required_wheel_contacts:
            unsafe = True
            reason = "contact_loss"
        elif abs(float(max_lateral_slip)) > self.cfg.max_lateral_slip:
            unsafe = True
            reason = "lateral_slip"
        elif abs(float(max_rolling_residual)) > self.cfg.max_rolling_residual:
            unsafe = True
            reason = "rolling_residual"
        elif not qp_success:
            unsafe = True
            reason = "qp_failure"

        if self._state == SafetyState.TERMINATE:
            return self._decision(self._last_finite_target, "terminal_latched")
        if unsafe:
            self._safe_count = 0
            self._unsafe_count += 1
            if self._unsafe_count >= self.cfg.unsafe_samples_to_advance:
                stopped_for_retract = (
                    self._state != SafetyState.HOLD
                    or abs(float(base_speed)) < self.cfg.retract_base_speed
                )
                if stopped_for_retract:
                    self._advance()
                self._unsafe_count = 0
        else:
            self._unsafe_count = 0
            self._safe_count += 1
            if self._safe_count >= self.cfg.safe_samples_to_recover:
                self._recover()
                self._safe_count = 0
        return self._decision(current_arm_target, reason)
