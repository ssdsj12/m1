"""Pure, fail-closed acceptance checks for bilateral wire contact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch


_FINGERTIP_COUNT = 5
_FORCE_SHAPE = (_FINGERTIP_COUNT, 3)
_POSITION_SHAPE = (3,)
_ENDPOINT_SHAPE = (2, 3)


def _float64_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on device cpu; got {value.device}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have exact shape {shape}; got {tuple(value.shape)}")
    return value.clone()


def _bool_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on device cpu; got {value.device}")
    if tuple(value.shape) != (_FINGERTIP_COUNT,):
        raise ValueError(
            f"{name} must have exact shape ({_FINGERTIP_COUNT},); got {tuple(value.shape)}"
        )
    return value.clone()


def _real_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error


@dataclass(frozen=True)
class WireContactObservation:
    """One deterministic simulation contact observation for both O6 hands.

    Public caller obligation: ``*_contact_mask`` and ``*_fingertip_forces_b``
    must already represent physical, selected-wire contacts filtered to the
    target wire. This evaluator does not perform contact-pair filtering or
    infer contact from fingertip proximity.
    """

    left_contact_mask: torch.Tensor
    right_contact_mask: torch.Tensor
    left_fingertip_forces_b: torch.Tensor
    right_fingertip_forces_b: torch.Tensor
    left_fingertip_positions_b: torch.Tensor
    right_fingertip_positions_b: torch.Tensor
    left_palm_position_b: torch.Tensor
    right_palm_position_b: torch.Tensor
    wire_endpoint_positions_b: torch.Tensor
    dt_s: float

    def __post_init__(self) -> None:
        for name in ("left_contact_mask", "right_contact_mask"):
            object.__setattr__(self, name, _bool_tensor(name, getattr(self, name)))
        for name in ("left_fingertip_forces_b", "right_fingertip_forces_b"):
            object.__setattr__(
                self, name, _float64_tensor(name, getattr(self, name), _FORCE_SHAPE)
            )
        for name in ("left_fingertip_positions_b", "right_fingertip_positions_b"):
            object.__setattr__(
                self, name, _float64_tensor(name, getattr(self, name), _FORCE_SHAPE)
            )
        for name in ("left_palm_position_b", "right_palm_position_b"):
            object.__setattr__(
                self, name, _float64_tensor(name, getattr(self, name), _POSITION_SHAPE)
            )
        object.__setattr__(
            self,
            "wire_endpoint_positions_b",
            _float64_tensor(
                "wire_endpoint_positions_b", self.wire_endpoint_positions_b, _ENDPOINT_SHAPE
            ),
        )
        object.__setattr__(self, "dt_s", _real_scalar("dt_s", self.dt_s))


@dataclass(frozen=True)
class WireAcceptanceConfig:
    """Explicit thresholds used by :func:`evaluate_wire_contact`."""

    min_contact_force_n: float = 0.2
    min_contacts_per_hand: int = 2
    max_endpoint_slip_m: float = 0.02
    required_bilateral_contact_s: float = 0.5
    required_lift_m: float = 0.05
    required_hold_s: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "min_contact_force_n",
            "max_endpoint_slip_m",
            "required_bilateral_contact_s",
            "required_lift_m",
            "required_hold_s",
        ):
            value = _real_scalar(name, getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if isinstance(self.min_contacts_per_hand, bool) or not isinstance(
            self.min_contacts_per_hand, int
        ):
            raise TypeError("min_contacts_per_hand must be an integer")
        if not 1 <= self.min_contacts_per_hand <= _FINGERTIP_COUNT:
            raise ValueError(
                f"min_contacts_per_hand must be in [1, {_FINGERTIP_COUNT}]"
            )


@dataclass(frozen=True)
class WireContactEvaluation:
    left_valid_contacts: int
    right_valid_contacts: int
    left_slip_m: float
    right_slip_m: float
    bilateral_contact: bool
    finite: bool
    failure_reason: str | None


class WireAcceptancePhase(str, Enum):
    """Deterministic lifecycle for a physically evidenced wire grasp."""

    CONTACT = "contact"
    LIFT = "lift"
    HOLD = "hold"
    SUCCEEDED = "succeeded"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class WireAcceptanceState:
    """Immutable acceptance result for the most recent simulation tick."""

    phase: WireAcceptancePhase
    bilateral_contact_elapsed_s: float
    lift_elapsed_s: float
    hold_elapsed_s: float
    lift_permitted: bool
    request_reclamp: bool
    success: bool
    terminal_reason: str | None


class WireAcceptanceTracker:
    """Accumulate real-contact grasp/lift acceptance in simulation time only."""

    def __init__(self, config: WireAcceptanceConfig) -> None:
        if not isinstance(config, WireAcceptanceConfig):
            raise TypeError("config must be WireAcceptanceConfig")
        self.config = config
        self._reset_center_height_m = 0.0
        self.reset(wire_center_height_m=0.0)

    @property
    def state(self) -> WireAcceptanceState:
        return self._state

    def reset(self, *, wire_center_height_m: float) -> None:
        """Clear all episode state and establish the next lift-height baseline."""

        self._reset_center_height_m = self._finite_scalar(
            "wire_center_height_m", wire_center_height_m
        )
        self._state = WireAcceptanceState(
            phase=WireAcceptancePhase.CONTACT,
            bilateral_contact_elapsed_s=0.0,
            lift_elapsed_s=0.0,
            hold_elapsed_s=0.0,
            lift_permitted=False,
            request_reclamp=False,
            success=False,
            terminal_reason=None,
        )

    def update(
        self,
        evaluation: WireContactEvaluation,
        *,
        wire_center_height_m: float,
        dt_s: float,
        safety_feasible: bool,
    ) -> WireAcceptanceState:
        """Advance one accepted-control interval without consulting wall-clock time."""

        if not isinstance(evaluation, WireContactEvaluation):
            raise TypeError("evaluation must be WireContactEvaluation")
        center_height_m = self._finite_scalar(
            "wire_center_height_m", wire_center_height_m
        )
        elapsed_s = self._positive_scalar("dt_s", dt_s)
        if not isinstance(safety_feasible, bool):
            raise TypeError("safety_feasible must be a bool")

        state = self._state
        if state.phase in (
            WireAcceptancePhase.SUCCEEDED,
            WireAcceptancePhase.TERMINATED,
        ):
            return state
        if not safety_feasible:
            return self._terminate("safety_rejected")
        if not evaluation.finite:
            return self._terminate("non_finite_contact_observation")
        if evaluation.failure_reason is not None and evaluation.failure_reason.startswith(
            "endpoint_slip_exceeds_"
        ):
            return self._terminate("excessive_endpoint_slip")
        if not evaluation.bilateral_contact:
            return self._handle_contact_loss(state)

        contact_elapsed_s = state.bilateral_contact_elapsed_s + elapsed_s
        if state.phase is WireAcceptancePhase.CONTACT:
            if not self._duration_reached(
                contact_elapsed_s, self.config.required_bilateral_contact_s
            ):
                return self._set_state(
                    phase=WireAcceptancePhase.CONTACT,
                    bilateral_contact_elapsed_s=contact_elapsed_s,
                )
            return self._set_state(
                phase=WireAcceptancePhase.LIFT,
                bilateral_contact_elapsed_s=contact_elapsed_s,
                lift_permitted=True,
            )

        if state.phase is WireAcceptancePhase.LIFT:
            lift_elapsed_s = state.lift_elapsed_s + elapsed_s
            if self._lifted(center_height_m):
                return self._set_state(
                    phase=WireAcceptancePhase.HOLD,
                    bilateral_contact_elapsed_s=contact_elapsed_s,
                    lift_elapsed_s=lift_elapsed_s,
                    hold_elapsed_s=elapsed_s,
                    lift_permitted=True,
                )
            return self._set_state(
                phase=WireAcceptancePhase.LIFT,
                bilateral_contact_elapsed_s=contact_elapsed_s,
                lift_elapsed_s=lift_elapsed_s,
                lift_permitted=True,
            )

        if state.phase is WireAcceptancePhase.HOLD:
            if not self._lifted(center_height_m):
                return self._set_state(
                    phase=WireAcceptancePhase.LIFT,
                    bilateral_contact_elapsed_s=contact_elapsed_s,
                    lift_elapsed_s=state.lift_elapsed_s + elapsed_s,
                    hold_elapsed_s=0.0,
                    lift_permitted=True,
                )
            hold_elapsed_s = state.hold_elapsed_s + elapsed_s
            if self._duration_reached(hold_elapsed_s, self.config.required_hold_s):
                return self._set_state(
                    phase=WireAcceptancePhase.SUCCEEDED,
                    bilateral_contact_elapsed_s=contact_elapsed_s,
                    lift_elapsed_s=state.lift_elapsed_s,
                    hold_elapsed_s=hold_elapsed_s,
                    lift_permitted=True,
                    success=True,
                )
            return self._set_state(
                phase=WireAcceptancePhase.HOLD,
                bilateral_contact_elapsed_s=contact_elapsed_s,
                lift_elapsed_s=state.lift_elapsed_s,
                hold_elapsed_s=hold_elapsed_s,
                lift_permitted=True,
            )
        raise RuntimeError(f"unsupported wire acceptance phase {state.phase!r}")

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        scalar = _real_scalar(name, value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar

    @classmethod
    def _positive_scalar(cls, name: str, value: float) -> float:
        scalar = cls._finite_scalar(name, value)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive")
        return scalar

    def _lifted(self, center_height_m: float) -> bool:
        return (
            center_height_m - self._reset_center_height_m
            >= self.config.required_lift_m
        )

    @staticmethod
    def _duration_reached(elapsed_s: float, required_s: float) -> bool:
        return elapsed_s > required_s or math.isclose(
            elapsed_s, required_s, rel_tol=0.0, abs_tol=1.0e-12
        )

    def _handle_contact_loss(self, state: WireAcceptanceState) -> WireAcceptanceState:
        if state.phase in (WireAcceptancePhase.LIFT, WireAcceptancePhase.HOLD):
            return self._set_state(
                phase=WireAcceptancePhase.CONTACT,
                request_reclamp=True,
            )
        return self._set_state(phase=WireAcceptancePhase.CONTACT)

    def _terminate(self, reason: str) -> WireAcceptanceState:
        return self._set_state(
            phase=WireAcceptancePhase.TERMINATED,
            terminal_reason=reason,
        )

    def _set_state(self, **changes: object) -> WireAcceptanceState:
        values = {
            "phase": WireAcceptancePhase.CONTACT,
            "bilateral_contact_elapsed_s": 0.0,
            "lift_elapsed_s": 0.0,
            "hold_elapsed_s": 0.0,
            "lift_permitted": False,
            "request_reclamp": False,
            "success": False,
            "terminal_reason": None,
        }
        values.update(changes)
        self._state = WireAcceptanceState(**values)
        return self._state


def _observation_is_finite(observation: WireContactObservation) -> bool:
    tensors = (
        observation.left_fingertip_forces_b,
        observation.right_fingertip_forces_b,
        observation.left_fingertip_positions_b,
        observation.right_fingertip_positions_b,
        observation.left_palm_position_b,
        observation.right_palm_position_b,
        observation.wire_endpoint_positions_b,
    )
    return math.isfinite(observation.dt_s) and all(
        bool(torch.isfinite(value).all()) for value in tensors
    )


def _qualifying_contact_mask(
    mask: torch.Tensor, forces_b: torch.Tensor, threshold_n: float
) -> torch.Tensor:
    return mask & (torch.linalg.vector_norm(forces_b, dim=1) > threshold_n)


def _endpoint_slip_m(
    palm_position_b: torch.Tensor,
    endpoint_position_b: torch.Tensor,
) -> float:
    return float(torch.linalg.vector_norm(palm_position_b - endpoint_position_b).item())


def evaluate_wire_contact(
    observation: WireContactObservation,
    config: WireAcceptanceConfig,
) -> WireContactEvaluation:
    """Evaluate one observation without mutating it or retaining any history.

    Endpoint slip is the distance from each palm to its assigned wire endpoint
    (left endpoint first, right endpoint second).
    """

    if not isinstance(observation, WireContactObservation):
        raise TypeError("observation must be WireContactObservation")
    if not isinstance(config, WireAcceptanceConfig):
        raise TypeError("config must be WireAcceptanceConfig")

    finite = _observation_is_finite(observation)
    if not finite:
        return WireContactEvaluation(
            left_valid_contacts=0,
            right_valid_contacts=0,
            left_slip_m=0.0,
            right_slip_m=0.0,
            bilateral_contact=False,
            finite=False,
            failure_reason="non_finite_contact_observation",
        )

    left_qualifying = _qualifying_contact_mask(
        observation.left_contact_mask,
        observation.left_fingertip_forces_b,
        config.min_contact_force_n,
    )
    right_qualifying = _qualifying_contact_mask(
        observation.right_contact_mask,
        observation.right_fingertip_forces_b,
        config.min_contact_force_n,
    )
    left_valid = int(left_qualifying.sum().item())
    right_valid = int(right_qualifying.sum().item())
    left_slip = _endpoint_slip_m(
        observation.left_palm_position_b,
        observation.wire_endpoint_positions_b[0],
    )
    right_slip = _endpoint_slip_m(
        observation.right_palm_position_b,
        observation.wire_endpoint_positions_b[1],
    )

    if (
        left_valid < config.min_contacts_per_hand
        or right_valid < config.min_contacts_per_hand
    ):
        reason = "insufficient_valid_contacts"
    elif (
        left_slip > config.max_endpoint_slip_m
        or right_slip > config.max_endpoint_slip_m
    ):
        reason = f"endpoint_slip_exceeds_{config.max_endpoint_slip_m:g}_m"
    else:
        reason = None
    return WireContactEvaluation(
        left_valid_contacts=left_valid,
        right_valid_contacts=right_valid,
        left_slip_m=left_slip,
        right_slip_m=right_slip,
        bilateral_contact=reason is None,
        finite=True,
        failure_reason=reason,
    )


__all__ = [
    "WireAcceptanceConfig",
    "WireAcceptancePhase",
    "WireAcceptanceState",
    "WireAcceptanceTracker",
    "WireContactEvaluation",
    "WireContactObservation",
    "evaluate_wire_contact",
]
