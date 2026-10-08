"""Pure, fail-closed acceptance checks for bilateral wire contact."""

from __future__ import annotations

from dataclasses import dataclass
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
    """One deterministic simulation contact observation for both O6 hands."""

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

    def __post_init__(self) -> None:
        for name in ("min_contact_force_n", "max_endpoint_slip_m"):
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
    return mask & (torch.linalg.vector_norm(forces_b, dim=1) >= threshold_n)


def _endpoint_slip_m(
    fingertip_positions_b: torch.Tensor,
    qualifying_mask: torch.Tensor,
    endpoint_position_b: torch.Tensor,
) -> float:
    if not bool(qualifying_mask.any()):
        return 0.0
    contact_center_b = fingertip_positions_b[qualifying_mask].mean(dim=0)
    return float(torch.linalg.vector_norm(contact_center_b - endpoint_position_b).item())


def evaluate_wire_contact(
    observation: WireContactObservation,
    config: WireAcceptanceConfig,
) -> WireContactEvaluation:
    """Evaluate one observation without mutating it or retaining any history.

    Endpoint slip is the distance from each hand's qualifying fingertip centre
    to its assigned wire endpoint (left endpoint first, right endpoint second).
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
        observation.left_fingertip_positions_b,
        left_qualifying,
        observation.wire_endpoint_positions_b[0],
    )
    right_slip = _endpoint_slip_m(
        observation.right_fingertip_positions_b,
        right_qualifying,
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
    "WireContactEvaluation",
    "WireContactObservation",
    "evaluate_wire_contact",
]
