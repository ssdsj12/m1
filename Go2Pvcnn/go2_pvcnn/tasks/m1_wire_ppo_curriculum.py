"""Reward and approval gates for the two-stage wire PPO policy.

This module is intentionally independent of Isaac Lab.  Rollout adapters can
convert their diagnostics to :class:`WirePpoState`, while CPU tests and the
GPU environment share exactly the same acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections.abc import Mapping, Sequence
from typing import Any


class WirePpoStage(str, Enum):
    APPROVAL = "approval"
    ACTION = "action"


@dataclass(frozen=True)
class WirePpoRewardCfg:
    """Weights and physical thresholds shared by Stage A and Stage B."""

    contact_force_threshold: float = 0.2
    minimum_contacts_per_hand: int = 2
    minimum_lift_m: float = 0.05
    minimum_hold_s: float = 1.0
    maximum_slip_m: float = 0.02
    approach_weight: float = 1.0
    contact_reward: float = 2.0
    grasp_weight: float = 1.0
    lift_weight: float = 2.0
    hold_weight: float = 1.0
    safety_failure_penalty: float = -10.0
    action_centerline_weight: float = 1.0
    action_tangent_weight: float = 0.5
    action_deformation_weight: float = 0.5
    action_synchronization_weight: float = 0.5
    action_force_balance_weight: float = 0.5
    action_slip_weight: float = 1.0
    action_latent_smoothness_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.contact_force_threshold <= 0:
            raise ValueError("contact_force_threshold must be positive")
        if self.minimum_contacts_per_hand < 1:
            raise ValueError("minimum_contacts_per_hand must be positive")
        if self.minimum_lift_m < 0 or self.minimum_hold_s < 0 or self.maximum_slip_m < 0:
            raise ValueError("acceptance distances and durations must be non-negative")
        if self.safety_failure_penalty >= 0:
            raise ValueError("safety_failure_penalty must be negative")


@dataclass(frozen=True)
class WirePpoState:
    """Minimal finite diagnostic snapshot consumed by the PPO reward."""

    approach_error: float = 0.0
    contacts_left: int = 0
    contacts_right: int = 0
    contact_forces_left: tuple[float, ...] = ()
    contact_forces_right: tuple[float, ...] = ()
    lift_height: float = 0.0
    hold_time: float = 0.0
    slip: float = 0.0
    safety_failure: bool = False
    wire_centerline_error: float = 0.0
    tangent_error: float = 0.0
    deformation_error: float = 0.0
    synchronization_error: float = 0.0
    force_balance_error: float = 0.0
    latent_delta_norm: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WirePpoState":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown wire PPO state fields: {sorted(unknown)}")
        data = dict(value)
        for field in ("contact_forces_left", "contact_forces_right"):
            if field in data and not isinstance(data[field], tuple):
                data[field] = tuple(data[field])
        return cls(**data)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    approach: float = 0.0
    contact: float = 0.0
    grasp: float = 0.0
    lift: float = 0.0
    hold: float = 0.0
    safety: float = 0.0
    centerline: float = 0.0
    tangent: float = 0.0
    deformation: float = 0.0
    synchronization: float = 0.0
    force_balance: float = 0.0
    slip: float = 0.0
    latent_smoothness: float = 0.0

    def approval_terms(self) -> dict[str, float]:
        return {
            "approach": self.approach,
            "contact": self.contact,
            "grasp": self.grasp,
            "lift": self.lift,
            "hold": self.hold,
            "safety": self.safety,
        }

    def action_terms(self) -> dict[str, float]:
        return {
            **self.approval_terms(),
            "centerline": self.centerline,
            "tangent": self.tangent,
            "deformation": self.deformation,
            "synchronization": self.synchronization,
            "force_balance": self.force_balance,
            "slip": self.slip,
            "latent_smoothness": self.latent_smoothness,
        }


def _state(value: WirePpoState | Mapping[str, Any]) -> WirePpoState:
    if isinstance(value, WirePpoState):
        return value
    if isinstance(value, Mapping):
        return WirePpoState.from_mapping(value)
    raise TypeError("wire PPO state must be WirePpoState or a mapping")


def _finite_state(state: WirePpoState) -> None:
    scalar_names = (
        "approach_error", "lift_height", "hold_time", "slip",
        "wire_centerline_error", "tangent_error", "deformation_error",
        "synchronization_error", "force_balance_error", "latent_delta_norm",
    )
    for name in scalar_names:
        value = getattr(state, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if isinstance(state.safety_failure, bool) is False:
        raise TypeError("safety_failure must be a bool")
    for name in ("contacts_left", "contacts_right"):
        value = getattr(state, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("contact_forces_left", "contact_forces_right"):
        forces = getattr(state, name)
        if not isinstance(forces, Sequence) or isinstance(forces, (str, bytes)):
            raise TypeError(f"{name} must be a sequence")
        if any(not isinstance(force, (int, float)) or not math.isfinite(float(force)) for force in forces):
            raise ValueError(f"{name} must contain only finite forces")


def approval_episode(
    current: WirePpoState | Mapping[str, Any], cfg: WirePpoRewardCfg | None = None
) -> bool:
    """Return true only when a complete, safe bilateral approval episode exists."""

    cfg = cfg or WirePpoRewardCfg()
    state = _state(current)
    try:
        _finite_state(state)
    except (TypeError, ValueError):
        return False
    if state.safety_failure:
        return False
    if state.contacts_left < cfg.minimum_contacts_per_hand or state.contacts_right < cfg.minimum_contacts_per_hand:
        return False
    for count, forces in (
        (state.contacts_left, state.contact_forces_left),
        (state.contacts_right, state.contact_forces_right),
    ):
        if len(forces) < count or any(force <= cfg.contact_force_threshold for force in forces[:count]):
            return False
    return (
        state.lift_height >= cfg.minimum_lift_m
        and state.hold_time >= cfg.minimum_hold_s
        and state.slip <= cfg.maximum_slip_m
    )


def _progress(previous: float, current: float) -> float:
    return previous - current


def compute_wire_ppo_reward(
    previous: WirePpoState | Mapping[str, Any],
    current: WirePpoState | Mapping[str, Any],
    stage: WirePpoStage,
    cfg: WirePpoRewardCfg | None = None,
) -> RewardBreakdown:
    """Compute a deterministic reward without mutating either diagnostic snapshot."""

    if not isinstance(stage, WirePpoStage):
        raise ValueError("stage must be WirePpoStage")
    cfg = cfg or WirePpoRewardCfg()
    previous_state, current_state = _state(previous), _state(current)
    _finite_state(previous_state)
    _finite_state(current_state)
    contact = cfg.contact_reward * (
        min(current_state.contacts_left, cfg.minimum_contacts_per_hand)
        + min(current_state.contacts_right, cfg.minimum_contacts_per_hand)
    ) / (2.0 * cfg.minimum_contacts_per_hand)
    breakdown = RewardBreakdown(
        total=0.0,
        approach=cfg.approach_weight * _progress(previous_state.approach_error, current_state.approach_error),
        contact=contact,
        grasp=cfg.grasp_weight * max(0.0, min(current_state.contacts_left, current_state.contacts_right) / cfg.minimum_contacts_per_hand),
        lift=cfg.lift_weight * max(0.0, current_state.lift_height - previous_state.lift_height),
        hold=cfg.hold_weight * max(0.0, current_state.hold_time - previous_state.hold_time),
        safety=cfg.safety_failure_penalty if current_state.safety_failure else 0.0,
    )
    if stage is WirePpoStage.APPROVAL:
        return RewardBreakdown(total=sum(breakdown.approval_terms().values()), **{name: getattr(breakdown, name) for name in breakdown.__dataclass_fields__ if name != "total"})
    action = RewardBreakdown(
        total=0.0,
        approach=breakdown.approach, contact=breakdown.contact, grasp=breakdown.grasp,
        lift=breakdown.lift, hold=breakdown.hold, safety=breakdown.safety,
        centerline=cfg.action_centerline_weight * _progress(previous_state.wire_centerline_error, current_state.wire_centerline_error),
        tangent=cfg.action_tangent_weight * _progress(previous_state.tangent_error, current_state.tangent_error),
        deformation=cfg.action_deformation_weight * _progress(previous_state.deformation_error, current_state.deformation_error),
        synchronization=cfg.action_synchronization_weight * _progress(previous_state.synchronization_error, current_state.synchronization_error),
        force_balance=cfg.action_force_balance_weight * _progress(previous_state.force_balance_error, current_state.force_balance_error),
        slip=cfg.action_slip_weight * max(0.0, previous_state.slip - current_state.slip),
        latent_smoothness=cfg.action_latent_smoothness_weight * max(0.0, previous_state.latent_delta_norm - current_state.latent_delta_norm),
    )
    return RewardBreakdown(total=sum(action.action_terms().values()), **{name: getattr(action, name) for name in action.__dataclass_fields__ if name != "total"})


__all__ = [
    "RewardBreakdown", "WirePpoRewardCfg", "WirePpoStage", "WirePpoState",
    "approval_episode", "compute_wire_ppo_reward",
]
