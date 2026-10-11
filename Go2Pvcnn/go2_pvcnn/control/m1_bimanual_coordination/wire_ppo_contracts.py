"""Strict contracts for the 15-dimensional PPO wire policy interface.

The policy only emits normalized latent commands.  This module deliberately
does not mutate an environment or controller: malformed data is rejected
before a command object can be handed to either one.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import torch


WIRE_PPO_ACTION_DIM = 15
_LATENT_BOUNDS = (-1.0, 1.0)


@dataclass(frozen=True)
class WirePpoObservationSpec:
    """Shape contract for one normalized wire-policy observation.

    The groups mirror the runtime boundary and are kept explicit so additions
    cannot silently change the policy input width.
    """

    wire_state_dim: int = 80
    bimanual_state_dim: int = 84
    arm_state_dim: int = 28
    safety_phase_dim: int = 20
    previous_action_dim: int = WIRE_PPO_ACTION_DIM

    @property
    def observation_dim(self) -> int:
        return (
            self.wire_state_dim
            + self.bimanual_state_dim
            + self.arm_state_dim
            + self.safety_phase_dim
            + self.previous_action_dim
        )

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.previous_action_dim != WIRE_PPO_ACTION_DIM:
            raise ValueError("previous_action_dim must equal the PPO action dimension")

    def validate(self, observation: torch.Tensor) -> None:
        _validate_normalized_vector(
            "observation", observation, self.observation_dim
        )


@dataclass(frozen=True)
class WireLatentCommand:
    """Decoded, still-normalized command groups sent to the safety adapter."""

    palm_delta: torch.Tensor
    hand_delta: torch.Tensor
    phase_force: torch.Tensor

    def __post_init__(self) -> None:
        _validate_normalized_vector("palm_delta", self.palm_delta, 6)
        _validate_normalized_vector("hand_delta", self.hand_delta, 6)
        _validate_normalized_vector("phase_force", self.phase_force, 3)
        if not (
            self.palm_delta.device == self.hand_delta.device == self.phase_force.device
        ):
            raise ValueError("decoded command groups must share one device")
        if not (
            self.palm_delta.dtype == self.hand_delta.dtype == self.phase_force.dtype
        ):
            raise TypeError("decoded command groups must share one dtype")

    @property
    def as_vector(self) -> torch.Tensor:
        return torch.cat((self.palm_delta, self.hand_delta, self.phase_force), dim=0)


def _validate_floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have a floating dtype; got {value.dtype}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_normalized_vector(name: str, value: torch.Tensor, size: int) -> None:
    _validate_floating_tensor(name, value)
    if tuple(value.shape) != (size,):
        raise ValueError(f"{name} must have exact shape ({size},); got {tuple(value.shape)}")
    lower, upper = _LATENT_BOUNDS
    if not torch.all((value >= lower) & (value <= upper)).item():
        raise ValueError(f"{name} must be bounded in [{lower}, {upper}]")


def decode_wire_latent(
    action: torch.Tensor, *, device: torch.device | str, dtype: torch.dtype
) -> WireLatentCommand:
    """Validate and split a normalized PPO action without side effects."""

    _validate_normalized_vector("action", action, WIRE_PPO_ACTION_DIM)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch.dtype")
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise TypeError("device must be a valid torch device") from exc
    decoded = action.to(device=target_device, dtype=dtype).clone()
    if not torch.isfinite(decoded).all().item():
        raise ValueError("decoded action must remain finite")
    return WireLatentCommand(
        palm_delta=decoded[:6].clone(),
        hand_delta=decoded[6:12].clone(),
        phase_force=decoded[12:15].clone(),
    )


def _validate_reward(reward: torch.Tensor | float, *, device: torch.device) -> None:
    if isinstance(reward, bool):
        raise TypeError("reward must be a finite scalar")
    if isinstance(reward, torch.Tensor):
        if not reward.dtype.is_floating_point or tuple(reward.shape) != ():
            raise TypeError("reward tensor must be a floating scalar tensor")
        if reward.device != device:
            raise ValueError("reward tensor must share the observation device")
        if not torch.isfinite(reward).item():
            raise ValueError("reward must be finite")
        return
    if not isinstance(reward, numbers.Real) or not torch.isfinite(torch.tensor(float(reward))):
        raise TypeError("reward must be a finite scalar")


def validate_wire_ppo_transition(
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor | float,
    done: bool,
) -> None:
    """Validate one PPO transition atomically, raising before any state change."""

    spec = WirePpoObservationSpec()
    spec.validate(observation)
    _validate_normalized_vector("action", action, WIRE_PPO_ACTION_DIM)
    if action.device != observation.device:
        raise ValueError("action and observation must share a device")
    if action.dtype != observation.dtype:
        raise TypeError("action and observation must share a dtype")
    _validate_reward(reward, device=observation.device)
    if not isinstance(done, bool):
        raise TypeError("done must be a bool")


__all__ = [
    "WIRE_PPO_ACTION_DIM",
    "WireLatentCommand",
    "WirePpoObservationSpec",
    "decode_wire_latent",
    "validate_wire_ppo_transition",
]
