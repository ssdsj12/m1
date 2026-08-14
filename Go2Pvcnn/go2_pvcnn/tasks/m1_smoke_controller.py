"""Open-loop M1 smoke controller actions.

The output order matches ``M1SmokeActionsCfg``: 12 leg-position actions first,
then 4 wheel-velocity actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


ROLLING_MODE = "rolling"
WAVE_MODE = "wave"


@dataclass(frozen=True)
class M1SmokeControllerCfg:
    """Numerical parameters for the open-loop M1 smoke controller."""

    leg_action_scale: float = 0.25
    wheel_action_scale: float = 8.0
    rolling_wheel_velocity: float = 0.5
    wave_wheel_velocity: float = 1.5
    wave_amplitude: float = 0.0
    wave_knee_ratio: float = 1.5
    wave_frequency: float = 1.0
    wave_phase_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5)
    wheel_velocity_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def _positive_sine(time_s: float, phase_offset: float, cfg: M1SmokeControllerCfg) -> float:
    phase = 2.0 * math.pi * (cfg.wave_frequency * time_s + phase_offset)
    return max(0.0, math.sin(phase))


def build_m1_smoke_action(
    *,
    num_envs: int,
    time_s: float,
    mode: str,
    cfg: M1SmokeControllerCfg | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a normalized M1 smoke action tensor.

    The IsaacLab action terms apply their own scale. This function therefore
    returns normalized commands: desired joint deltas divided by
    ``leg_action_scale`` and desired wheel speeds divided by
    ``wheel_action_scale``.
    """

    cfg = cfg or M1SmokeControllerCfg()
    action = torch.zeros((num_envs, 16), dtype=torch.float32, device=device)

    if mode == ROLLING_MODE:
        wheel_velocity = cfg.rolling_wheel_velocity
    elif mode == WAVE_MODE:
        wheel_velocity = cfg.wave_wheel_velocity
        for leg_index, phase_offset in enumerate(cfg.wave_phase_offsets):
            lift = cfg.wave_amplitude * _positive_sine(time_s, phase_offset, cfg)
            base_index = leg_index * 3
            direction = 1.0 if leg_index < 2 else -1.0
            action[:, base_index] = 0.0
            action[:, base_index + 1] = direction * lift / cfg.leg_action_scale
            action[:, base_index + 2] = -direction * lift * cfg.wave_knee_ratio / cfg.leg_action_scale
    else:
        raise ValueError(f"Unsupported M1 smoke control mode: {mode!r}")

    wheel_signs = torch.tensor(cfg.wheel_velocity_signs, dtype=action.dtype, device=device)
    action[:, 12:] = wheel_signs * (wheel_velocity / cfg.wheel_action_scale)
    return action
