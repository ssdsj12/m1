"""Pure contracts for the M1 + Panda folded-load locomotion curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class CommandFamily(IntEnum):
    """Episode-level command families used for exact metric attribution."""

    STATIONARY = 0
    STRAIGHT = 1
    TURN_IN_PLACE = 2
    COMBINED = 3


@dataclass(frozen=True)
class ResetRanges:
    """Symmetric reset deviations and material ranges for one stage."""

    root_xy: float = 0.0
    root_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_linear_velocity: float = 0.0
    root_angular_velocity: float = 0.0
    leg_position: float = 0.0
    friction: tuple[float, float] = (1.0, 1.0)
    root_z: tuple[float, float] = (0.0, 0.0)
    wheel_position: tuple[float, float] = (0.0, 0.0)
    panda_position: tuple[float, float] = (0.0, 0.0)
    panda_velocity: tuple[float, float] = (0.0, 0.0)
    restitution: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class StageSpec:
    """Immutable command and reset contract for one curriculum stage."""

    name: str
    parent: str | None
    vx_limit: float
    wz_limit: float
    completed_episode_window: int
    reset: ResetRanges


@dataclass(frozen=True)
class CommandBatch:
    """Sampled body-twist commands and their episode family labels."""

    twist: torch.Tensor
    family: torch.Tensor


STAGE_ORDER = (
    "L0-C0",
    "L1-C1",
    "L1-C2",
    "L1-C3",
    "L1-C4",
    "L2-D1",
    "L2-D2",
    "L2-D3",
)

_DETERMINISTIC_RESET = ResetRanges()
_D1_RESET = ResetRanges(
    root_xy=0.005,
    root_rpy=(0.01, 0.01, 0.01),
    root_linear_velocity=0.01,
    root_angular_velocity=0.02,
    leg_position=0.005,
    friction=(0.95, 1.05),
)
_D2_RESET = ResetRanges(
    root_xy=0.01,
    root_rpy=(0.015, 0.015, 0.025),
    root_linear_velocity=0.025,
    root_angular_velocity=0.05,
    leg_position=0.01,
    friction=(0.90, 1.10),
)
_D3_RESET = ResetRanges(
    root_xy=0.02,
    root_rpy=(0.03, 0.03, 0.05),
    root_linear_velocity=0.05,
    root_angular_velocity=0.10,
    leg_position=0.02,
    friction=(0.80, 1.20),
)

_STAGES = {
    "L0-C0": StageSpec("L0-C0", None, 0.05, 0.15, 200, _DETERMINISTIC_RESET),
    "L1-C1": StageSpec("L1-C1", "L0-C0", 0.08, 0.25, 200, _DETERMINISTIC_RESET),
    "L1-C2": StageSpec("L1-C2", "L1-C1", 0.12, 0.35, 200, _DETERMINISTIC_RESET),
    "L1-C3": StageSpec("L1-C3", "L1-C2", 0.16, 0.48, 200, _DETERMINISTIC_RESET),
    "L1-C4": StageSpec("L1-C4", "L1-C3", 0.20, 0.60, 200, _DETERMINISTIC_RESET),
    "L2-D1": StageSpec("L2-D1", "L1-C4", 0.20, 0.60, 400, _D1_RESET),
    "L2-D2": StageSpec("L2-D2", "L2-D1", 0.20, 0.60, 400, _D2_RESET),
    "L2-D3": StageSpec("L2-D3", "L2-D2", 0.20, 0.60, 400, _D3_RESET),
}


def stage_spec(name: str) -> StageSpec:
    """Return the immutable contract for ``name``."""

    try:
        return _STAGES[name]
    except KeyError as exc:
        raise ValueError(f"unknown folded-load stage: {name!r}") from exc


def _nonzero_uniform(
    count: int,
    limit: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    values = torch.rand(count, generator=generator, device=device)
    epsilon = torch.finfo(values.dtype).eps
    return (epsilon + values * (1.0 - epsilon)) * limit


def _signed_magnitudes(
    count: int,
    limit: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    magnitude = _nonzero_uniform(count, limit, generator=generator, device=device)
    signs = torch.randint(0, 2, (count,), generator=generator, device=device)
    return magnitude * signs.mul(2).sub(1)


def sample_episode_commands(
    num_envs: int,
    stage: StageSpec,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> CommandBatch:
    """Sample one seeded, episode-constant command for each environment."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))
    probabilities = torch.tensor((0.20, 0.25, 0.20, 0.35), device=torch_device)
    family = torch.multinomial(
        probabilities,
        num_samples=num_envs,
        replacement=True,
        generator=generator,
    )
    twist = torch.zeros((num_envs, 3), device=torch_device)

    straight = family == int(CommandFamily.STRAIGHT)
    turning = family == int(CommandFamily.TURN_IN_PLACE)
    combined = family == int(CommandFamily.COMBINED)
    moving = straight | combined
    rotating = turning | combined
    twist[moving, 0] = _signed_magnitudes(
        int(moving.sum().item()),
        stage.vx_limit,
        generator=generator,
        device=torch_device,
    )
    twist[rotating, 2] = _signed_magnitudes(
        int(rotating.sum().item()),
        stage.wz_limit,
        generator=generator,
        device=torch_device,
    )
    return CommandBatch(twist=twist, family=family)


def balanced_eval_commands(
    num_envs: int,
    stage: StageSpec,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return a deterministic command table balanced in 16-row blocks."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if num_envs % 16 != 0:
        raise ValueError("num_envs must be a multiple of 16")
    vx = stage.vx_limit
    wz = stage.wz_limit
    block = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (vx, 0.0, 0.0),
            (vx, 0.0, 0.0),
            (-vx, 0.0, 0.0),
            (-vx, 0.0, 0.0),
            (0.0, 0.0, wz),
            (0.0, 0.0, wz),
            (0.0, 0.0, -wz),
            (0.0, 0.0, -wz),
            (vx, 0.0, wz),
            (vx, 0.0, -wz),
            (-vx, 0.0, wz),
            (-vx, 0.0, -wz),
        ),
        device=device,
    )
    return block.repeat(num_envs // 16, 1)


def classify_command_buckets(commands: torch.Tensor) -> dict[str, torch.Tensor]:
    """Classify commands into stationary and signed directional buckets."""

    if not isinstance(commands, torch.Tensor) or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands must be a [N, 3] tensor")
    vx = commands[:, 0]
    wz = commands[:, 2]
    return {
        "stationary": vx.eq(0.0) & wz.eq(0.0),
        "forward": vx.gt(0.0),
        "reverse": vx.lt(0.0),
        "left": wz.gt(0.0),
        "right": wz.lt(0.0),
    }


__all__ = [
    "CommandBatch",
    "CommandFamily",
    "ResetRanges",
    "STAGE_ORDER",
    "StageSpec",
    "balanced_eval_commands",
    "classify_command_buckets",
    "sample_episode_commands",
    "stage_spec",
]
