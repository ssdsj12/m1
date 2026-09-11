"""Pure helpers for independent vector-environment bimanual commands."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .contracts import BimanualCommand


def stack_lane_actions(
    commands: Sequence[BimanualCommand], *, device: torch.device | str
) -> torch.Tensor:
    if not commands:
        raise ValueError("commands must not be empty")
    if any(not isinstance(command, BimanualCommand) for command in commands):
        raise TypeError("commands must contain only BimanualCommand values")
    return torch.stack(tuple(command.effort for command in commands)).to(
        device=device, dtype=torch.float32
    )
