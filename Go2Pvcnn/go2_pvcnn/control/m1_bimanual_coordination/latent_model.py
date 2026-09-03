"""PyTorch conditional action autoencoder and 200 Hz body controller."""

from __future__ import annotations

import torch
from torch import nn

from .latent_contracts import (
    ACTION_DIM,
    HORIZON,
    LATENT_DIM,
    STATE_DIM,
    TASK_FEATURE_DIM,
)


class LatentActionModel(nn.Module):
    """Compress a full teacher trajectory to z16 and reproduce its actions."""

    def __init__(self) -> None:
        super().__init__()
        encoder_input = STATE_DIM + HORIZON * (ACTION_DIM + TASK_FEATURE_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, LATENT_DIM),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, HORIZON * ACTION_DIM),
        )
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM + 1 + ACTION_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, ACTION_DIM),
            nn.Tanh(),
        )

    def encode(
        self,
        state: torch.Tensor,
        teacher_action: torch.Tensor,
        teacher_task: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (state, teacher_action.flatten(1), teacher_task.flatten(1)), dim=1
        )
        return self.encoder(features)

    def decode_trajectory(
        self, state: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        decoded = self.decoder(torch.cat((state, latent), dim=1))
        return decoded.reshape(-1, HORIZON, ACTION_DIM)

    def body_action(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        phase: torch.Tensor,
        last_effort: torch.Tensor,
    ) -> torch.Tensor:
        return self.body(torch.cat((state, latent, phase, last_effort), dim=1))


__all__ = ["LatentActionModel"]
