"""Explicit temporal history and GRU estimator/actor for Student S1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
)


class StudentHistoryBuffer:
    """Per-environment rolling history without hidden global recurrent state."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch.dtype")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self._value = torch.zeros(
            num_envs,
            STUDENT_HISTORY_LENGTH,
            STUDENT_OBSERVATION_DIM,
            device=self.device,
            dtype=self.dtype,
        )

    def append(self, observation: torch.Tensor) -> None:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        expected = (self.num_envs, STUDENT_OBSERVATION_DIM)
        if tuple(observation.shape) != expected:
            raise ValueError(
                f"observation must have shape {expected}, got {tuple(observation.shape)}"
            )
        if observation.device != self.device:
            raise ValueError(
                f"observation must be on device {self.device}, got {observation.device}"
            )
        if observation.dtype != self.dtype:
            raise TypeError(
                f"observation must have dtype {self.dtype}, got {observation.dtype}"
            )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation must contain only finite values")
        self._value[:, :-1].copy_(self._value[:, 1:].clone())
        self._value[:, -1].copy_(observation)

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        if env_ids is None:
            self._value.zero_()
            return
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("env_ids must be a one-dimensional integer tensor")
            indices = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(
            env_ids, (str, bytes)
        ):
            indices = list(env_ids)
        else:
            raise TypeError("env_ids must be an integer tensor or sequence")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= self.num_envs
            for index in indices
        ):
            raise IndexError("env_ids contains an invalid environment index")
        normalized = torch.tensor(
            indices, device=self.device, dtype=torch.long
        )
        self._value[normalized] = 0.0

    @property
    def value(self) -> torch.Tensor:
        return self._value.clone()


@dataclass(frozen=True)
class StudentNetworkCfg:
    observation_dim: int = STUDENT_OBSERVATION_DIM
    history_length: int = STUDENT_HISTORY_LENGTH
    action_dim: int = STUDENT_ACTION_DIM
    gru_hidden_dim: int = 128
    latent_dim: int = 32
    wrench_dim: int = 6

    def __post_init__(self) -> None:
        if self.observation_dim != STUDENT_OBSERVATION_DIM:
            raise ValueError("observation_dim must remain 100")
        if self.history_length != STUDENT_HISTORY_LENGTH:
            raise ValueError("history_length must remain 10")
        if self.action_dim != STUDENT_ACTION_DIM:
            raise ValueError("action_dim must remain 23")
        for name in ("gru_hidden_dim", "latent_dim", "wrench_dim"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class StudentOutput:
    wrench_hat: torch.Tensor
    latent: torch.Tensor
    safety_logit: torch.Tensor
    raw_action: torch.Tensor
    action: torch.Tensor


class M1PandaStudent(nn.Module):
    """GRU estimator plus bounded residual actor."""

    def __init__(self, cfg: StudentNetworkCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.observation_dim,
            hidden_size=cfg.gru_hidden_dim,
            batch_first=True,
        )
        self.estimator_head = nn.Linear(
            cfg.gru_hidden_dim, cfg.wrench_dim + cfg.latent_dim
        )
        self.safety_head = nn.Linear(cfg.gru_hidden_dim, 1)
        actor_input_dim = (
            cfg.observation_dim
            + cfg.gru_hidden_dim
            + cfg.wrench_dim
            + cfg.latent_dim
        )
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, cfg.action_dim),
        )

    def forward(self, history: torch.Tensor) -> StudentOutput:
        if not isinstance(history, torch.Tensor):
            raise TypeError("history must be a torch.Tensor")
        expected_tail = (
            self.cfg.history_length,
            self.cfg.observation_dim,
        )
        if history.ndim != 3 or tuple(history.shape[1:]) != expected_tail:
            raise ValueError(
                "history must have shape "
                f"[E,{self.cfg.history_length},{self.cfg.observation_dim}], "
                f"got {tuple(history.shape)}"
            )
        parameter = next(self.parameters())
        if history.device != parameter.device:
            raise ValueError(
                f"history must be on device {parameter.device}, got {history.device}"
            )
        if history.dtype != parameter.dtype:
            raise TypeError(
                f"history must have dtype {parameter.dtype}, got {history.dtype}"
            )
        if not bool(torch.isfinite(history).all()):
            raise ValueError("history must contain only finite values")

        encoded_sequence, _ = self.gru(history)
        encoding = encoded_sequence[:, -1]
        estimate = self.estimator_head(encoding)
        wrench_hat, latent = torch.split(
            estimate,
            (self.cfg.wrench_dim, self.cfg.latent_dim),
            dim=-1,
        )
        safety_logit = self.safety_head(encoding)
        actor_input = torch.cat(
            (history[:, -1], encoding, wrench_hat, latent), dim=-1
        )
        raw_action = self.actor(actor_input)
        action = torch.tanh(raw_action)
        return StudentOutput(
            wrench_hat=wrench_hat,
            latent=latent,
            safety_logit=safety_logit,
            raw_action=raw_action,
            action=action,
        )


__all__ = [
    "M1PandaStudent",
    "StudentHistoryBuffer",
    "StudentNetworkCfg",
    "StudentOutput",
]
