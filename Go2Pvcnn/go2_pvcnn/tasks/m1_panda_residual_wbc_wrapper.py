"""Independent eight-action controller boundary over accepted WBC Teachers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    MountWrenchFeedback,
    MountWrenchFeedbackCfg,
    WholeBodyResidualCfg,
    WholeBodyResidualCommand,
    WholeBodyResidualComposer,
    WholeBodyResidualDiagnostics,
    select_residual_command,
)


@dataclass(frozen=True)
class ResidualWbcStep:
    teacher_commands: tuple[object, ...]
    applied_residual: WholeBodyResidualCommand
    correction_wrench_b: torch.Tensor
    filtered_mount_wrench_b: torch.Tensor
    residual_diagnostics: WholeBodyResidualDiagnostics
    predicted_mount_wrench_b: torch.Tensor


class M1PandaResidualWbcController:
    """Keep composer, filter, and Teacher mutable state isolated per environment."""

    def __init__(
        self,
        teachers: Sequence[object],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        base_seed: int = 0,
        residual_cfg: WholeBodyResidualCfg | None = None,
        feedback_cfg: MountWrenchFeedbackCfg | None = None,
    ) -> None:
        if isinstance(teachers, (str, bytes)) or not isinstance(teachers, Sequence):
            raise TypeError("teachers must be a non-empty sequence")
        if len(teachers) == 0:
            raise ValueError("teachers must be a non-empty sequence")
        if len({id(teacher) for teacher in teachers}) != len(teachers):
            raise ValueError("teachers must be distinct instances")
        if not isinstance(base_seed, int) or isinstance(base_seed, bool):
            raise TypeError("base_seed must be an integer")
        self.teachers = list(teachers)
        self.num_envs = len(self.teachers)
        self.device = torch.device(device)
        self.dtype = dtype
        self.base_seed = base_seed
        self._composer = WholeBodyResidualComposer(
            self.num_envs, self.device, dtype, residual_cfg
        )
        self._feedback = MountWrenchFeedback(
            self.num_envs, self.device, dtype, feedback_cfg
        )

    @property
    def previous_physical(self) -> torch.Tensor:
        return self._composer.previous_physical

    @property
    def filtered_mount_wrench_b(self) -> torch.Tensor:
        return self._feedback.filtered_wrench

    def _env_id_list(
        self, env_ids: torch.Tensor | Iterable[int] | None
    ) -> list[int]:
        if env_ids is None:
            values = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.dtype != torch.long:
                raise TypeError("env_ids tensor must have dtype torch.int64")
            values = [int(value) for value in env_ids.detach().cpu().reshape(-1)]
        elif isinstance(env_ids, Iterable) and not isinstance(env_ids, (str, bytes)):
            raw = list(env_ids)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
                raise TypeError("env_ids must contain integers")
            values = [int(value) for value in raw]
        else:
            raise TypeError("env_ids must be an int64 tensor, integer iterable, or None")
        if any(value < 0 or value >= self.num_envs for value in values):
            raise IndexError(f"env_ids must be in [0,{self.num_envs})")
        if len(set(values)) != len(values):
            raise ValueError("env_ids must not contain duplicates")
        return values

    def _validate_step_inputs(
        self,
        states: Sequence[object],
        normalized_residual: torch.Tensor,
        measured_mount_wrench_b: torch.Tensor,
        leg_soft_limits: torch.Tensor,
        predicted_mount_wrench_b: torch.Tensor | None = None,
    ) -> None:
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise TypeError("states must be a sequence")
        if len(states) != self.num_envs:
            raise ValueError("states length must match Teacher count")
        tensors = (
            ("normalized_residual", normalized_residual, (self.num_envs, 8)),
            (
                "measured_mount_wrench_b",
                measured_mount_wrench_b,
                (self.num_envs, 6),
            ),
            ("leg_soft_limits", leg_soft_limits, (self.num_envs, 12, 2)),
        )
        for name, value, shape in tensors:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape must be {shape}")
            if value.dtype != self.dtype:
                raise TypeError(f"{name} dtype must match controller dtype")
            if value.device != self.device:
                raise ValueError(f"{name} device must match controller device")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(leg_soft_limits[..., 0] > leg_soft_limits[..., 1]).item():
            raise ValueError("leg_soft_limits lower bounds must not exceed upper bounds")
        if predicted_mount_wrench_b is not None:
            name = "predicted_mount_wrench_b"
            if not isinstance(predicted_mount_wrench_b, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(predicted_mount_wrench_b.shape) != (self.num_envs, 6):
                raise ValueError(f"{name} shape must be ({self.num_envs}, 6)")
            if predicted_mount_wrench_b.dtype != self.dtype:
                raise TypeError(f"{name} dtype must match controller dtype")
            if predicted_mount_wrench_b.device != self.device:
                raise ValueError(f"{name} device must match controller device")
            if not torch.isfinite(predicted_mount_wrench_b).all().item():
                raise ValueError(f"{name} must contain only finite values")

    def reset(
        self,
        env_ids: torch.Tensor | Iterable[int] | None = None,
        *,
        states: Sequence[object] | None = None,
    ) -> None:
        ids = self._env_id_list(env_ids)
        if states is not None:
            if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
                raise TypeError("states must be a sequence or None")
            if len(states) != self.num_envs:
                raise ValueError("states length must match Teacher count")
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._composer.reset(ids_tensor)
        self._feedback.reset(ids_tensor)
        if states is not None:
            for env_id in ids:
                self.teachers[env_id].reset(
                    states[env_id], seed=self.base_seed + env_id
                )

    def step(
        self,
        *,
        states: Sequence[object],
        normalized_residual: torch.Tensor,
        measured_mount_wrench_b: torch.Tensor,
        leg_soft_limits: torch.Tensor,
        predicted_mount_wrench_b: torch.Tensor | None = None,
        teacher_kwargs: Sequence[dict] | None = None,
    ) -> ResidualWbcStep:
        self._validate_step_inputs(
            states,
            normalized_residual,
            measured_mount_wrench_b,
            leg_soft_limits,
            predicted_mount_wrench_b,
        )
        if predicted_mount_wrench_b is None:
            predicted_mount_wrench_b = torch.zeros(
                (self.num_envs, 6), dtype=self.dtype, device=self.device
            )
        else:
            predicted_mount_wrench_b = predicted_mount_wrench_b.clone()
        if teacher_kwargs is None:
            kwargs_per_env = [{} for _ in range(self.num_envs)]
        else:
            if len(teacher_kwargs) != self.num_envs or any(
                not isinstance(value, dict) for value in teacher_kwargs
            ):
                raise ValueError("teacher_kwargs must contain one dict per environment")
            kwargs_per_env = [dict(value) for value in teacher_kwargs]
        raw_command, diagnostics = self._composer.step(normalized_residual)
        correction = self._feedback.update(
            measured_mount_wrench_b,
            raw_command.wrench_b - predicted_mount_wrench_b,
        )
        physical = raw_command.physical.clone()
        physical[:, :6] = correction
        applied = WholeBodyResidualCommand(
            physical=physical,
            wrench_b=correction.clone(),
            delta_height=raw_command.delta_height.clone(),
            delta_stance=raw_command.delta_stance.clone(),
        )
        commands = []
        for env_id, teacher in enumerate(self.teachers):
            kwargs = kwargs_per_env[env_id]
            if "residual_command" in kwargs or "leg_soft_limits" in kwargs:
                raise ValueError(
                    "teacher_kwargs cannot override the residual control boundary"
                )
            commands.append(
                teacher.step(
                    states[env_id],
                    residual_command=select_residual_command(applied, env_id),
                    leg_soft_limits=leg_soft_limits[env_id].clone(),
                    **kwargs,
                )
            )
        return ResidualWbcStep(
            teacher_commands=tuple(commands),
            applied_residual=WholeBodyResidualCommand(
                physical=applied.physical.clone(),
                wrench_b=applied.wrench_b.clone(),
                delta_height=applied.delta_height.clone(),
                delta_stance=applied.delta_stance.clone(),
            ),
            correction_wrench_b=correction.clone(),
            filtered_mount_wrench_b=self._feedback.filtered_wrench,
            residual_diagnostics=diagnostics,
            predicted_mount_wrench_b=predicted_mount_wrench_b.clone(),
        )


__all__ = ["M1PandaResidualWbcController", "ResidualWbcStep"]
