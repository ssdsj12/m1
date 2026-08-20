"""One-Teacher-per-environment orchestration for Student S1 collection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


_MUTABLE_CONTROLLER_ATTRS = (
    "schedule",
    "trajectory",
    "motion_distributor",
    "qp_backend",
    "safety",
    "settling_center",
    "history",
    "first_failure",
)


class BatchedRollingTeacherBank:
    """Keep C1a Teacher controller state isolated per simulated environment."""

    def __init__(self, teachers: Sequence[object], adapters: Sequence[object], *, base_seed: int = 0):
        if len(teachers) != len(adapters) or len(teachers) == 0:
            raise ValueError("one adapter is required for every Teacher")
        if len({id(adapter) for adapter in adapters}) != len(adapters):
            raise ValueError("each Teacher requires a distinct adapter")
        self.teachers = list(teachers)
        self.adapters = list(adapters)
        self.base_seed = int(base_seed)
        self._validate_distinct_teacher_state()

    def _validate_distinct_teacher_state(self) -> None:
        for attr in _MUTABLE_CONTROLLER_ATTRS:
            seen: dict[int, int] = {}
            for index, teacher in enumerate(self.teachers):
                if not hasattr(teacher, attr):
                    continue
                value = getattr(teacher, attr)
                value_id = id(value)
                if value_id in seen:
                    raise ValueError(
                        f"Teacher {index} shares mutable {attr} with Teacher {seen[value_id]}"
                    )
                seen[value_id] = index

    def reset(self, env_ids: torch.Tensor | Iterable[int], states: Sequence[object]) -> None:
        for env_id in self._env_id_list(env_ids):
            self.teachers[env_id].reset(
                states[env_id],
                seed=self.base_seed + env_id,
            )

    def step(self, states: Sequence[object], mission_samples: Sequence[object]) -> list[object]:
        if len(states) != len(self.teachers) or len(mission_samples) != len(self.teachers):
            raise ValueError("states and mission_samples must match Teacher count")
        return [
            teacher.step(state, mission_sample=mission_sample)
            for teacher, state, mission_sample in zip(
                self.teachers,
                states,
                mission_samples,
                strict=True,
            )
        ]

    def _env_id_list(self, env_ids: torch.Tensor | Iterable[int]) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            raw_ids = env_ids.detach().cpu().reshape(-1).tolist()
        else:
            raw_ids = list(env_ids)
        env_id_list = [int(env_id) for env_id in raw_ids]
        for env_id in env_id_list:
            if env_id < 0 or env_id >= len(self.teachers):
                raise IndexError(f"env_id must be in [0,{len(self.teachers)}), got {env_id}")
        return env_id_list


__all__ = ["BatchedRollingTeacherBank"]
