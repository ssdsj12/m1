"""Deterministic two-stage M1 + Panda coordinated mission state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MissionPhase(str, Enum):
    FOLD_AND_NAVIGATE = "FOLD_AND_NAVIGATE"
    ARRIVE_HOLD = "ARRIVE_HOLD"
    UNFOLD_AND_TRACK = "UNFOLD_AND_TRACK"
    COORDINATED_TRACK = "COORDINATED_TRACK"


@dataclass(frozen=True)
class CoordinatedMissionCfg:
    physics_dt: float = 0.005
    arrive_position_tolerance_m: float = 0.05
    arrive_yaw_tolerance_rad: float = 0.08
    settled_steps: int = 20
    folded_arm_target: torch.Tensor | None = None
    unfold_duration_s: float = 1.0
    base_assist_radius_m: float = 0.25

    def __post_init__(self) -> None:
        if self.physics_dt <= 0.0 or self.settled_steps <= 0:
            raise ValueError("physics_dt and settled_steps must be positive")
        if self.arrive_position_tolerance_m < 0.0 or self.arrive_yaw_tolerance_rad < 0.0:
            raise ValueError("arrival tolerances must be non-negative")
        if self.unfold_duration_s <= 0.0 or self.base_assist_radius_m < 0.0:
            raise ValueError("unfold duration must be positive and radius non-negative")
        target = self.folded_arm_target
        if target is None:
            target = torch.zeros(7, dtype=torch.float32)
            object.__setattr__(self, "folded_arm_target", target)
        if not isinstance(target, torch.Tensor) or target.shape != (7,) or not target.is_floating_point():
            raise ValueError("folded_arm_target must be a floating 7-vector")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("folded_arm_target must be finite")


@dataclass(frozen=True)
class CoordinatedMissionState:
    phase: MissionPhase
    step: int
    target_base_pose: torch.Tensor
    arm_target: torch.Tensor
    ee_target_pose: torch.Tensor
    ee_target_twist: torch.Tensor
    settled_count: int


def _check(name: str, value: torch.Tensor, width: int) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating data")


class CoordinatedMission:
    def __init__(self, cfg: CoordinatedMissionCfg | None = None):
        self.cfg = cfg or CoordinatedMissionCfg()
        self._initialized = False

    def reset(
        self,
        *,
        base_pose: torch.Tensor,
        ee_pose: torch.Tensor,
        target_base_pose: torch.Tensor,
        ee_target_pose: torch.Tensor,
        seed: int,
    ) -> None:
        _check("base_pose", base_pose, 3)
        _check("ee_pose", ee_pose, 6)
        _check("target_base_pose", target_base_pose, 3)
        _check("ee_target_pose", ee_target_pose, 6)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if base_pose.dtype != self.cfg.folded_arm_target.dtype:
            raise TypeError("mission tensors must match folded_arm_target dtype")
        self._phase = MissionPhase.FOLD_AND_NAVIGATE
        self._step = -1
        self._settled_count = 0
        self._start_arm = ee_pose.new_zeros(7)
        self._target_base_pose = target_base_pose.detach().clone()
        self._ee_target_pose = ee_target_pose.detach().clone()
        self._last_ee_pose = ee_pose.detach().clone()
        self._seed = seed
        self._initialized = True

    def _arrival(self, base_pose: torch.Tensor) -> bool:
        position_error = torch.linalg.vector_norm(base_pose[:2] - self._target_base_pose[:2])
        yaw_error = torch.remainder(base_pose[2] - self._target_base_pose[2] + torch.pi, 2 * torch.pi) - torch.pi
        return bool(
            position_error <= self.cfg.arrive_position_tolerance_m
            and yaw_error.abs() <= self.cfg.arrive_yaw_tolerance_rad
        )

    def step(
        self,
        *,
        base_pose: torch.Tensor,
        ee_pose: torch.Tensor,
        ee_target_pose: torch.Tensor,
        ee_target_twist: torch.Tensor,
    ) -> CoordinatedMissionState:
        if not self._initialized:
            raise RuntimeError("mission must be reset before step")
        _check("base_pose", base_pose, 3)
        _check("ee_pose", ee_pose, 6)
        _check("ee_target_pose", ee_target_pose, 6)
        _check("ee_target_twist", ee_target_twist, 6)
        if base_pose.dtype != self._target_base_pose.dtype:
            raise TypeError("mission tensors must keep reset dtype")
        self._step += 1
        self._ee_target_pose = ee_target_pose.detach().clone()
        if self._phase is MissionPhase.FOLD_AND_NAVIGATE:
            if self._arrival(base_pose):
                self._settled_count += 1
                if self._settled_count >= self.cfg.settled_steps:
                    self._phase = MissionPhase.ARRIVE_HOLD
            else:
                self._settled_count = 0
        elif self._phase is MissionPhase.ARRIVE_HOLD:
            self._phase = MissionPhase.UNFOLD_AND_TRACK
            self._start_arm = self.cfg.folded_arm_target.to(device=ee_pose.device, dtype=ee_pose.dtype)
        elif self._phase is MissionPhase.UNFOLD_AND_TRACK:
            elapsed = (self._step - self.cfg.settled_steps - 1) * self.cfg.physics_dt
            if elapsed >= self.cfg.unfold_duration_s:
                self._phase = MissionPhase.COORDINATED_TRACK
        self._last_ee_pose = ee_pose.detach().clone()
        if self._phase is MissionPhase.FOLD_AND_NAVIGATE or self._phase is MissionPhase.ARRIVE_HOLD:
            arm_target = self.cfg.folded_arm_target.to(device=ee_pose.device, dtype=ee_pose.dtype)
        elif self._phase is MissionPhase.UNFOLD_AND_TRACK:
            duration = max(self.cfg.unfold_duration_s, self.cfg.physics_dt)
            elapsed = max(0.0, (self._step - self.cfg.settled_steps - 1) * self.cfg.physics_dt)
            fraction = min(1.0, elapsed / duration)
            arm_target = self._start_arm + fraction * (ee_target_pose.new_ones(7) * 0.0)
        else:
            arm_target = self._start_arm
        return CoordinatedMissionState(
            phase=self._phase,
            step=self._step,
            target_base_pose=self._target_base_pose.clone(),
            arm_target=arm_target.clone(),
            ee_target_pose=self._ee_target_pose.clone(),
            ee_target_twist=ee_target_twist.clone(),
            settled_count=self._settled_count,
        )


__all__ = ["CoordinatedMission", "CoordinatedMissionCfg", "CoordinatedMissionState", "MissionPhase"]
