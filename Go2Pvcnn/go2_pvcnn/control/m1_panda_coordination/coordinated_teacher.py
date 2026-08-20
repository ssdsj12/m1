"""Pure coordination adapter joining mission state and bounded base assistance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base_assist import BaseAssistCfg, compute_base_assist
from .coordinated_mission import CoordinatedMission, CoordinatedMissionState, MissionPhase


@dataclass(frozen=True)
class CoordinatedTeacherDecision:
    phase: MissionPhase
    mission: CoordinatedMissionState
    base_velocity: torch.Tensor
    base_assist_active: bool
    base_assist_reason: str
    arm_margin_before: torch.Tensor
    arm_margin_after: torch.Tensor


class CoordinatedTeacherAdapter:
    """Keep mission transitions separate from the existing WBC/QP Teacher."""

    def __init__(self, *, mission: CoordinatedMission | None = None, assist_cfg: BaseAssistCfg | None = None):
        self.mission = mission or CoordinatedMission()
        self.assist_cfg = assist_cfg or BaseAssistCfg()
        self._arrived_pose: torch.Tensor | None = None
        self._previous_velocity: torch.Tensor | None = None

    def reset(self, base_pose, ee_pose, target_base_pose, ee_target_pose, *, seed: int) -> None:
        self.mission.reset(
            base_pose=base_pose,
            ee_pose=ee_pose,
            target_base_pose=target_base_pose,
            ee_target_pose=ee_target_pose,
            seed=seed,
        )
        self._arrived_pose = target_base_pose.detach().clone()
        self._previous_velocity = torch.zeros_like(base_pose)

    def step(self, base_pose, ee_pose, ee_target_pose, ee_target_twist, arm_margin_before, arm_margin_after, sigma_min=None) -> CoordinatedTeacherDecision:
        if self._arrived_pose is None or self._previous_velocity is None:
            raise RuntimeError("adapter must be reset before step")
        state = self.mission.step(
            base_pose=base_pose,
            ee_pose=ee_pose,
            ee_target_pose=ee_target_pose,
            ee_target_twist=ee_target_twist,
        )
        sigma = arm_margin_before if sigma_min is None else sigma_min
        if state.phase is not MissionPhase.COORDINATED_TRACK:
            velocity = torch.zeros_like(base_pose)
            self._previous_velocity.zero_()
            return CoordinatedTeacherDecision(
                state.phase, state, velocity, False, "mission_phase", arm_margin_before.clone(), arm_margin_after.clone()
            )
        decision = compute_base_assist(
            base_pose=base_pose,
            arrived_base_pose=self._arrived_pose,
            target_base_pose=self._arrived_pose,
            arm_margin_before=arm_margin_before,
            arm_margin_after=arm_margin_after,
            sigma_min=sigma,
            previous_velocity=self._previous_velocity,
            dt=self.mission.cfg.physics_dt,
            cfg=self.assist_cfg,
        )
        self._previous_velocity = decision.base_velocity.detach().clone()
        return CoordinatedTeacherDecision(
            state.phase,
            state,
            decision.base_velocity,
            decision.active,
            decision.reason,
            decision.arm_margin_before,
            decision.arm_margin_after,
        )


__all__ = ["CoordinatedTeacherAdapter", "CoordinatedTeacherDecision"]
