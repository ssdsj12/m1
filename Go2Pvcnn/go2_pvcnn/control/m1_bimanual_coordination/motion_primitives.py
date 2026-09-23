"""Closed-loop phase targets for the first bimanual lift cycle."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import BimanualPhase, BimanualSnapshot
from .grasp_goal import BimanualGraspGoal


@dataclass(frozen=True)
class MotionPrimitiveCfg:
    dt: float = 0.04
    horizon_steps: int = 25
    lift_height_m: float = 0.10
    lift_speed_m_s: float = 0.025
    lower_speed_m_s: float = 0.025
    palm_linear_speed_m_s: float = 0.04
    palm_angular_speed_rad_s: float = 0.175
    preload_force_n: float = 5.0
    grasp_force_n: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.horizon_steps, bool) or self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        for name in (
            "dt",
            "lift_height_m",
            "lift_speed_m_s",
            "lower_speed_m_s",
            "palm_linear_speed_m_s",
            "palm_angular_speed_rad_s",
            "preload_force_n",
            "grasp_force_n",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.preload_force_n >= self.grasp_force_n:
            raise ValueError("preload_force_n must be below grasp_force_n")


@dataclass(frozen=True)
class ManipulationTarget:
    box_pose_b: torch.Tensor
    left_palm_in_box: torch.Tensor
    right_palm_in_box: torch.Tensor
    target_normal_force_n: float
    recovery_side: str | None
    grasp_goal: BimanualGraspGoal | None = None

    @property
    def left_fingertip_targets(self):
        return () if self.grasp_goal is None else self.grasp_goal.left_fingertip_targets

    @property
    def right_fingertip_targets(self):
        return () if self.grasp_goal is None else self.grasp_goal.right_fingertip_targets

    @property
    def left_contact_targets(self):
        return () if self.grasp_goal is None else self.grasp_goal.left_contact_targets

    @property
    def right_contact_targets(self):
        return () if self.grasp_goal is None else self.grasp_goal.right_contact_targets

    @property
    def clamp_criteria(self):
        return None if self.grasp_goal is None else self.grasp_goal.clamp_criteria

    @property
    def lift_criteria(self):
        return None if self.grasp_goal is None else self.grasp_goal.lift_criteria


class BimanualMotionPrimitive:
    """Generate bounded object targets from measured state and mission phase."""

    def __init__(self, cfg: MotionPrimitiveCfg | None = None) -> None:
        self.cfg = MotionPrimitiveCfg() if cfg is None else cfg
        self._initial_box_pose: torch.Tensor | None = None
        self._left_palm_in_box: torch.Tensor | None = None
        self._right_palm_in_box: torch.Tensor | None = None

    def reset(self) -> None:
        self._initial_box_pose = None
        self._left_palm_in_box = None
        self._right_palm_in_box = None

    @staticmethod
    def _has_contact(snapshot: BimanualSnapshot, side: str) -> bool:
        hand = snapshot.left_hand if side == "left" else snapshot.right_hand
        return bool(hand.contact_mask.any()) and hand.contact_consistent

    def _capture_grasp(self, snapshot: BimanualSnapshot) -> None:
        self._left_palm_in_box = (
            snapshot.left_arm.palm_pose_b - snapshot.box.pose_b
        ).clone()
        self._right_palm_in_box = (
            snapshot.right_arm.palm_pose_b - snapshot.box.pose_b
        ).clone()

    def target(
        self,
        phase: BimanualPhase,
        snapshot: BimanualSnapshot,
        grasp_goal: BimanualGraspGoal | None = None,
    ) -> ManipulationTarget:
        if not isinstance(phase, BimanualPhase):
            raise TypeError("phase must be BimanualPhase")
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if self._initial_box_pose is None:
            self._initial_box_pose = snapshot.box.pose_b.clone()

        left_contact = self._has_contact(snapshot, "left")
        right_contact = self._has_contact(snapshot, "right")
        recovery_side = None
        if left_contact != right_contact:
            recovery_side = "right" if left_contact else "left"
        elif not left_contact and not right_contact and phase in {
            BimanualPhase.GRASP,
            BimanualPhase.LIFT,
            BimanualPhase.HOLD,
            BimanualPhase.LOWER,
        }:
            recovery_side = "both"

        if phase is BimanualPhase.GRASP and left_contact and right_contact:
            self._capture_grasp(snapshot)
        if self._left_palm_in_box is None or self._right_palm_in_box is None:
            self._capture_grasp(snapshot)

        horizon = self.cfg.horizon_steps
        box_pose = snapshot.box.pose_b.repeat(horizon, 1)
        both_contact = left_contact and right_contact
        lift_height = (
            self.cfg.lift_height_m
            if grasp_goal is None
            else grasp_goal.lift_criteria.height_m
        )
        if phase is BimanualPhase.LIFT and both_contact:
            goal_z = float(self._initial_box_pose[2]) + lift_height
            increments = self.cfg.lift_speed_m_s * self.cfg.dt * torch.arange(
                1, horizon + 1, dtype=torch.float64
            )
            box_pose[:, 2] = torch.clamp(
                snapshot.box.pose_b[2] + increments,
                max=goal_z,
            )
        elif phase is BimanualPhase.HOLD and both_contact:
            box_pose[:, 2] = float(self._initial_box_pose[2]) + lift_height
        elif phase in {BimanualPhase.LOWER, BimanualPhase.LOWER_SAFE}:
            goal_z = float(self._initial_box_pose[2])
            decrements = self.cfg.lower_speed_m_s * self.cfg.dt * torch.arange(
                1, horizon + 1, dtype=torch.float64
            )
            box_pose[:, 2] = torch.clamp(
                snapshot.box.pose_b[2] - decrements,
                min=goal_z,
            )

        preload_force = (
            self.cfg.preload_force_n
            if grasp_goal is None
            else grasp_goal.clamp_criteria.min_normal_force_n
        )
        grasp_force = (
            self.cfg.grasp_force_n
            if grasp_goal is None
            else grasp_goal.clamp_criteria.min_normal_force_n
        )
        if phase is BimanualPhase.PRELOAD:
            target_force = preload_force
        elif phase in {
            BimanualPhase.GRASP,
            BimanualPhase.LIFT,
            BimanualPhase.HOLD,
            BimanualPhase.LOWER,
            BimanualPhase.HOLD_SAFE,
            BimanualPhase.LOWER_SAFE,
        }:
            target_force = grasp_force
        else:
            target_force = 0.0
        return ManipulationTarget(
            box_pose_b=box_pose,
            left_palm_in_box=self._left_palm_in_box.clone(),
            right_palm_in_box=self._right_palm_in_box.clone(),
            target_normal_force_n=target_force,
            recovery_side=recovery_side,
            grasp_goal=grasp_goal,
        )
