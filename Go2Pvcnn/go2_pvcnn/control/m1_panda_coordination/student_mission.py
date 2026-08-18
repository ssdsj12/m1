"""Deployable C1a mission and nominal commands for Student S1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import CONTROLLED_DOF, require_tensor
from .rolling_contact import RollingContactCfg, wheel_speed_from_base_velocity
from .rolling_teacher import (
    LongitudinalCommandSchedule,
    LongitudinalScheduleCfg,
    PlanarBodyFrameTrajectory,
)
from .student_contracts import StudentNominalCommand
from .trajectory import BandLimitedTrajectoryCfg


@dataclass(frozen=True)
class StudentMissionSample:
    phase: int
    shaped_vx: float
    target_pose: torch.Tensor
    target_twist: torch.Tensor
    nominal: StudentNominalCommand


class StudentS1Mission:
    """One-environment deployable schedule, EE target, and nominal command."""

    def __init__(
        self,
        schedule_cfg: LongitudinalScheduleCfg | None = None,
        trajectory_cfg: BandLimitedTrajectoryCfg | None = None,
        rolling_contact_cfg: RollingContactCfg | None = None,
    ) -> None:
        self._schedule = LongitudinalCommandSchedule(schedule_cfg)
        self._trajectory = PlanarBodyFrameTrajectory(
            trajectory_cfg
            or BandLimitedTrajectoryCfg(
                position_amplitude=0.005,
                orientation_amplitude=0.01,
            )
        )
        self._rolling_contact_cfg = (
            rolling_contact_cfg or RollingContactCfg()
        )
        self._initialized = False

    @staticmethod
    def _validate_vector(
        name: str,
        value: torch.Tensor,
        width: int,
        reference: torch.Tensor | None = None,
    ) -> None:
        require_tensor(name, value, trailing_shape=(width,))
        if value.ndim != 1 or not value.dtype.is_floating_point:
            raise ValueError(f"{name} must have shape ({width},) and floating dtype")
        if reference is not None:
            if value.dtype != reference.dtype:
                raise TypeError(f"{name} dtype must match settled_controlled_q")
            if value.device != reference.device:
                raise ValueError(f"{name} device must match settled_controlled_q")

    def reset(
        self,
        center_pose: torch.Tensor,
        root_xy_yaw: torch.Tensor,
        *,
        settled_controlled_q: torch.Tensor,
        seed: int,
    ) -> None:
        self._validate_vector(
            "settled_controlled_q", settled_controlled_q, CONTROLLED_DOF
        )
        self._validate_vector(
            "center_pose", center_pose, 6, settled_controlled_q
        )
        self._validate_vector(
            "root_xy_yaw", root_xy_yaw, 3, settled_controlled_q
        )
        self._schedule.reset()
        self._trajectory.reset(center_pose, root_xy_yaw, seed=seed)
        self._settled_controlled_q = settled_controlled_q.detach().clone()
        self._initialized = True

    def sample(
        self,
        mission_step: int,
        root_xy_yaw: torch.Tensor,
        root_vxy_yawrate: torch.Tensor,
    ) -> StudentMissionSample:
        if not self._initialized:
            raise RuntimeError("mission must be reset before sampling")
        self._validate_vector(
            "root_xy_yaw", root_xy_yaw, 3, self._settled_controlled_q
        )
        self._validate_vector(
            "root_vxy_yawrate",
            root_vxy_yawrate,
            3,
            self._settled_controlled_q,
        )
        longitudinal = self._schedule.sample(mission_step)
        trajectory = self._trajectory.sample(
            mission_step * self._schedule.cfg.physics_dt,
            root_xy_yaw,
            root_vxy_yawrate,
        )
        position = self._settled_controlled_q.unsqueeze(0).clone()
        velocity = torch.zeros_like(position)
        vx = self._settled_controlled_q.new_tensor(
            longitudinal.shaped_target_mps
        )
        velocity[0, 12:16] = wheel_speed_from_base_velocity(
            vx, self._rolling_contact_cfg
        )
        return StudentMissionSample(
            phase=longitudinal.phase,
            shaped_vx=longitudinal.shaped_target_mps,
            target_pose=trajectory.pose,
            target_twist=trajectory.twist,
            nominal=StudentNominalCommand(
                position=position,
                velocity=velocity,
            ),
        )


__all__ = ["StudentMissionSample", "StudentS1Mission"]
