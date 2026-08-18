"""Deterministic command and trajectory primitives for rolling WBC control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import require_tensor
from .trajectory import (
    BandLimitedPoseTrajectory,
    BandLimitedTrajectoryCfg,
    TrajectorySample,
)


@dataclass(frozen=True)
class LongitudinalScheduleCfg:
    """Five-phase C1a speed schedule and its physical slew limit."""

    physics_dt: float = 0.005
    phase_steps: int = 800
    phase_targets_mps: tuple[float, ...] = (0.0, 0.05, 0.10, 0.0, -0.05)
    max_acceleration_mps2: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be finite and positive")
        if isinstance(self.phase_steps, bool) or not isinstance(
            self.phase_steps, int
        ) or self.phase_steps <= 0:
            raise ValueError("phase_steps must be a positive integer")
        if len(self.phase_targets_mps) != 5 or any(
            not math.isfinite(target) for target in self.phase_targets_mps
        ):
            raise ValueError("phase_targets_mps must contain five finite values")
        if (
            not math.isfinite(self.max_acceleration_mps2)
            or self.max_acceleration_mps2 <= 0.0
        ):
            raise ValueError("max_acceleration_mps2 must be finite and positive")


@dataclass(frozen=True)
class LongitudinalCommand:
    phase: int
    raw_target_mps: float
    shaped_target_mps: float


class LongitudinalCommandSchedule:
    """Generate exactly one rate-limited longitudinal command per mission step."""

    def __init__(self, cfg: LongitudinalScheduleCfg | None = None):
        self.cfg = cfg or LongitudinalScheduleCfg()
        self.reset()

    def reset(self) -> None:
        self._shaped_target_mps = 0.0
        self._last_step = -1

    def sample(
        self, mission_step: int, safety_scale: float = 1.0
    ) -> LongitudinalCommand:
        if isinstance(mission_step, bool) or not isinstance(mission_step, int):
            raise TypeError("mission_step must be an integer")
        if mission_step != self._last_step + 1:
            raise ValueError("mission_step must advance exactly once")
        if (
            not math.isfinite(float(safety_scale))
            or float(safety_scale) < 0.0
            or float(safety_scale) > 1.0
        ):
            raise ValueError("safety_scale must be finite and in [0, 1]")

        phase = min(
            mission_step // self.cfg.phase_steps,
            len(self.cfg.phase_targets_mps) - 1,
        )
        raw = self.cfg.phase_targets_mps[phase]
        requested = float(safety_scale) * raw
        maximum_delta = (
            self.cfg.max_acceleration_mps2 * self.cfg.physics_dt
        )
        delta = max(
            -maximum_delta,
            min(maximum_delta, requested - self._shaped_target_mps),
        )
        self._shaped_target_mps += delta
        self._last_step = mission_step
        return LongitudinalCommand(phase, raw, self._shaped_target_mps)


def _rotation_2d(yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        )
    )


class PlanarBodyFrameTrajectory:
    """Advect a band-limited local EE trajectory with planar root motion."""

    def __init__(self, cfg: BandLimitedTrajectoryCfg | None = None):
        self._trajectory = BandLimitedPoseTrajectory(cfg)
        self._reset_root_yaw: torch.Tensor | None = None

    @staticmethod
    def _validate_root(
        name: str, value: torch.Tensor, reference: torch.Tensor
    ) -> None:
        require_tensor(name, value, trailing_shape=(3,))
        if value.ndim != 1:
            raise ValueError(f"{name} must be one 3-vector")
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} dtype must match pose")
        if value.device != reference.device:
            raise ValueError(f"{name} device must match pose")

    def reset(
        self,
        center_pose: torch.Tensor,
        root_xy_yaw: torch.Tensor,
        *,
        seed: int,
    ) -> None:
        require_tensor("center_pose", center_pose, trailing_shape=(6,))
        if center_pose.ndim != 1 or not center_pose.is_floating_point():
            raise ValueError("center_pose must be one floating 6-vector")
        self._validate_root("root_xy_yaw", root_xy_yaw, center_pose)

        yaw = root_xy_yaw[2]
        local_center = center_pose.detach().clone()
        world_offset = center_pose[:2] - root_xy_yaw[:2]
        local_center[:2] = _rotation_2d(-yaw) @ world_offset
        self._trajectory.reset(local_center, seed=seed)
        self._reset_root_yaw = yaw.detach().clone()

    def sample(
        self,
        time_s: float,
        root_xy_yaw: torch.Tensor,
        root_vxy_yawrate: torch.Tensor,
    ) -> TrajectorySample:
        if self._reset_root_yaw is None:
            raise RuntimeError("trajectory must be reset before sampling")
        self._validate_root("root_xy_yaw", root_xy_yaw, self._reset_root_yaw)
        self._validate_root(
            "root_vxy_yawrate", root_vxy_yawrate, self._reset_root_yaw
        )

        local = self._trajectory.sample(time_s)
        yaw = root_xy_yaw[2]
        yaw_rate = root_vxy_yawrate[2]
        rotation = _rotation_2d(yaw)
        offset_world_xy = rotation @ local.pose[:2]
        local_velocity_world = rotation @ local.twist[:2]
        tangential_velocity = yaw_rate * torch.stack(
            (-offset_world_xy[1], offset_world_xy[0])
        )

        pose = local.pose.clone()
        pose[:2] = root_xy_yaw[:2] + offset_world_xy
        pose[5] += yaw - self._reset_root_yaw

        twist = local.twist.clone()
        twist[:2] = (
            root_vxy_yawrate[:2]
            + local_velocity_world
            + tangential_velocity
        )
        twist[5] += yaw_rate

        acceleration = local.acceleration.clone()
        acceleration[:2] = (
            rotation @ local.acceleration[:2]
            + 2.0
            * yaw_rate
            * torch.stack((-local_velocity_world[1], local_velocity_world[0]))
            - yaw_rate.square() * offset_world_xy
        )
        return TrajectorySample(
            pose=pose,
            twist=twist,
            acceleration=acceleration,
        )
