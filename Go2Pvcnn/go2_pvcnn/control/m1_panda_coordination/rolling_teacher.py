"""Deterministic command and trajectory primitives for rolling WBC control."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch

from .contracts import CONTROLLED_DOF, COORD_DOF, require_tensor
from .impedance import apply_impedance
from .motion_distribution import MotionDistributionResult, distribute_motion
from .qp_backend import DenseQpResult
from .rolling_contact import RollingContactCfg, wheel_speed_from_base_velocity
from .rolling_wbc import solve_rolling_wbc
from .safety import (
    BalanceSafetySupervisor,
    SafetyCfg,
    SafetyDecision,
    SafetyState,
)
from .standing_wbc import StandingWbcInput, StandingWbcResult
from .teacher import TeacherState
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


@dataclass(frozen=True)
class RollingTeacherCfg:
    physics_dt: float = 0.005
    distribution_interval: int = 4
    arm_position_gain: float = 20.0
    arm_velocity_gain: float = 5.0
    wheel_integral_gain: float = 5.0
    wheel_integral_limit: float = 2.0
    maximum_wheel_acceleration: float = 100.0


@dataclass(frozen=True)
class RollingTeacherState:
    mission_step: int
    teacher_state: TeacherState
    root_xy_yaw: torch.Tensor
    root_vxy_yawrate: torch.Tensor
    max_rolling_residual_mps: float


@dataclass(frozen=True)
class RollingTeacherCommand:
    effort: torch.Tensor
    q_des: torch.Tensor
    qd_des: torch.Tensor
    target_pose: torch.Tensor
    target_twist: torch.Tensor
    motion_distribution: MotionDistributionResult
    qp_result: DenseQpResult
    safety_state: SafetyState
    safety_reason: str
    motion_failure_reason: str | None
    phase: int
    raw_base_velocity_mps: float
    shaped_base_velocity_mps: float
    wheel_velocity_target: torch.Tensor
    terminate: bool


class M1PandaRollingWbcTeacher:
    """One-environment deterministic C1a rolling whole-body Teacher."""

    def __init__(
        self,
        *,
        kp: torch.Tensor,
        kd: torch.Tensor,
        effort_limit: torch.Tensor,
        safe_arm_target: torch.Tensor,
        cfg: RollingTeacherCfg | None = None,
        schedule_cfg: LongitudinalScheduleCfg | None = None,
        safety_cfg: SafetyCfg | None = None,
        rolling_contact_cfg: RollingContactCfg | None = None,
        trajectory: PlanarBodyFrameTrajectory | None = None,
        motion_distribution_fn: Callable | None = None,
        wbc_solver_fn: Callable[[StandingWbcInput], StandingWbcResult]
        | None = None,
    ):
        self.cfg = cfg or RollingTeacherCfg()
        for name, value in (
            ("physics_dt", self.cfg.physics_dt),
            ("arm_position_gain", self.cfg.arm_position_gain),
            ("arm_velocity_gain", self.cfg.arm_velocity_gain),
            ("wheel_integral_gain", self.cfg.wheel_integral_gain),
            ("wheel_integral_limit", self.cfg.wheel_integral_limit),
            ("maximum_wheel_acceleration", self.cfg.maximum_wheel_acceleration),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cfg.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")
        if (
            isinstance(self.cfg.distribution_interval, bool)
            or not isinstance(self.cfg.distribution_interval, int)
            or self.cfg.distribution_interval <= 0
        ):
            raise ValueError("distribution_interval must be a positive integer")
        for name, value, size in (
            ("kp", kp, CONTROLLED_DOF),
            ("kd", kd, CONTROLLED_DOF),
            ("effort_limit", effort_limit, CONTROLLED_DOF),
            ("safe_arm_target", safe_arm_target, 7),
        ):
            require_tensor(name, value, trailing_shape=(size,))
            if value.ndim != 1 or not value.is_floating_point():
                raise ValueError(f"{name} must be one floating vector")
        for name, value in (
            ("kd", kd),
            ("effort_limit", effort_limit),
            ("safe_arm_target", safe_arm_target),
        ):
            if value.dtype != kp.dtype or value.device != kp.device:
                raise ValueError(f"{name} dtype and device must match kp")

        self._kp = kp.detach().clone()
        self._kd = kd.detach().clone()
        self._effort_limit = effort_limit.detach().clone()
        self._schedule = LongitudinalCommandSchedule(schedule_cfg)
        self._trajectory = trajectory or PlanarBodyFrameTrajectory(
            BandLimitedTrajectoryCfg(
                position_amplitude=0.005,
                orientation_amplitude=0.01,
            )
        )
        self._rolling_contact_cfg = rolling_contact_cfg or RollingContactCfg()
        self._motion_distribution_fn = (
            motion_distribution_fn or distribute_motion
        )
        self._wbc_solver_fn = wbc_solver_fn or solve_rolling_wbc
        self._safety = BalanceSafetySupervisor(
            safety_cfg or SafetyCfg(), safe_arm_target
        )
        self._initialized = False

    @staticmethod
    def _validate_state(state: RollingTeacherState) -> None:
        if not isinstance(state, RollingTeacherState):
            raise TypeError("state must be a RollingTeacherState")
        if isinstance(state.mission_step, bool) or not isinstance(
            state.mission_step, int
        ):
            raise TypeError("mission_step must be an integer")
        if state.mission_step < 0:
            raise ValueError("mission_step must be non-negative")
        if not isinstance(state.teacher_state, TeacherState):
            raise TypeError("teacher_state must be a TeacherState")
        for name, value in (
            ("root_xy_yaw", state.root_xy_yaw),
            ("root_vxy_yawrate", state.root_vxy_yawrate),
        ):
            require_tensor(name, value, trailing_shape=(3,))
            if value.ndim != 1:
                raise ValueError(f"{name} must be one 3-vector")
            if value.dtype != state.teacher_state.controlled_q.dtype:
                raise TypeError(f"{name} dtype must match controlled_q")
            if value.device != state.teacher_state.controlled_q.device:
                raise ValueError(f"{name} device must match controlled_q")
        if not math.isfinite(float(state.max_rolling_residual_mps)):
            raise ValueError("max_rolling_residual_mps must be finite")

    @staticmethod
    def _zero_distribution(
        state: RollingTeacherState, base_velocity: torch.Tensor
    ) -> MotionDistributionResult:
        qd = torch.zeros_like(state.teacher_state.coord_q)
        qd[:3] = base_velocity
        return MotionDistributionResult(
            qd_coord=qd,
            base_active=torch.tensor(
                bool(torch.any(base_velocity != 0.0).item()),
                dtype=torch.bool,
                device=qd.device,
            ),
            sigma_min=state.teacher_state.sigma_min.clone(),
            phi=qd.new_tensor(0.0),
            psi=qd.new_tensor(0.0),
            saturated=torch.zeros(COORD_DOF, dtype=torch.bool, device=qd.device),
        )

    @staticmethod
    def _wbc_is_verified(result: StandingWbcResult) -> bool:
        return bool(
            result.qp_result.success
            and result.effort is not None
            and torch.isfinite(result.effort).all().item()
            and torch.isfinite(result.qdd).all().item()
            and torch.isfinite(result.contact_force).all().item()
        )

    def reset(self, state: RollingTeacherState, *, seed: int) -> None:
        self._validate_state(state)
        teacher_state = state.teacher_state
        self._schedule.reset()
        self._trajectory.reset(
            teacher_state.ee_pose, state.root_xy_yaw, seed=seed
        )
        self._trajectory_seed = seed
        self._trajectory_time_s = 0.0
        self._safety.reset(teacher_state.controlled_q[-7:])
        self._last_safety_decision: SafetyDecision | None = None
        self._coord_velocity = torch.zeros_like(teacher_state.coord_q)
        self._leg_target = teacher_state.controlled_q[:12].detach().clone()
        self._arm_target = teacher_state.controlled_q[-7:].detach().clone()
        self._wheel_velocity_integral = torch.zeros(
            4,
            dtype=teacher_state.controlled_q.dtype,
            device=teacher_state.controlled_q.device,
        )
        self._last_verified_q_des = teacher_state.controlled_q.detach().clone()
        self._last_verified_qd_des = teacher_state.controlled_qd.detach().clone()
        self._last_distribution: MotionDistributionResult | None = None
        self._initialized = True

    def restart_mission(
        self, state: RollingTeacherState, *, seed: int
    ) -> None:
        """Restart scored commands without disturbing settled low-level state."""

        if not self._initialized:
            raise RuntimeError("teacher must be reset before mission restart")
        self._validate_state(state)
        self._schedule.reset()
        self._trajectory.reset(
            state.teacher_state.ee_pose, state.root_xy_yaw, seed=seed
        )
        self._trajectory_seed = seed
        self._trajectory_time_s = 0.0
        self._safety.reset(self._arm_target)
        self._last_safety_decision = None

    def _prior_scale(self) -> tuple[float, float]:
        if self._last_safety_decision is None:
            return 1.0, 1.0
        return (
            self._last_safety_decision.base_velocity_scale,
            self._last_safety_decision.twist_scale,
        )

    def _motion_update(
        self,
        state: RollingTeacherState,
        sample: TrajectorySample,
        base_velocity: torch.Tensor,
        twist_scale: float,
    ) -> tuple[MotionDistributionResult, str | None]:
        teacher_state = state.teacher_state
        try:
            result = self._motion_distribution_fn(
                coordinated_jacobian=teacher_state.coordinated_jacobian,
                pose_error=sample.pose - teacher_state.ee_pose,
                desired_twist=sample.twist * twist_scale,
                prescribed_base_velocity=base_velocity,
                q=teacher_state.coord_q,
                qd=teacher_state.coord_qd,
                q_min=teacher_state.coord_q_min,
                q_max=teacher_state.coord_q_max,
                v_max=teacher_state.coord_v_max,
                a_max=teacher_state.coord_a_max,
                manipulability_gradient=teacher_state.manipulability_gradient,
                sigma_min=teacher_state.sigma_min,
                dt=self.cfg.physics_dt * self.cfg.distribution_interval,
            )
            require_tensor(
                "motion_distribution.qd_coord",
                result.qd_coord,
                trailing_shape=(COORD_DOF,),
            )
            return result, None
        except (RuntimeError, ValueError, TypeError) as error:
            fallback = self._last_distribution or self._zero_distribution(
                state, base_velocity
            )
            qd = fallback.qd_coord.clone()
            qd[:3] = base_velocity
            return replace(fallback, qd_coord=qd), f"{type(error).__name__}: {error}"

    def _build_wbc_input(
        self,
        state: RollingTeacherState,
        base_velocity: torch.Tensor,
        wheel_velocity: torch.Tensor,
        *,
        arm_target_override: torch.Tensor | None = None,
    ) -> StandingWbcInput:
        teacher_state = state.teacher_state
        horizon = self.cfg.physics_dt * self.cfg.distribution_interval
        base_acceleration = teacher_state.wbc_input.base_acceleration.clone()
        base_acceleration[0] = (
            base_velocity[0] - state.root_vxy_yawrate[0]
        ) / horizon
        base_acceleration[1] = -state.root_vxy_yawrate[1] / horizon
        base_acceleration[5] = -state.root_vxy_yawrate[2] / horizon
        wheel_acceleration = torch.clamp(
            (
                wheel_velocity
                - teacher_state.controlled_qd[12:16]
            )
            / horizon,
            min=-self.cfg.maximum_wheel_acceleration,
            max=self.cfg.maximum_wheel_acceleration,
        )
        if arm_target_override is None:
            arm_acceleration = (
                self._coord_velocity[3:]
                - teacher_state.controlled_qd[-7:]
            ) / horizon
        else:
            arm_acceleration = (
                self.cfg.arm_position_gain
                * (arm_target_override - teacher_state.controlled_q[-7:])
                - self.cfg.arm_velocity_gain
                * teacher_state.controlled_qd[-7:]
            )
        return replace(
            teacher_state.wbc_input,
            base_acceleration=base_acceleration,
            wheel_acceleration=wheel_acceleration,
            arm_acceleration=arm_acceleration,
        )

    def step(self, state: RollingTeacherState) -> RollingTeacherCommand:
        if not self._initialized:
            raise RuntimeError("teacher must be reset before step")
        self._validate_state(state)
        teacher_state = state.teacher_state
        base_scale, twist_scale = self._prior_scale()
        longitudinal = self._schedule.sample(state.mission_step, base_scale)
        base_velocity = teacher_state.coord_q.new_tensor(
            (longitudinal.shaped_target_mps, 0.0, 0.0)
        )
        wheel_velocity = wheel_speed_from_base_velocity(
            base_velocity[0], self._rolling_contact_cfg
        )
        sample = self._trajectory.sample(
            self._trajectory_time_s,
            state.root_xy_yaw,
            base_velocity,
        )
        prior_safety_state = self._safety.state
        high_level_hold = prior_safety_state >= SafetyState.HOLD
        motion_failure_reason = None
        if high_level_hold:
            distribution = self._zero_distribution(state, base_velocity)
            self._last_distribution = distribution
            self._coord_velocity = distribution.qd_coord.clone()
        elif (
            self._last_distribution is None
            or state.mission_step % self.cfg.distribution_interval == 0
        ):
            distribution, motion_failure_reason = self._motion_update(
                state, sample, base_velocity, twist_scale
            )
            self._last_distribution = distribution
            self._coord_velocity = distribution.qd_coord.detach().clone()
        else:
            distribution = self._last_distribution
            qd = distribution.qd_coord.clone()
            qd[:3] = base_velocity
            distribution = replace(
                distribution,
                qd_coord=qd,
                base_active=torch.tensor(
                    bool(torch.any(base_velocity != 0.0).item()),
                    dtype=torch.bool,
                    device=qd.device,
                ),
            )

        if not high_level_hold:
            horizon = self.cfg.physics_dt * self.cfg.distribution_interval
            self._arm_target = (
                teacher_state.controlled_q[-7:]
                + self._coord_velocity[3:] * horizon
            )

        wbc_input = self._build_wbc_input(
            state, base_velocity, wheel_velocity
        )
        wbc_result = self._wbc_solver_fn(wbc_input)
        verified = (
            self._wbc_is_verified(wbc_result)
            and motion_failure_reason is None
        )
        decision = self._safety.update(
            roll=teacher_state.roll,
            pitch=teacher_state.pitch,
            wheel_contact_count=teacher_state.wheel_contact_count,
            max_lateral_slip=teacher_state.max_lateral_slip,
            qp_success=verified,
            signals_finite=teacher_state.signals_finite,
            current_arm_target=self._arm_target,
            max_rolling_residual=state.max_rolling_residual_mps,
            base_speed=float(state.root_vxy_yawrate[0].item()),
        )
        self._last_safety_decision = decision

        entered_hold = (
            decision.state >= SafetyState.HOLD
            and prior_safety_state < SafetyState.HOLD
        )
        recovered_hold = (
            decision.state < SafetyState.HOLD
            and prior_safety_state >= SafetyState.HOLD
        )
        if entered_hold or recovered_hold:
            self._trajectory.reset(
                teacher_state.ee_pose,
                state.root_xy_yaw,
                seed=self._trajectory_seed,
            )
            self._trajectory_time_s = 0.0
            sample = self._trajectory.sample(
                0.0,
                state.root_xy_yaw,
                base_velocity,
            )
        if recovered_hold:
            self._arm_target = teacher_state.controlled_q[-7:].detach().clone()
        if decision.state >= SafetyState.HOLD:
            self._arm_target = decision.arm_target.detach().clone()
            wbc_input = self._build_wbc_input(
                state,
                base_velocity,
                wheel_velocity,
                arm_target_override=self._arm_target,
            )
            wbc_result = self._wbc_solver_fn(wbc_input)
            verified = (
                self._wbc_is_verified(wbc_result)
                and motion_failure_reason is None
            )
        else:
            self._trajectory_time_s += self.cfg.physics_dt

        controlled_indices = torch.cat(
            (
                teacher_state.wbc_input.leg_generalized_indices,
                teacher_state.wbc_input.wheel_generalized_indices,
                teacher_state.wbc_input.arm_generalized_indices,
            )
        ).to(device=teacher_state.controlled_q.device)
        bias_effort = teacher_state.wbc_input.bias_force.index_select(
            0, controlled_indices
        ).to(
            device=teacher_state.controlled_q.device,
            dtype=teacher_state.controlled_q.dtype,
        )
        if verified:
            qdd_controlled = wbc_result.qdd.index_select(
                0, controlled_indices
            ).to(
                device=teacher_state.controlled_q.device,
                dtype=teacher_state.controlled_q.dtype,
            )
            qd_des = (
                teacher_state.controlled_qd
                + qdd_controlled * self.cfg.physics_dt
            )
            qd_des[:12] = 0.0
            qd_des[12:16] = wheel_velocity
            qd_des[-7:] = (
                torch.zeros_like(self._coord_velocity[3:])
                if decision.state >= SafetyState.HOLD
                else self._coord_velocity[3:]
            )
            q_des = (
                teacher_state.controlled_q
                + qd_des * self.cfg.physics_dt
            )
            q_des[:12] = self._leg_target
            q_des[-7:] = self._arm_target
            tau_ff = wbc_result.effort.to(
                device=teacher_state.controlled_q.device,
                dtype=teacher_state.controlled_q.dtype,
            )
            tau_ff[-7:] = bias_effort[-7:]
            effort = apply_impedance(
                teacher_state.controlled_q,
                teacher_state.controlled_qd,
                q_des,
                qd_des,
                tau_ff,
                self._kp,
                self._kd,
                self._effort_limit,
            )
            self._last_verified_q_des = q_des.detach().clone()
            self._last_verified_qd_des = qd_des.detach().clone()
        else:
            q_des = self._last_verified_q_des.clone()
            qd_des = self._last_verified_qd_des.clone()
            fallback_feedforward = torch.zeros_like(
                teacher_state.controlled_q
            )
            fallback_feedforward[-7:] = bias_effort[-7:]
            effort = apply_impedance(
                teacher_state.controlled_q,
                teacher_state.controlled_qd,
                q_des,
                qd_des,
                fallback_feedforward,
                self._kp,
                self._kd,
                self._effort_limit,
            )

        self._wheel_velocity_integral = torch.clamp(
            self._wheel_velocity_integral
            + (
                wheel_velocity
                - teacher_state.controlled_qd[12:16]
            )
            * self.cfg.physics_dt,
            min=-self.cfg.wheel_integral_limit,
            max=self.cfg.wheel_integral_limit,
        )
        effort = effort.clone()
        effort[12:16] = torch.clamp(
            effort[12:16]
            + self.cfg.wheel_integral_gain
            * self._wheel_velocity_integral,
            min=-self._effort_limit[12:16],
            max=self._effort_limit[12:16],
        )
        return RollingTeacherCommand(
            effort=effort,
            q_des=q_des,
            qd_des=qd_des,
            target_pose=sample.pose,
            target_twist=sample.twist * decision.twist_scale,
            motion_distribution=distribution,
            qp_result=wbc_result.qp_result,
            safety_state=decision.state,
            safety_reason=decision.reason,
            motion_failure_reason=motion_failure_reason,
            phase=longitudinal.phase,
            raw_base_velocity_mps=longitudinal.raw_target_mps,
            shaped_base_velocity_mps=longitudinal.shaped_target_mps,
            wheel_velocity_target=wheel_velocity,
            terminate=decision.terminate,
        )
