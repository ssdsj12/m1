"""Deterministic orchestration of trajectory, distribution, WBC, and safety."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch

from .contracts import CONTROLLED_DOF, COORD_DOF, require_tensor
from .impedance import apply_impedance
from .motion_distribution import (
    MotionDistributionResult,
    distribute_motion,
)
from .qp_backend import DenseQpResult
from .safety import BalanceSafetySupervisor, SafetyCfg, SafetyState
from .standing_wbc import StandingWbcInput, StandingWbcResult, solve_standing_wbc
from .trajectory import BandLimitedPoseTrajectory, TrajectorySample


@dataclass(frozen=True)
class TeacherCfg:
    physics_dt: float = 0.005
    distribution_interval: int = 4
    arm_position_gain: float = 20.0
    arm_velocity_gain: float = 5.0


@dataclass(frozen=True)
class TeacherState:
    physics_step: int
    time_s: float
    ee_pose: torch.Tensor
    coordinated_jacobian: torch.Tensor
    coord_q: torch.Tensor
    coord_qd: torch.Tensor
    coord_q_min: torch.Tensor
    coord_q_max: torch.Tensor
    coord_v_max: torch.Tensor
    coord_a_max: torch.Tensor
    manipulability_gradient: torch.Tensor
    sigma_min: torch.Tensor
    wbc_input: StandingWbcInput
    controlled_q: torch.Tensor
    controlled_qd: torch.Tensor
    roll: float
    pitch: float
    wheel_contact_count: int
    max_lateral_slip: float
    signals_finite: bool


@dataclass(frozen=True)
class TeacherCommand:
    effort: torch.Tensor
    q_des: torch.Tensor
    qd_des: torch.Tensor
    target_pose: torch.Tensor
    target_twist: torch.Tensor
    motion_distribution: MotionDistributionResult
    qp_result: DenseQpResult
    safety_state: SafetyState
    terminate: bool


class M1PandaWbcTeacher:
    """One-environment deterministic C0 Teacher without Isaac dependencies."""

    def __init__(
        self,
        *,
        kp: torch.Tensor,
        kd: torch.Tensor,
        effort_limit: torch.Tensor,
        safe_arm_target: torch.Tensor,
        cfg: TeacherCfg | None = None,
        trajectory: BandLimitedPoseTrajectory | None = None,
        motion_distribution_fn: Callable | None = None,
        wbc_solver_fn: Callable[[StandingWbcInput], StandingWbcResult] | None = None,
    ):
        self.cfg = cfg or TeacherCfg()
        for name, value, trailing_shape in (
            ("kp", kp, (CONTROLLED_DOF,)),
            ("kd", kd, (CONTROLLED_DOF,)),
            ("effort_limit", effort_limit, (CONTROLLED_DOF,)),
            ("safe_arm_target", safe_arm_target, (7,)),
        ):
            require_tensor(name, value, trailing_shape=trailing_shape)
            if value.ndim != 1 or not value.is_floating_point():
                raise ValueError(f"{name} must be one floating vector")
        for name, value in (("kd", kd), ("effort_limit", effort_limit)):
            if value.dtype != kp.dtype or value.device != kp.device:
                raise ValueError(f"{name} dtype and device must match kp")
        if safe_arm_target.dtype != kp.dtype or safe_arm_target.device != kp.device:
            raise ValueError("safe_arm_target dtype and device must match kp")

        self._kp = kp.detach().clone()
        self._kd = kd.detach().clone()
        self._effort_limit = effort_limit.detach().clone()
        self._trajectory = trajectory or BandLimitedPoseTrajectory()
        self._motion_distribution_fn = (
            motion_distribution_fn or self._default_motion_distribution
        )
        self._wbc_solver_fn = wbc_solver_fn or solve_standing_wbc
        self._safety = BalanceSafetySupervisor(SafetyCfg(), safe_arm_target)
        self._initialized = False

    @staticmethod
    def _default_motion_distribution(**kwargs) -> MotionDistributionResult:
        kwargs.pop("physics_step")
        return distribute_motion(**kwargs)

    @staticmethod
    def _zero_distribution(state: TeacherState) -> MotionDistributionResult:
        return MotionDistributionResult(
            qd_coord=torch.zeros_like(state.coord_q),
            base_active=torch.tensor(False, device=state.coord_q.device),
            sigma_min=state.sigma_min.clone(),
            phi=state.coord_q.new_tensor(0.0),
            psi=state.coord_q.new_tensor(0.0),
            saturated=torch.zeros(
                COORD_DOF, dtype=torch.bool, device=state.coord_q.device
            ),
        )

    def _validate_state(self, state: TeacherState) -> None:
        if not isinstance(state, TeacherState):
            raise TypeError("state must be a TeacherState")
        if isinstance(state.physics_step, bool) or not isinstance(state.physics_step, int):
            raise TypeError("physics_step must be an integer")
        if state.physics_step < 0:
            raise ValueError("physics_step must be non-negative")
        if not math.isfinite(float(state.time_s)) or state.time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        require_tensor("ee_pose", state.ee_pose, trailing_shape=(6,))
        require_tensor(
            "controlled_q", state.controlled_q, trailing_shape=(CONTROLLED_DOF,)
        )
        require_tensor(
            "controlled_qd", state.controlled_qd, trailing_shape=(CONTROLLED_DOF,)
        )
        if state.controlled_q.shape != self._kp.shape:
            raise ValueError("controlled_q shape must match Teacher gains")
        if state.controlled_q.dtype != self._kp.dtype or state.controlled_q.device != self._kp.device:
            raise ValueError("controlled_q dtype and device must match Teacher gains")
        if state.controlled_qd.shape != state.controlled_q.shape:
            raise ValueError("controlled_qd shape must match controlled_q")

    def reset(self, state: TeacherState, *, seed: int) -> None:
        self._validate_state(state)
        self._trajectory.reset(state.ee_pose, seed=seed)
        self._safety.reset(state.controlled_q[-7:])
        self._coord_velocity = torch.zeros_like(state.coord_q)
        self._arm_target = state.controlled_q[-7:].detach().clone()
        self._last_verified_q_des = state.controlled_q.detach().clone()
        self._last_verified_qd_des = state.controlled_qd.detach().clone()
        self._last_distribution: MotionDistributionResult | None = None
        self._initialized = True

    def _motion_update(
        self, state: TeacherState, sample: TrajectorySample
    ) -> tuple[MotionDistributionResult, bool]:
        if self._safety.state == SafetyState.TRACK:
            twist_scale = 1.0
        elif self._safety.state == SafetyState.SCALE:
            twist_scale = self._safety.cfg.scaled_twist_factor
        else:
            twist_scale = 0.0
        try:
            result = self._motion_distribution_fn(
                physics_step=state.physics_step,
                coordinated_jacobian=state.coordinated_jacobian,
                pose_error=sample.pose - state.ee_pose,
                desired_twist=sample.twist * twist_scale,
                q=state.coord_q,
                qd=state.coord_qd,
                q_min=state.coord_q_min,
                q_max=state.coord_q_max,
                v_max=state.coord_v_max,
                a_max=state.coord_a_max,
                manipulability_gradient=state.manipulability_gradient,
                sigma_min=state.sigma_min,
                dt=self.cfg.physics_dt * self.cfg.distribution_interval,
            )
            require_tensor(
                "motion_distribution.qd_coord",
                result.qd_coord,
                trailing_shape=(COORD_DOF,),
            )
            return result, False
        except (RuntimeError, ValueError, TypeError):
            fallback = self._last_distribution or self._zero_distribution(state)
            return fallback, True

    def _build_wbc_input(
        self,
        state: TeacherState,
        *,
        arm_target_override: torch.Tensor | None = None,
        stop_wheels: bool = False,
    ) -> StandingWbcInput:
        base_acceleration = state.wbc_input.base_acceleration.clone()
        base_acceleration[0] = (
            self._coord_velocity[0] - state.coord_qd[0]
        ) / self.cfg.physics_dt
        base_acceleration[1] = (
            self._coord_velocity[1] - state.coord_qd[1]
        ) / self.cfg.physics_dt
        base_acceleration[5] = (
            self._coord_velocity[2] - state.coord_qd[2]
        ) / self.cfg.physics_dt
        if arm_target_override is None:
            arm_acceleration = (
                self._coord_velocity[3:] - state.controlled_qd[-7:]
            ) / self.cfg.physics_dt
        else:
            arm_acceleration = (
                self.cfg.arm_position_gain
                * (arm_target_override - state.controlled_q[-7:])
                - self.cfg.arm_velocity_gain * state.controlled_qd[-7:]
            )
        wheel_acceleration = state.wbc_input.wheel_acceleration.clone()
        if stop_wheels:
            wheel_acceleration = -state.controlled_qd[12:16] / self.cfg.physics_dt
        return replace(
            state.wbc_input,
            base_acceleration=base_acceleration,
            arm_acceleration=arm_acceleration,
            wheel_acceleration=wheel_acceleration,
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

    def step(self, state: TeacherState) -> TeacherCommand:
        if not self._initialized:
            raise RuntimeError("teacher must be reset before step")
        self._validate_state(state)
        sample = self._trajectory.sample(state.time_s)
        motion_failed = False
        if (
            self._last_distribution is None
            or state.physics_step % self.cfg.distribution_interval == 0
        ):
            distribution, motion_failed = self._motion_update(state, sample)
            self._last_distribution = distribution
            self._coord_velocity = distribution.qd_coord.detach().clone()
        distribution = self._last_distribution
        assert distribution is not None

        self._arm_target = (
            self._arm_target + self._coord_velocity[3:] * self.cfg.physics_dt
        )
        wbc_input = self._build_wbc_input(state)
        wbc_result = self._wbc_solver_fn(wbc_input)
        verified = self._wbc_is_verified(wbc_result) and not motion_failed
        decision = self._safety.update(
            roll=state.roll,
            pitch=state.pitch,
            wheel_contact_count=state.wheel_contact_count,
            max_lateral_slip=state.max_lateral_slip,
            qp_success=verified,
            signals_finite=state.signals_finite,
            current_arm_target=self._arm_target,
        )
        if decision.state >= SafetyState.HOLD:
            self._arm_target = decision.arm_target.detach().clone()
            override_input = self._build_wbc_input(
                state,
                arm_target_override=self._arm_target,
                stop_wheels=decision.stop_wheels,
            )
            wbc_result = self._wbc_solver_fn(override_input)
            verified = self._wbc_is_verified(wbc_result) and not motion_failed

        if verified:
            controlled_indices = torch.cat(
                (
                    state.wbc_input.leg_generalized_indices,
                    state.wbc_input.wheel_generalized_indices,
                    state.wbc_input.arm_generalized_indices,
                )
            ).to(device=wbc_result.qdd.device)
            qdd_controlled = wbc_result.qdd.index_select(0, controlled_indices).to(
                device=state.controlled_q.device, dtype=state.controlled_q.dtype
            )
            qd_des = state.controlled_qd + qdd_controlled * self.cfg.physics_dt
            q_des = state.controlled_q + qd_des * self.cfg.physics_dt
            q_des[-7:] = self._arm_target
            tau_ff = wbc_result.effort.to(
                device=state.controlled_q.device, dtype=state.controlled_q.dtype
            )
            effort = apply_impedance(
                state.controlled_q,
                state.controlled_qd,
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
            effort = apply_impedance(
                state.controlled_q,
                state.controlled_qd,
                q_des,
                qd_des,
                torch.zeros_like(state.controlled_q),
                self._kp,
                self._kd,
                self._effort_limit,
            )

        return TeacherCommand(
            effort=effort,
            q_des=q_des,
            qd_des=qd_des,
            target_pose=sample.pose,
            target_twist=sample.twist * decision.twist_scale,
            motion_distribution=distribution,
            qp_result=wbc_result.qp_result,
            safety_state=decision.state,
            terminate=decision.terminate,
        )
