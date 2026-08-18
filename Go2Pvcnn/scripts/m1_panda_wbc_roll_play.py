#!/usr/bin/env python3
"""Run the deterministic C1a rolling Teacher on the combined M1 + Panda."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import sys
import traceback

import torch


THIS_FILE = Path(__file__).resolve()
SCRIPTS_ROOT = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
for path in (SCRIPTS_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m1_panda_wbc_play import (
    MAX_CONTINUOUS_ARM_TARGET_STEP_RAD,
    PhysxTeacherAdapter,
    atomic_write_summary,
    build_teacher_gains,
    _termination_cause,
)
from go2_pvcnn.control.m1_panda_coordination.rolling_contact import (
    RollingContactMetrics,
    rolling_contact_metrics,
)
from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import (
    M1PandaRollingWbcTeacher,
    PlanarBodyFrameTrajectory,
    RollingTeacherState,
)
from go2_pvcnn.control.m1_panda_coordination.safety import SafetyState
from go2_pvcnn.control.m1_panda_coordination.trajectory import (
    BandLimitedTrajectoryCfg,
)


TASK_ID = "Isaac-M1-Panda-Wbc-Teacher-C1a-v0"
PHYSICS_DT = 0.005
SETTLE_STEPS = 100
MAX_SETTLE_STEPS = 600
SETTLE_STABLE_SAMPLES = 20
SETTLE_MAX_BASE_SPEED_MPS = 0.02
SETTLE_MAX_CONTACT_SPEED_MPS = 0.04
MISSION_STEPS = 4000
PHASE_STEPS = 800
WHEEL_RADIUS_M = 0.095


@dataclass
class SettlingGate:
    """Start scoring only after the composite is continuously near rest."""

    minimum_steps: int = SETTLE_STEPS
    required_stable_samples: int = SETTLE_STABLE_SAMPLES
    maximum_steps: int = MAX_SETTLE_STEPS
    max_base_speed_mps: float = SETTLE_MAX_BASE_SPEED_MPS
    max_contact_speed_mps: float = SETTLE_MAX_CONTACT_SPEED_MPS
    _stable_samples: int = field(default=0, init=False, repr=False)

    def update(
        self,
        *,
        physics_step: int,
        heading_speed_mps: float,
        rolling_residual_mps: float,
        lateral_slip_mps: float,
        wheel_contact_count: int,
        base_contact: int,
        signals_finite: bool,
    ) -> bool:
        stable = bool(
            physics_step >= self.minimum_steps
            and signals_finite
            and wheel_contact_count == 4
            and base_contact == 0
            and abs(float(heading_speed_mps)) <= self.max_base_speed_mps
            and abs(float(rolling_residual_mps))
            <= self.max_contact_speed_mps
            and abs(float(lateral_slip_mps))
            <= self.max_contact_speed_mps
        )
        self._stable_samples = self._stable_samples + 1 if stable else 0
        if self._stable_samples >= self.required_stable_samples:
            return True
        if physics_step >= self.maximum_steps:
            raise RuntimeError(
                "settling did not converge before the maximum step"
            )
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Play the deterministic M1 + Panda rolling WBC Teacher."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Mission steps; zero runs until the application closes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--disable-target-motion", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--stats-interval", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_args(args) -> None:
    if args.steps < 0 or args.steps > MISSION_STEPS:
        raise ValueError(f"--steps must be in [0, {MISSION_STEPS}]")
    if args.num_envs != 1:
        raise ValueError("C1a supports exactly one environment")
    if args.stats_interval <= 0:
        raise ValueError("--stats-interval must be positive")


@dataclass
class C1aSummary:
    seed: int
    requested_steps: int
    steps: int = 0
    finite: bool = True
    qp_feasible_count: int = 0
    vx_squared_error_sum: float = 0.0
    phase_counts: dict[str, int] = field(default_factory=dict)
    forward_displacement_m: float = 0.0
    reverse_displacement_m: float = 0.0
    stop_settle_time_s: float | None = None
    max_rolling_residual_mps: float = 0.0
    max_lateral_slip_mps: float = 0.0
    min_wheel_contact_count: int = 4
    max_wheel_velocity_spread_radps: float = 0.0
    wheel_effort_saturation_count: int = 0
    wheel_direction_mismatch_count: int = 0
    max_qp_equality_residual: float = 0.0
    max_qp_inequality_violation: float = 0.0
    max_ee_position_error_m: float = 0.0
    min_singular_value: float = math.inf
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    joint_limit_violations: int = 0
    base_contacts: int = 0
    non_finite_count: int = 0
    arm_snap_count: int = 0
    max_arm_target_step_rad: float = 0.0
    reset_count: int = 0
    safety_state_counts: dict[str, int] = field(default_factory=dict)
    safety_reason_counts: dict[str, int] = field(default_factory=dict)
    track_scale_count: int = 0
    hold_or_worse_count: int = 0
    exit_reason: str = "not_started"
    _previous_arm_target: torch.Tensor | None = field(default=None, repr=False)
    _stop_phase_start_step: int | None = field(default=None, repr=False)

    @property
    def completed_phase_count(self) -> int:
        return sum(self.phase_counts.get(str(index), 0) > 0 for index in range(5))

    @property
    def vx_rmse_mps(self) -> float:
        return math.sqrt(self.vx_squared_error_sum / max(self.steps, 1))

    def to_dict(self) -> dict:
        denominator = max(self.steps, 1)
        sigma = self.min_singular_value
        if not math.isfinite(sigma):
            sigma = 0.0
        payload = {
            "seed": self.seed,
            "requested_steps": self.requested_steps,
            "steps": self.steps,
            "finite": self.finite,
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "completed_phase_count": self.completed_phase_count,
            "vx_rmse_mps": self.vx_rmse_mps,
            "forward_displacement_m": self.forward_displacement_m,
            "reverse_displacement_m": self.reverse_displacement_m,
            "stop_settle_time_s": self.stop_settle_time_s,
            "max_rolling_residual_mps": self.max_rolling_residual_mps,
            "max_lateral_slip_mps": self.max_lateral_slip_mps,
            "min_wheel_contact_count": self.min_wheel_contact_count,
            "max_wheel_velocity_spread_radps": self.max_wheel_velocity_spread_radps,
            "wheel_effort_saturation_count": self.wheel_effort_saturation_count,
            "wheel_direction_mismatch_count": self.wheel_direction_mismatch_count,
            "qp_feasible_count": self.qp_feasible_count,
            "qp_feasible_rate": self.qp_feasible_count / denominator,
            "max_qp_equality_residual": self.max_qp_equality_residual,
            "max_qp_inequality_violation": self.max_qp_inequality_violation,
            "max_ee_position_error_m": self.max_ee_position_error_m,
            "min_singular_value": sigma,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "joint_limit_violations": self.joint_limit_violations,
            "base_contacts": self.base_contacts,
            "non_finite_count": self.non_finite_count,
            "arm_snap_count": self.arm_snap_count,
            "max_arm_target_step_rad": self.max_arm_target_step_rad,
            "reset_count": self.reset_count,
            "safety_state_counts": dict(sorted(self.safety_state_counts.items())),
            "safety_reason_counts": dict(sorted(self.safety_reason_counts.items())),
            "track_scale_count": self.track_scale_count,
            "hold_or_worse_count": self.hold_or_worse_count,
            "exit_reason": self.exit_reason,
        }
        payload["hard_gates_passed"] = formal_hard_gates_pass(self)
        return payload


def formal_hard_gates_pass(summary: C1aSummary) -> bool:
    denominator = max(summary.steps, 1)
    gates = (
        summary.requested_steps == MISSION_STEPS,
        summary.steps == MISSION_STEPS,
        summary.completed_phase_count == 5,
        summary.finite,
        summary.vx_rmse_mps <= 0.03,
        summary.stop_settle_time_s is not None,
        summary.stop_settle_time_s is not None
        and summary.stop_settle_time_s <= 1.0,
        summary.forward_displacement_m > 0.0,
        summary.reverse_displacement_m < 0.0,
        summary.max_rolling_residual_mps <= 0.05,
        summary.max_lateral_slip_mps <= 0.05,
        summary.min_wheel_contact_count == 4,
        summary.max_abs_roll_rad <= math.radians(10.0),
        summary.max_abs_pitch_rad <= math.radians(10.0),
        summary.max_ee_position_error_m <= 0.03,
        summary.wheel_direction_mismatch_count == 0,
        summary.qp_feasible_count / denominator >= 0.999,
        summary.track_scale_count / denominator >= 0.99,
        summary.hold_or_worse_count == 0,
        summary.joint_limit_violations == 0,
        summary.base_contacts == 0,
        summary.non_finite_count == 0,
        summary.arm_snap_count == 0,
        summary.reset_count == 0,
        summary.exit_reason == "steps_complete",
    )
    return all(gates)


def smoke_gates_pass(summary: C1aSummary) -> bool:
    denominator = max(summary.steps, 1)
    return bool(
        summary.steps == summary.requested_steps
        and summary.steps > 0
        and summary.finite
        and summary.qp_feasible_count / denominator >= 0.999
        and summary.min_wheel_contact_count == 4
        and summary.base_contacts == 0
        and summary.non_finite_count == 0
        and summary.reset_count == 0
        and summary.hold_or_worse_count == 0
        and summary.exit_reason == "smoke_complete"
    )


class RollingPhysxTeacherAdapter(PhysxTeacherAdapter):
    """Extend the C0 live adapter with bottom-point rolling measurements."""

    def build_rolling_state(
        self, physics_step: int, mission_step: int
    ) -> tuple[RollingTeacherState, int, RollingContactMetrics]:
        teacher_state, base_contact = super().build_state(physics_step)
        metrics = rolling_contact_metrics(
            self.latest_contact_jacobian,
            self.latest_generalized_velocity,
            yaw=float(self.latest_root_xy_yaw[2].item()),
        )
        teacher_state = replace(
            teacher_state,
            max_lateral_slip=metrics.max_lateral_slip_mps,
        )
        return (
            RollingTeacherState(
                mission_step=mission_step,
                teacher_state=teacher_state,
                root_xy_yaw=self.latest_root_xy_yaw.clone(),
                root_vxy_yawrate=self.latest_root_vxy_yawrate.clone(),
                max_rolling_residual_mps=(
                    metrics.max_longitudinal_residual_mps
                ),
            ),
            base_contact,
            metrics,
        )


def _heading_velocity(state: RollingTeacherState) -> tuple[float, float]:
    yaw = float(state.root_xy_yaw[2].item())
    vx_w = float(state.root_vxy_yawrate[0].item())
    vy_w = float(state.root_vxy_yawrate[1].item())
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * vx_w + sine * vy_w, -sine * vx_w + cosine * vy_w


def _update_summary(
    summary: C1aSummary,
    state: RollingTeacherState,
    command,
    base_contact: int,
    adapter: RollingPhysxTeacherAdapter,
    reset_cause: str,
) -> None:
    teacher_state = state.teacher_state
    vx, lateral_velocity = _heading_velocity(state)
    phase_key = str(command.phase)
    summary.steps += 1
    summary.phase_counts[phase_key] = summary.phase_counts.get(phase_key, 0) + 1
    summary.vx_squared_error_sum += (
        vx - command.shaped_base_velocity_mps
    ) ** 2
    if command.raw_base_velocity_mps > 0.0:
        summary.forward_displacement_m += vx * PHYSICS_DT
    elif command.raw_base_velocity_mps < 0.0:
        summary.reverse_displacement_m += vx * PHYSICS_DT
    if command.phase == 3 and summary._stop_phase_start_step is None:
        summary._stop_phase_start_step = summary.steps - 1
    if (
        command.phase == 3
        and summary.stop_settle_time_s is None
        and abs(vx) <= 0.02
        and summary._stop_phase_start_step is not None
    ):
        summary.stop_settle_time_s = (
            summary.steps - 1 - summary._stop_phase_start_step
        ) * PHYSICS_DT

    command_finite = bool(torch.isfinite(command.effort).all().item())
    summary.finite = summary.finite and teacher_state.signals_finite and command_finite
    summary.non_finite_count += int(
        not teacher_state.signals_finite or not command_finite
    )
    summary.qp_feasible_count += int(command.qp_result.success)
    summary.max_qp_equality_residual = max(
        summary.max_qp_equality_residual,
        float(command.qp_result.max_equality_residual),
    )
    summary.max_qp_inequality_violation = max(
        summary.max_qp_inequality_violation,
        float(command.qp_result.max_inequality_violation),
    )
    summary.max_rolling_residual_mps = max(
        summary.max_rolling_residual_mps,
        state.max_rolling_residual_mps,
    )
    summary.max_lateral_slip_mps = max(
        summary.max_lateral_slip_mps,
        teacher_state.max_lateral_slip,
        abs(lateral_velocity),
    )
    summary.min_wheel_contact_count = min(
        summary.min_wheel_contact_count,
        teacher_state.wheel_contact_count,
    )
    wheel_velocity = adapter.latest_wheel_velocity
    summary.max_wheel_velocity_spread_radps = max(
        summary.max_wheel_velocity_spread_radps,
        float((wheel_velocity.max() - wheel_velocity.min()).abs().item()),
    )
    wheel_effort = command.effort[12:16].abs()
    wheel_limit = teacher_state.wbc_input.effort_limit[12:16]
    summary.wheel_effort_saturation_count += int(
        torch.any(wheel_effort >= wheel_limit - 1.0e-6).item()
    )
    if torch.any(command.wheel_velocity_target.abs() >= 0.1).item():
        direction_product = command.wheel_velocity_target * wheel_velocity
        summary.wheel_direction_mismatch_count += int(
            torch.any(direction_product < -0.002).item()
        )

    ee_error = float(
        torch.linalg.vector_norm(
            command.target_pose[:3] - teacher_state.ee_pose[:3]
        ).item()
    )
    summary.max_ee_position_error_m = max(
        summary.max_ee_position_error_m, ee_error
    )
    summary.min_singular_value = min(
        summary.min_singular_value,
        float(teacher_state.sigma_min.item()),
    )
    summary.max_abs_roll_rad = max(
        summary.max_abs_roll_rad, abs(teacher_state.roll)
    )
    summary.max_abs_pitch_rad = max(
        summary.max_abs_pitch_rad, abs(teacher_state.pitch)
    )
    summary.joint_limit_violations += adapter.joint_limit_violations()
    summary.base_contacts += base_contact
    summary.reset_count += int(reset_cause != "none")
    safety = command.safety_state.name
    summary.safety_state_counts[safety] = summary.safety_state_counts.get(safety, 0) + 1
    reason = command.safety_reason
    summary.safety_reason_counts[reason] = summary.safety_reason_counts.get(reason, 0) + 1
    summary.track_scale_count += int(command.safety_state <= SafetyState.SCALE)
    summary.hold_or_worse_count += int(command.safety_state >= SafetyState.HOLD)

    arm_target = command.q_des[-7:].detach().cpu()
    if summary._previous_arm_target is not None:
        target_step = float(
            torch.max(torch.abs(arm_target - summary._previous_arm_target)).item()
        )
        summary.max_arm_target_step_rad = max(
            summary.max_arm_target_step_rad, target_step
        )
        summary.arm_snap_count += int(
            target_step > MAX_CONTINUOUS_ARM_TARGET_STEP_RAD
        )
    summary._previous_arm_target = arm_target.clone()


def _format_diagnostics(step: int, state: RollingTeacherState, command) -> str:
    vx, vy = _heading_velocity(state)
    ee_error = float(
        torch.linalg.vector_norm(
            command.target_pose[:3] - state.teacher_state.ee_pose[:3]
        ).item()
    )
    return (
        f"[WBC C1a] step={step} phase={command.phase} "
        f"vx_raw={command.raw_base_velocity_mps:.4f} "
        f"vx_cmd={command.shaped_base_velocity_mps:.4f} "
        f"vx={vx:.4f} vy={vy:.4f} ee_error={ee_error:.5f} "
        f"qp_feasible={command.qp_result.success} "
        f"qp_eq={command.qp_result.max_equality_residual:.3e} "
        f"qp_ineq={command.qp_result.max_inequality_violation:.3e} "
        f"roll={state.teacher_state.roll:.5f} "
        f"pitch={state.teacher_state.pitch:.5f} "
        f"wheel_contacts={state.teacher_state.wheel_contact_count} "
        f"rolling_residual={state.max_rolling_residual_mps:.5f} "
        f"lateral_slip={state.teacher_state.max_lateral_slip:.5f} "
        f"safety={command.safety_state.name} "
        f"reason={command.safety_reason} "
        f"motion_failure={command.motion_failure_reason or 'none'}"
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    summary = C1aSummary(seed=args.seed, requested_steps=args.steps)
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args).app

        import gymnasium as gym
        from isaaclab.utils import math as math_utils
        from isaaclab_tasks.utils import parse_env_cfg

        import go2_pvcnn.tasks  # noqa: F401

        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
        env.reset(seed=args.seed)
        adapter = RollingPhysxTeacherAdapter(
            env,
            math_utils,
            wheel_radius_m=WHEEL_RADIUS_M,
        )
        initial_state, _, _ = adapter.build_rolling_state(0, 0)
        kp, kd = build_teacher_gains()
        amplitude = 0.0 if args.disable_target_motion else 0.005
        orientation_amplitude = (
            0.0 if args.disable_target_motion else 0.01
        )
        trajectory = PlanarBodyFrameTrajectory(
            BandLimitedTrajectoryCfg(
                position_amplitude=amplitude,
                orientation_amplitude=orientation_amplitude,
            )
        )
        teacher = M1PandaRollingWbcTeacher(
            kp=kp,
            kd=kd,
            effort_limit=initial_state.teacher_state.wbc_input.effort_limit,
            safe_arm_target=initial_state.teacher_state.controlled_q[-7:],
            trajectory=trajectory,
        )
        teacher.reset(initial_state, seed=args.seed)

        physics_step = 0
        mission_step = 0
        settled = False
        settling_gate = SettlingGate()
        while simulation_app.is_running() and (
            args.steps == 0 or mission_step < args.steps
        ):
            schedule_step = mission_step if settled else physics_step
            state, base_contact, metrics = adapter.build_rolling_state(
                physics_step, schedule_step
            )
            if not settled:
                heading_speed, _ = _heading_velocity(state)
                try:
                    ready = settling_gate.update(
                        physics_step=physics_step,
                        heading_speed_mps=heading_speed,
                        rolling_residual_mps=(
                            metrics.max_longitudinal_residual_mps
                        ),
                        lateral_slip_mps=metrics.max_lateral_slip_mps,
                        wheel_contact_count=(
                            state.teacher_state.wheel_contact_count
                        ),
                        base_contact=base_contact,
                        signals_finite=state.teacher_state.signals_finite,
                    )
                except RuntimeError:
                    summary.exit_reason = "settling_timeout"
                    break
                if ready:
                    state = replace(state, mission_step=0)
                    teacher.restart_mission(state, seed=args.seed)
                    settled = True
            command = teacher.step(state)
            effort_action = command.effort.to(
                device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(effort_action)
            reset_cause = _termination_cause(env, terminated, truncated)
            physics_step += 1
            if settled:
                _update_summary(
                    summary,
                    state,
                    command,
                    base_contact,
                    adapter,
                    reset_cause,
                )
                mission_step += 1
            display_step = mission_step if settled else physics_step
            if display_step % args.stats_interval == 0:
                print(_format_diagnostics(display_step, state, command), flush=True)
            if reset_cause != "none":
                summary.exit_reason = reset_cause
                break
            if command.terminate:
                summary.exit_reason = "safety_terminate"
                break

        if summary.exit_reason == "not_started":
            if args.steps > 0 and mission_step == args.steps:
                summary.exit_reason = (
                    "steps_complete"
                    if args.steps == MISSION_STEPS
                    else "smoke_complete"
                )
            else:
                summary.exit_reason = "application_closed"
        payload = summary.to_dict()
        print("[WBC C1a summary] " + json.dumps(payload, sort_keys=True), flush=True)
        if args.summary_json is not None:
            atomic_write_summary(args.summary_json, payload)
        if args.steps == MISSION_STEPS:
            return 0 if formal_hard_gates_pass(summary) else 1
        if args.steps > 0:
            return 0 if smoke_gates_pass(summary) else 1
        return 0
    except Exception as error:
        summary.exit_reason = f"error:{type(error).__name__}"
        traceback.print_exc()
        if args.summary_json is not None:
            atomic_write_summary(args.summary_json, summary.to_dict())
        return 1
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
