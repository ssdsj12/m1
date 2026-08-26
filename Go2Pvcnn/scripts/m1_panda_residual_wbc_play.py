#!/usr/bin/env python3
"""Run a manual 8D residual probe over the accepted rolling WBC Teacher."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
    _termination_cause,
    atomic_write_summary,
    build_teacher_gains,
)
from m1_panda_wbc_roll_play import RollingPhysxTeacherAdapter
from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import (
    M1PandaRollingWbcTeacher,
    PlanarBodyFrameTrajectory,
)
from go2_pvcnn.control.m1_panda_coordination.trajectory import (
    BandLimitedTrajectoryCfg,
)
from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    MountWrenchFeedbackCfg,
)
from go2_pvcnn.tasks.m1_panda_residual_wbc_wrapper import (
    M1PandaResidualWbcController,
)


TASK_ID = "Isaac-M1-Panda-Residual-Wbc-v0"
WHEEL_RADIUS_M = 0.095


def build_arg_parser(
    *, include_app_launcher_args: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the M1 + Panda eight-channel residual WBC."
    )
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--residual-axis", type=int, default=-1)
    parser.add_argument("--residual-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--stats-interval", type=int, default=64)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cpu")
    return parser


def validate_args(args) -> None:
    if args.task != TASK_ID:
        raise ValueError(f"--task must be {TASK_ID}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.residual_axis < -1 or args.residual_axis >= 8:
        raise ValueError("--residual-axis must be -1 or in [0,7]")
    if not math.isfinite(args.residual_value) or not -1.0 <= args.residual_value <= 1.0:
        raise ValueError("--residual-value must be finite and in [-1,1]")
    if args.residual_axis == -1 and args.residual_value != 0.0:
        raise ValueError("--residual-value requires an explicit residual axis")
    if args.num_envs != 1:
        raise ValueError("the reference residual WBC supports exactly one environment")
    if args.stats_interval <= 0:
        raise ValueError("--stats-interval must be positive")


def build_normalized_residual(
    args, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    action = torch.zeros(1, 8, device=torch.device(device), dtype=dtype)
    if args.residual_axis >= 0:
        action[0, args.residual_axis] = float(args.residual_value)
    return action


@dataclass
class ResidualSmokeSummary:
    seed: int
    requested_steps: int
    steps: int = 0
    finite: bool = True
    qp_feasible_count: int = 0
    min_wheel_contact_count: int = 4
    base_contacts: int = 0
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    joint_limit_violations: int = 0
    reset_count: int = 0
    max_ee_position_error_m: float = 0.0
    max_abs_normalized_residual: float = 0.0
    max_abs_physical_residual: float = 0.0
    max_abs_filtered_mount_wrench: float = 0.0
    max_abs_correction_wrench: float = 0.0
    safety_state_counts: dict[str, int] = field(default_factory=dict)
    exit_reason: str = "not_started"

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "requested_steps": self.requested_steps,
            "steps": self.steps,
            "finite": self.finite,
            "qp_feasible_count": self.qp_feasible_count,
            "qp_feasible_rate": self.qp_feasible_count / max(self.steps, 1),
            "min_wheel_contact_count": self.min_wheel_contact_count,
            "base_contacts": self.base_contacts,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "joint_limit_violations": self.joint_limit_violations,
            "reset_count": self.reset_count,
            "max_ee_position_error_m": self.max_ee_position_error_m,
            "max_abs_normalized_residual": self.max_abs_normalized_residual,
            "max_abs_physical_residual": self.max_abs_physical_residual,
            "max_abs_filtered_mount_wrench": self.max_abs_filtered_mount_wrench,
            "max_abs_correction_wrench": self.max_abs_correction_wrench,
            "safety_state_counts": dict(sorted(self.safety_state_counts.items())),
            "exit_reason": self.exit_reason,
        }


def _update_summary(
    summary: ResidualSmokeSummary,
    state,
    command,
    residual_step,
    normalized_residual: torch.Tensor,
    base_contact: int,
    adapter,
    reset_cause: str,
) -> None:
    summary.steps += 1
    summary.qp_feasible_count += int(command.qp_result.success)
    summary.min_wheel_contact_count = min(
        summary.min_wheel_contact_count,
        state.teacher_state.wheel_contact_count,
    )
    summary.base_contacts += int(base_contact)
    summary.max_abs_roll_rad = max(
        summary.max_abs_roll_rad, abs(float(state.teacher_state.roll))
    )
    summary.max_abs_pitch_rad = max(
        summary.max_abs_pitch_rad, abs(float(state.teacher_state.pitch))
    )
    summary.joint_limit_violations += adapter.joint_limit_violations()
    summary.reset_count += int(reset_cause != "none")
    summary.max_ee_position_error_m = max(
        summary.max_ee_position_error_m,
        float(
            torch.linalg.vector_norm(
                command.target_pose[:3] - state.teacher_state.ee_pose[:3]
            ).item()
        ),
    )
    summary.max_abs_normalized_residual = max(
        summary.max_abs_normalized_residual,
        float(normalized_residual.abs().max().item()),
    )
    summary.max_abs_physical_residual = max(
        summary.max_abs_physical_residual,
        float(residual_step.applied_residual.physical.abs().max().item()),
    )
    summary.max_abs_filtered_mount_wrench = max(
        summary.max_abs_filtered_mount_wrench,
        float(residual_step.filtered_mount_wrench_b.abs().max().item()),
    )
    summary.max_abs_correction_wrench = max(
        summary.max_abs_correction_wrench,
        float(residual_step.correction_wrench_b.abs().max().item()),
    )
    name = command.safety_state.name
    summary.safety_state_counts[name] = summary.safety_state_counts.get(name, 0) + 1
    tensors = (
        command.effort,
        residual_step.applied_residual.physical,
        residual_step.filtered_mount_wrench_b,
        residual_step.correction_wrench_b,
    )
    summary.finite = summary.finite and all(
        bool(torch.isfinite(value).all().item()) for value in tensors
    )


def smoke_gates_pass(summary: ResidualSmokeSummary) -> bool:
    return bool(
        summary.steps == summary.requested_steps
        and summary.finite
        and summary.qp_feasible_count == summary.steps
        and summary.min_wheel_contact_count == 4
        and summary.base_contacts == 0
        and summary.max_abs_roll_rad <= math.radians(10.0)
        and summary.max_abs_pitch_rad <= math.radians(10.0)
        and summary.joint_limit_violations == 0
        and summary.reset_count == 0
        and summary.exit_reason == "steps_complete"
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    summary = ResidualSmokeSummary(seed=args.seed, requested_steps=args.steps)
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
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset(seed=args.seed)
        adapter = RollingPhysxTeacherAdapter(
            env, math_utils, wheel_radius_m=WHEEL_RADIUS_M
        )
        initial_state, _, _ = adapter.build_rolling_state(0, 0)
        kp, kd = build_teacher_gains()
        teacher = M1PandaRollingWbcTeacher(
            kp=kp,
            kd=kd,
            effort_limit=initial_state.teacher_state.wbc_input.effort_limit,
            safe_arm_target=initial_state.teacher_state.controlled_q[-7:],
            trajectory=PlanarBodyFrameTrajectory(
                BandLimitedTrajectoryCfg(
                    position_amplitude=0.005,
                    orientation_amplitude=0.01,
                )
            ),
        )
        teacher.reset(initial_state, seed=args.seed)
        controller = M1PandaResidualWbcController(
            [teacher],
            device="cpu",
            dtype=torch.float64,
            base_seed=args.seed,
            feedback_cfg=MountWrenchFeedbackCfg(
                bias_warmup_samples=args.warmup_steps
            ),
        )
        requested = build_normalized_residual(args, "cpu", torch.float64)
        zero = torch.zeros_like(requested)
        total_steps = args.warmup_steps + args.steps
        for physics_step in range(total_steps):
            if not simulation_app.is_running():
                summary.exit_reason = "application_closed"
                break
            state, base_contact, _ = adapter.build_rolling_state(
                physics_step, physics_step
            )
            normalized = zero if physics_step < args.warmup_steps else requested
            residual_step = controller.step(
                states=(state,),
                normalized_residual=normalized,
                measured_mount_wrench_b=adapter.read_mount_wrench_b().unsqueeze(0),
                leg_soft_limits=adapter.leg_soft_limits().unsqueeze(0),
            )
            command = residual_step.teacher_commands[0]
            effort_action = command.effort.to(
                device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(effort_action)
            reset_cause = _termination_cause(env, terminated, truncated)
            if physics_step >= args.warmup_steps:
                _update_summary(
                    summary,
                    state,
                    command,
                    residual_step,
                    normalized,
                    base_contact,
                    adapter,
                    reset_cause,
                )
            if (physics_step + 1) % args.stats_interval == 0:
                print(
                    f"[Residual WBC] physics_step={physics_step + 1} "
                    f"mission_steps={summary.steps} qp={command.qp_result.success} "
                    f"safety={command.safety_state.name} "
                    f"residual_max={float(residual_step.applied_residual.physical.abs().max().item()):.4f}",
                    flush=True,
                )
            if reset_cause != "none":
                summary.exit_reason = reset_cause
                break
            if command.terminate:
                summary.exit_reason = "safety_terminate"
                break
        if summary.exit_reason == "not_started":
            summary.exit_reason = (
                "steps_complete"
                if summary.steps == args.steps
                else "application_closed"
            )
        payload = summary.to_dict()
        print("[Residual WBC summary] " + json.dumps(payload, sort_keys=True), flush=True)
        if args.summary_json is not None:
            atomic_write_summary(args.summary_json, payload)
        return 0 if smoke_gates_pass(summary) else 1
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
