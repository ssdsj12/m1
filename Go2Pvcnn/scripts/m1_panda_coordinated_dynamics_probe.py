#!/usr/bin/env python3
"""Run long-horizon mount/dynamics gates for the coordinated M1 + Panda."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_ID = "Isaac-M1-Panda-Coordinated-v0"
CONTROL_DT_S = 0.005
MOUNT_POSITION_DRIFT_LIMIT_M = 1.0e-3
MOUNT_ORIENTATION_DRIFT_LIMIT_RAD = 1.0e-3


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hold", "controlled"), required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--stats-interval", type=int, default=200)
    parser.add_argument("--controlled-arm-amplitude-rad", type=float, default=0.05)
    parser.add_argument("--controlled-frequency-hz", type=float, default=0.20)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.steps <= 0 or args.stats_interval <= 0:
        raise ValueError("--num_envs, --steps and --stats-interval must be positive")
    if (
        not math.isfinite(args.controlled_arm_amplitude_rad)
        or not 0.0 <= args.controlled_arm_amplitude_rad <= 0.05
    ):
        raise ValueError("--controlled-arm-amplitude-rad must be finite and in [0,0.05]")
    if (
        not math.isfinite(args.controlled_frequency_hz)
        or not 0.0 < args.controlled_frequency_hz <= 0.5
    ):
        raise ValueError("--controlled-frequency-hz must be finite and in (0,0.5]")


def atomic_write_summary(path: Path, payload: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _exact_id(names: list[str], expected: str) -> int:
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body named {expected!r}, got {matches}")
    return matches[0]


def _relative_mount_pose(robot, math_utils, base_body_id: int, mount_body_id: int):
    base_position = robot.data.body_pos_w[:, base_body_id]
    base_quaternion = robot.data.body_quat_w[:, base_body_id]
    mount_position = robot.data.body_pos_w[:, mount_body_id]
    mount_quaternion = robot.data.body_quat_w[:, mount_body_id]
    relative_position = math_utils.quat_apply_inverse(
        base_quaternion, mount_position - base_position
    )
    relative_quaternion = math_utils.quat_mul(
        math_utils.quat_inv(base_quaternion), mount_quaternion
    )
    return relative_position, relative_quaternion


def _orientation_drift(math_utils, current: torch.Tensor, initial: torch.Tensor):
    delta = math_utils.quat_mul(current, math_utils.quat_inv(initial))
    return torch.linalg.vector_norm(math_utils.axis_angle_from_quat(delta), dim=-1)


def _termination_flags(env, name: str, num_envs: int, device: torch.device):
    try:
        value = env.termination_manager.get_term(name)
    except Exception:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(num_envs)


def _hard_gates(summary: dict[str, object]) -> bool:
    return bool(
        summary["steps_completed"] == summary["requested_steps"]
        and summary["non_finite_count"] == 0
        and summary["reset_count"] == 0
        and summary["base_contact_count"] == 0
        and summary["bad_orientation_count"] == 0
        and summary["joint_limit_violation_count"] == 0
        and summary["max_mount_position_drift_m"] <= MOUNT_POSITION_DRIFT_LIMIT_M
        and summary["max_mount_orientation_drift_rad"]
        <= MOUNT_ORIENTATION_DRIFT_LIMIT_RAD
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    env = None
    summary: dict[str, object] = {
        "task": TASK_ID,
        "mode": args.mode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_steps": args.steps,
        "steps_completed": 0,
        "max_mount_position_drift_m": 0.0,
        "max_mount_orientation_drift_rad": 0.0,
        "max_mount_force_n": 0.0,
        "max_mount_torque_nm": 0.0,
        "non_finite_count": 0,
        "reset_count": 0,
        "base_contact_count": 0,
        "bad_orientation_count": 0,
        "joint_limit_violation_count": 0,
        "max_abs_effort_nm": 0.0,
        "hard_gates_passed": False,
        "exit_reason": "not_started",
    }
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab.utils import math as math_utils
        from isaaclab_tasks.utils import parse_env_cfg

        from go2_pvcnn.assets.m1_panda import (
            M1_PANDA_BASE_BODY_NAME,
            M1_PANDA_MOUNT_BODY_NAME,
        )
        from go2_pvcnn.control.m1_panda_coordination.contracts import WbcJointMap
        from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
            build_teacher_gains,
        )

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        observation, _ = env.reset()
        robot = env.scene["robot"]
        device = torch.device(robot.device)
        joint_map = WbcJointMap.resolve(robot.joint_names)
        controlled_ids = joint_map.controlled.to(device)
        base_body_id = _exact_id(robot.body_names, M1_PANDA_BASE_BODY_NAME)
        mount_body_id = _exact_id(robot.body_names, M1_PANDA_MOUNT_BODY_NAME)
        initial_position, initial_quaternion = _relative_mount_pose(
            robot, math_utils, base_body_id, mount_body_id
        )
        initial_position = initial_position.clone()
        initial_quaternion = initial_quaternion.clone()
        hold_position = robot.data.joint_pos.index_select(1, controlled_ids).clone()
        kp_cpu, kd_cpu = build_teacher_gains()
        kp = kp_cpu.to(device=device, dtype=torch.float32).expand(args.num_envs, -1)
        kd = kd_cpu.to(device=device, dtype=torch.float32).expand(args.num_envs, -1)
        effort_limits = robot.data.joint_effort_limits.index_select(
            1, controlled_ids
        ).abs()
        arm_phase = torch.linspace(
            0.0, math.pi, args.num_envs, device=device, dtype=torch.float32
        )

        for step in range(args.steps):
            q = robot.data.joint_pos.index_select(1, controlled_ids)
            qd = robot.data.joint_vel.index_select(1, controlled_ids)
            target_position = hold_position.clone()
            target_velocity = torch.zeros_like(qd)
            if args.mode == "controlled":
                phase = (
                    2.0
                    * math.pi
                    * args.controlled_frequency_hz
                    * step
                    * CONTROL_DT_S
                    + arm_phase
                )
                displacement = args.controlled_arm_amplitude_rad * torch.sin(phase)
                velocity = (
                    args.controlled_arm_amplitude_rad
                    * 2.0
                    * math.pi
                    * args.controlled_frequency_hz
                    * torch.cos(phase)
                )
                target_position[:, 16] += displacement
                target_position[:, 19] -= 0.5 * displacement
                target_velocity[:, 16] = velocity
                target_velocity[:, 19] = -0.5 * velocity
            effort = kp * (target_position - q) + kd * (target_velocity - qd)
            effort = torch.clamp(effort, min=-effort_limits, max=effort_limits)
            observation, _, terminated, truncated, _ = env.step(effort)

            current_position, current_quaternion = _relative_mount_pose(
                robot, math_utils, base_body_id, mount_body_id
            )
            position_drift = torch.linalg.vector_norm(
                current_position - initial_position, dim=-1
            )
            orientation_drift = _orientation_drift(
                math_utils, current_quaternion, initial_quaternion
            )
            policy_observation = observation["policy"]
            mount_wrench = policy_observation[:, -6:]
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    q,
                    qd,
                    effort,
                    current_position,
                    current_quaternion,
                    mount_wrench,
                )
            )
            summary["non_finite_count"] += int(not finite)
            summary["max_mount_position_drift_m"] = max(
                summary["max_mount_position_drift_m"],
                float(position_drift.max().item()),
            )
            summary["max_mount_orientation_drift_rad"] = max(
                summary["max_mount_orientation_drift_rad"],
                float(orientation_drift.max().item()),
            )
            summary["max_mount_force_n"] = max(
                summary["max_mount_force_n"],
                float(torch.linalg.vector_norm(mount_wrench[:, :3], dim=-1).max().item()),
            )
            summary["max_mount_torque_nm"] = max(
                summary["max_mount_torque_nm"],
                float(torch.linalg.vector_norm(mount_wrench[:, 3:], dim=-1).max().item()),
            )
            summary["max_abs_effort_nm"] = max(
                summary["max_abs_effort_nm"], float(effort.abs().max().item())
            )
            done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
            summary["reset_count"] += int(done.sum().item())
            summary["base_contact_count"] += int(
                _termination_flags(env, "base_contact", args.num_envs, device)
                .sum()
                .item()
            )
            summary["bad_orientation_count"] += int(
                _termination_flags(env, "bad_orientation", args.num_envs, device)
                .sum()
                .item()
            )
            limits = robot.data.soft_joint_pos_limits.index_select(1, controlled_ids)
            violations = (q < limits[:, :, 0]) | (q > limits[:, :, 1])
            summary["joint_limit_violation_count"] += int(violations.sum().item())
            summary["steps_completed"] = step + 1
            if (step + 1) % args.stats_interval == 0:
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "mode": args.mode,
                            "mount_position_drift_m": summary[
                                "max_mount_position_drift_m"
                            ],
                            "mount_orientation_drift_rad": summary[
                                "max_mount_orientation_drift_rad"
                            ],
                            "reset_count": summary["reset_count"],
                            "non_finite_count": summary["non_finite_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not finite:
                summary["exit_reason"] = "non_finite"
                break
        else:
            summary["exit_reason"] = "steps_complete"
        summary["hard_gates_passed"] = _hard_gates(summary)
        atomic_write_summary(args.summary_json, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["hard_gates_passed"] else 2
    except BaseException as error:
        summary["exit_reason"] = f"{type(error).__name__}: {error}"
        atomic_write_summary(args.summary_json, summary)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
