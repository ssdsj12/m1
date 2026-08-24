#!/usr/bin/env python3
"""Verify coordinated reset DR and Panda-hand wrench physics on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TASK_ID = "Isaac-M1-Panda-Coordinated-v0"


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


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


def _snapshot(robot) -> dict[str, torch.Tensor]:
    return {
        "root": robot.data.root_state_w.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
    }


def _snapshots_equal(left, right) -> bool:
    return all(torch.equal(left[key], right[key]) for key in left)


def _reset_bounds(robot, env, math_utils, controlled_ids, leg_ids, wheel_ids, arm_ids):
    current_root = robot.data.root_state_w
    default_root = robot.data.default_root_state
    expected_position = default_root[:, :3] + env.scene.env_origins
    position_offset = current_root[:, :3] - expected_position
    relative_quat = math_utils.quat_mul(
        current_root[:, 3:7], math_utils.quat_inv(default_root[:, 3:7])
    )
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(relative_quat)
    angles = torch.stack((roll, pitch, yaw), dim=-1)
    angles = torch.atan2(torch.sin(angles), torch.cos(angles))
    joint_offset = robot.data.joint_pos - robot.data.default_joint_pos
    velocity_offset = robot.data.joint_vel - robot.data.default_joint_vel
    position_limits = robot.data.soft_joint_pos_limits
    velocity_limits = robot.data.soft_joint_vel_limits
    controlled_velocity_limits = velocity_limits[:, controlled_ids]
    usable_velocity_limit = torch.isfinite(controlled_velocity_limits) & (
        controlled_velocity_limits > 0.0
    )
    material = robot.root_physx_view.get_material_properties()
    static_friction = material[..., 0]
    dynamic_friction = material[..., 1]
    restitution = material[..., 2]
    gates = {
        "root_xy": bool((position_offset[:, :2].abs() <= 0.020001).all()),
        "root_z": bool((position_offset[:, 2].abs() <= 1.0e-6).all()),
        "root_roll_pitch": bool((angles[:, :2].abs() <= 0.030001).all()),
        "root_yaw": bool((angles[:, 2].abs() <= 0.050001).all()),
        "root_linear_velocity": bool(
            ((current_root[:, 7:10] - default_root[:, 7:10]).abs() <= 0.050001).all()
        ),
        "root_angular_velocity": bool(
            ((current_root[:, 10:13] - default_root[:, 10:13]).abs() <= 0.100001).all()
        ),
        "leg_position": bool((joint_offset[:, leg_ids].abs() <= 0.020001).all()),
        "arm_position": bool((joint_offset[:, arm_ids].abs() <= 0.030001).all()),
        "wheel_position_default": bool(
            (joint_offset[:, wheel_ids].abs() <= 1.0e-7).all()
        ),
        "position_soft_limits": bool(
            (
                (robot.data.joint_pos >= position_limits[..., 0])
                & (robot.data.joint_pos <= position_limits[..., 1])
            ).all()
        ),
        "controlled_velocity": bool(
            (velocity_offset[:, controlled_ids].abs() <= 0.050001).all()
        ),
        "velocity_soft_limits": bool(
            (
                ~usable_velocity_limit
                | (
                    robot.data.joint_vel[:, controlled_ids].abs()
                    <= controlled_velocity_limits + 1.0e-7
                )
            ).all()
        ),
        "static_friction": bool(
            ((static_friction >= 0.8) & (static_friction <= 1.2)).all()
        ),
        "dynamic_friction": bool(
            ((dynamic_friction >= 0.8) & (dynamic_friction <= 1.2)).all()
        ),
        "restitution": bool((restitution.abs() <= 1.0e-7).all()),
    }
    metrics = {
        "max_root_xy_offset_m": float(position_offset[:, :2].abs().max().item()),
        "max_root_orientation_offset_rad": float(angles.abs().max().item()),
        "max_leg_position_offset_rad": float(joint_offset[:, leg_ids].abs().max().item()),
        "max_arm_position_offset_rad": float(joint_offset[:, arm_ids].abs().max().item()),
        "max_controlled_velocity_offset_rad_s": float(
            velocity_offset[:, controlled_ids].abs().max().item()
        ),
        "controlled_velocity_limit_min_rad_s": float(
            velocity_limits[:, controlled_ids].min().item()
        ),
        "controlled_velocity_limit_max_rad_s": float(
            velocity_limits[:, controlled_ids].max().item()
        ),
        "static_friction_min": float(static_friction.min().item()),
        "static_friction_max": float(static_friction.max().item()),
    }
    return gates, metrics


def _termination_count(env, name: str) -> int:
    try:
        value = env.termination_manager.get_term(name)
    except Exception:
        return 0
    return int(torch.as_tensor(value, device=env.device).count_nonzero().item())


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs < 2 or args.steps <= 0:
        raise ValueError("num_envs must be at least two and steps must be positive")

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    env = None
    summary: dict[str, object] = {
        "task": TASK_ID,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_steps": args.steps,
        "steps_completed": 0,
        "same_seed_reset_match": False,
        "different_env_reset_diversity": False,
        "selected_reset_isolation": False,
        "selected_reset_changed": False,
        "reset_bounds_passed": False,
        "applied_hand_wrench_nonzero": False,
        "mount_wrench_response_nonzero": False,
        "non_finite_count": 0,
        "reset_count": 0,
        "base_contact_count": 0,
        "bad_orientation_count": 0,
        "hard_gates_passed": False,
        "reset_event_diagnostics_present": False,
        "reset_event_diagnostics_getter_present": False,
        "controlled_velocity_randomized": False,
    }
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import math as math_utils
        from isaaclab_tasks.utils import parse_env_cfg

        import go2_pvcnn.mdp as mdp
        from go2_pvcnn.assets import M1_LEG_JOINT_NAMES, M1_WHEEL_JOINT_NAMES
        from go2_pvcnn.assets.m1_panda import M1_PANDA_WBC_CONTROLLED_JOINT_NAMES
        from go2_pvcnn.tasks.m1_panda_coordinated_env_cfg import (
            configure_coordinated_training_domain_randomization,
        )
        from go2_pvcnn.tasks.m1_panda_coordinated_wrapper import (
            M1PandaCoordinatedEnvWrapper,
        )

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        configure_coordinated_training_domain_randomization(cfg, True)
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        robot = env.scene["robot"]
        device = torch.device(robot.device)
        all_ids = torch.arange(args.num_envs, device=device)

        torch.manual_seed(args.seed)
        env._reset_idx(all_ids)
        first_batch = _snapshot(robot)
        torch.manual_seed(args.seed)
        env._reset_idx(all_ids)
        second_batch = _snapshot(robot)
        summary["same_seed_reset_match"] = _snapshots_equal(
            first_batch, second_batch
        )
        diversity_vector = torch.cat(
            (second_batch["root"], second_batch["joint_pos"], second_batch["joint_vel"]),
            dim=1,
        )
        summary["different_env_reset_diversity"] = bool(
            (diversity_vector[1:] != diversity_vector[:1]).any()
        )

        selected_ids = torch.tensor([0], device=device)
        before_selected_reset = _snapshot(robot)
        torch.manual_seed(args.seed + 1)
        env._reset_idx(selected_ids)
        after_selected_reset = _snapshot(robot)
        reset_diagnostics = mdp.get_coordinated_joint_reset_diagnostics(robot)
        summary["reset_event_diagnostics_getter_present"] = reset_diagnostics is not None
        summary["reset_event_diagnostics_present"] = all(
            isinstance(
                getattr(
                    env,
                    f"m1_panda_coordinated_joint_reset_{kind}_deviation",
                    None,
                ),
                torch.Tensor,
            )
            for kind in ("position", "velocity")
        )
        summary["selected_reset_isolation"] = all(
            torch.equal(before_selected_reset[key][1:], after_selected_reset[key][1:])
            for key in before_selected_reset
        )
        summary["selected_reset_changed"] = any(
            not torch.equal(before_selected_reset[key][0], after_selected_reset[key][0])
            for key in before_selected_reset
        )

        leg_ids = torch.tensor(
            robot.find_joints(list(M1_LEG_JOINT_NAMES), preserve_order=True)[0],
            device=device,
        )
        wheel_ids = torch.tensor(
            robot.find_joints(list(M1_WHEEL_JOINT_NAMES), preserve_order=True)[0],
            device=device,
        )
        arm_names = [
            name
            for name in M1_PANDA_WBC_CONTROLLED_JOINT_NAMES
            if name.startswith("panda_joint")
        ]
        arm_ids = torch.tensor(
            robot.find_joints(arm_names, preserve_order=True)[0], device=device
        )
        controlled_ids = torch.cat((leg_ids, wheel_ids, arm_ids))
        reset_gates, reset_metrics = _reset_bounds(
            robot,
            env,
            math_utils,
            controlled_ids,
            leg_ids,
            wheel_ids,
            arm_ids,
        )
        summary["reset_bound_gates"] = reset_gates
        summary.update(reset_metrics)
        summary["reset_bounds_passed"] = all(reset_gates.values())
        summary["controlled_velocity_randomized"] = bool(
            reset_metrics["max_controlled_velocity_offset_rad_s"] > 0.0
        )

        wrapper = M1PandaCoordinatedEnvWrapper(
            env, training_randomization=True, seed=args.seed
        )
        baseline_mount_wrench_b = mdp.m1_panda_mount_wrench_b(
            env,
            SceneEntityCfg("robot"),
            mount_body_name="panda_link0",
            base_body_name="BASE_LINK",
        ).clone()
        max_mount_response = 0.0
        max_force_norm = 0.0
        max_torque_norm = 0.0
        for step in range(args.steps):
            observations, _, dones, extras = wrapper.step(
                torch.zeros(args.num_envs, 23, device=device)
            )
            wrench = wrapper.current_wrench_b
            mount_wrench_b = mdp.m1_panda_mount_wrench_b(
                env,
                SceneEntityCfg("robot"),
                mount_body_name="panda_link0",
                base_body_name="BASE_LINK",
            )
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    observations,
                    wrench,
                    mount_wrench_b,
                    robot.data.root_state_w,
                    robot.data.joint_pos,
                    robot.data.joint_vel,
                )
            )
            summary["non_finite_count"] += int(not finite)
            summary["reset_count"] += int(dones.sum().item())
            summary["base_contact_count"] += _termination_count(env, "base_contact")
            summary["bad_orientation_count"] += _termination_count(
                env, "bad_orientation"
            )
            max_force_norm = max(
                max_force_norm,
                float(torch.linalg.vector_norm(wrench[:, :3], dim=-1).max().item()),
            )
            max_torque_norm = max(
                max_torque_norm,
                float(torch.linalg.vector_norm(wrench[:, 3:], dim=-1).max().item()),
            )
            max_mount_response = max(
                max_mount_response,
                float(
                    torch.linalg.vector_norm(
                        mount_wrench_b - baseline_mount_wrench_b, dim=-1
                    ).max().item()
                ),
            )
            summary["steps_completed"] = step + 1
            if not finite:
                break

        summary["max_applied_force_norm_n"] = max_force_norm
        summary["max_applied_torque_norm_nm"] = max_torque_norm
        summary["max_mount_wrench_response"] = max_mount_response
        summary["applied_hand_wrench_nonzero"] = bool(
            max_force_norm > 0.0 or max_torque_norm > 0.0
        )
        summary["mount_wrench_response_nonzero"] = max_mount_response > 1.0e-6
        summary["wrench_bounds_passed"] = bool(
            max_force_norm <= (3.0**0.5) * 20.0 + 1.0e-5
            and max_torque_norm <= (3.0**0.5) * 5.0 + 1.0e-5
        )
        summary["target_body"] = "panda_hand"
        summary["mount_signal"] = "mount_wrench_b"
        summary["hard_gates_passed"] = bool(
            summary["steps_completed"] == args.steps
            and summary["same_seed_reset_match"]
            and summary["different_env_reset_diversity"]
            and summary["selected_reset_isolation"]
            and summary["selected_reset_changed"]
            and summary["reset_bounds_passed"]
            and summary["controlled_velocity_randomized"]
            and summary["wrench_bounds_passed"]
            and summary["applied_hand_wrench_nonzero"]
            and summary["mount_wrench_response_nonzero"]
            and summary["non_finite_count"] == 0
            and summary["reset_count"] == 0
            and summary["base_contact_count"] == 0
            and summary["bad_orientation_count"] == 0
        )
        atomic_write_summary(args.output, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["hard_gates_passed"] else 2
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_write_summary(args.output, summary)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
