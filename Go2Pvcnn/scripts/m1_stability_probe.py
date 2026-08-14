#!/usr/bin/env python3
"""Probe M1 stability under fixed random actions or fixed wheel actions."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
for path in (GO2PVCNN_ROOT, GO2PVCNN_ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Measure M1 stability under zero/random leg actions.")
    parser.add_argument("--task", default="Isaac-M1-Walk-v0")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stds", default="0,0.02,0.05,0.10,0.15,0.30")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--wheel-actions",
        default=None,
        help="Optional comma-separated raw wheel actions. Use one value for all wheels or four values.",
    )
    parser.add_argument(
        "--leg-actions",
        default=None,
        help="Optional comma-separated 12 raw leg actions in FAR/FBL/RAR/RBL order.",
    )
    parser.add_argument("--stand-height", type=float, default=None)
    parser.add_argument("--front-hip", type=float, default=None)
    parser.add_argument("--front-knee", type=float, default=None)
    parser.add_argument("--rear-hip", type=float, default=None)
    parser.add_argument("--rear-knee", type=float, default=None)
    parser.add_argument(
        "--wheel-effort-limit",
        type=float,
        default=None,
        help="Override the wheel actuator effort limit for controlled physics probes.",
    )
    parser.add_argument(
        "--wheel-damping",
        type=float,
        default=None,
        help="Override the wheel velocity-loop damping gain for controlled physics probes.",
    )
    parser.add_argument("--explicit-wheel-actuator", action="store_true")
    parser.add_argument("--episode-length", type=float, default=None)
    parser.add_argument("--wheel-pulse-actions", default=None)
    parser.add_argument("--wheel-pulse-period-steps", type=int, default=50)
    parser.add_argument("--wheel-pulse-steps", type=int, default=8)
    parser.add_argument("--leg-stiffness", type=float, default=None)
    parser.add_argument("--leg-damping", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_stds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_wheel_actions(value: str | None) -> list[float] | None:
    if value is None:
        return None
    actions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(actions) == 1:
        return actions * 4
    if len(actions) != 4:
        raise ValueError("--wheel-actions must contain one value or four comma-separated values")
    return actions


def _parse_leg_actions(value: str | None) -> list[float] | None:
    if value is None:
        return None
    actions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(actions) != 12:
        raise ValueError("--leg-actions must contain 12 comma-separated values")
    return actions


def _termination_terms(env) -> dict[str, torch.Tensor]:
    manager = getattr(env.unwrapped, "termination_manager", None)
    if manager is None:
        return {}
    terms: dict[str, torch.Tensor] = {}
    for name in ("bad_orientation", "base_contact", "time_out"):
        try:
            term = manager.get_term(name)
        except Exception:
            term = None
        if term is not None:
            terms[name] = torch.as_tensor(term, dtype=torch.bool, device=env.unwrapped.device).reshape(-1)
    if "time_out" not in terms and getattr(manager, "time_outs", None) is not None:
        terms["time_out"] = torch.as_tensor(manager.time_outs, dtype=torch.bool, device=env.unwrapped.device).reshape(-1)
    return terms


def _run_one(
    env,
    *,
    action_std: float,
    steps: int,
    seed: int,
    clip: float,
    leg_actions: list[float] | None,
    wheel_actions: list[float] | None,
    wheel_pulse_actions: list[float] | None,
    wheel_pulse_period_steps: int,
    wheel_pulse_steps: int,
) -> dict[str, float | int]:
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    if action_dim != 16:
        raise RuntimeError(f"Expected 16 M1 actions, got {action_dim}")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    env.reset()

    reset_seen = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    first_reset_step = torch.full((env.unwrapped.num_envs,), steps, dtype=torch.long, device=device)
    reason_counts = {"bad_orientation": 0, "base_contact": 0, "time_out": 0, "unknown": 0}
    root_start = robot.data.root_pos_w.clone()
    max_root_x = root_start[:, 0].clone()
    root_last_before_step = root_start.clone()
    root_at_first_done = root_start.clone()
    wheel_body_ids, _ = robot.find_bodies(".*_FOOT_LINK", preserve_order=True)
    max_wheel_height = robot.data.body_pos_w[:, wheel_body_ids, 2].clone()
    min_height = float(robot.data.root_pos_w[:, 2].min().item())
    max_tilt = 0.0
    fixed_leg_actions = None
    if leg_actions is not None:
        fixed_leg_actions = torch.tensor(
            leg_actions, dtype=torch.float32, device=device
        ).reshape(1, 12)
    fixed_wheel_actions = None
    if wheel_actions is not None:
        fixed_wheel_actions = torch.tensor(wheel_actions, dtype=torch.float32, device=device).reshape(1, 4)
    pulse_wheel_actions = None
    if wheel_pulse_actions is not None:
        pulse_wheel_actions = torch.tensor(wheel_pulse_actions, dtype=torch.float32, device=device).reshape(1, 4)

    for step in range(steps):
        if action_std == 0.0:
            actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=device)
        else:
            actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=device)
            actions[:, :12] = torch.randn(
                (env.unwrapped.num_envs, 12),
                generator=generator,
                device=device,
            ) * action_std
            actions = torch.clamp(actions, -clip, clip)
        if fixed_wheel_actions is not None:
            actions[:, 12:16] = fixed_wheel_actions
        if fixed_leg_actions is not None:
            actions[:, :12] = fixed_leg_actions
        if pulse_wheel_actions is not None and step % wheel_pulse_period_steps < wheel_pulse_steps:
            actions[:, 12:16] = pulse_wheel_actions

        root_last_before_step = robot.data.root_pos_w.clone()
        _, _, terminated, truncated, _ = env.step(actions)
        done = (torch.as_tensor(terminated, dtype=torch.bool, device=device).reshape(-1)
                | torch.as_tensor(truncated, dtype=torch.bool, device=device).reshape(-1))
        terms = _termination_terms(env)

        new_done = done & torch.logical_not(reset_seen)
        for env_id in torch.nonzero(new_done, as_tuple=False).flatten().tolist():
            env_int = int(env_id)
            reason = "unknown"
            for name in ("bad_orientation", "base_contact", "time_out"):
                term = terms.get(name)
                if term is not None and bool(term[env_int].item()):
                    reason = name
                    break
            reason_counts[reason] = int(reason_counts[reason]) + 1
            first_reset_step[env_int] = step
            root_at_first_done[env_int] = root_last_before_step[env_int]
        reset_seen |= done

        heights = robot.data.root_pos_w[:, 2]
        max_wheel_height = torch.maximum(
            max_wheel_height, robot.data.body_pos_w[:, wheel_body_ids, 2]
        )
        max_root_x = torch.maximum(max_root_x, robot.data.root_pos_w[:, 0])
        min_height = min(min_height, float(heights.min().item()))
        projected_g = robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(torch.abs(projected_g[:, 2]), -1.0, 1.0))
        max_tilt = max(max_tilt, float(tilt.max().item()))

    reset_count = int(reset_seen.sum().item())
    never_reset = int(env.unwrapped.num_envs - reset_count)
    first_steps = first_reset_step[reset_seen]
    mean_first_reset_step = float(first_steps.float().mean().item()) if int(first_steps.numel()) else float(steps)
    root_delta = robot.data.root_pos_w - root_start
    root_delta_pre_reset = torch.where(reset_seen.unsqueeze(-1), root_at_first_done - root_start, root_delta)
    wheel_ids, _ = robot.find_joints(".*_FOOT_JOINT")
    wheel_pos = robot.data.joint_pos[:, wheel_ids]
    wheel_vel = robot.data.joint_vel[:, wheel_ids]
    wheel_torque = robot.data.applied_torque[:, wheel_ids]
    wheel_target = robot.data.joint_vel_target[:, wheel_ids]
    return {
        "std": action_std,
        "reset_count": reset_count,
        "never_reset": never_reset,
        "mean_first_reset_step": mean_first_reset_step,
        "bad_orientation": int(reason_counts["bad_orientation"]),
        "base_contact": int(reason_counts["base_contact"]),
        "time_out": int(reason_counts["time_out"]),
        "unknown": int(reason_counts["unknown"]),
        "min_height": min_height,
        "max_tilt_rad": max_tilt,
        "mean_dx": float(root_delta[:, 0].mean().item()),
        "mean_dy": float(root_delta[:, 1].mean().item()),
        "mean_dz": float(root_delta[:, 2].mean().item()),
        "mean_dx_pre_reset": float(root_delta_pre_reset[:, 0].mean().item()),
        "mean_dy_pre_reset": float(root_delta_pre_reset[:, 1].mean().item()),
        "mean_dz_pre_reset": float(root_delta_pre_reset[:, 2].mean().item()),
        "mean_max_dx": float((max_root_x - root_start[:, 0]).mean().item()),
        "mean_wheel_pos": float(wheel_pos.mean().item()),
        "mean_wheel_vel": float(wheel_vel.mean().item()),
        "wheel_pos_by_index": wheel_pos.mean(dim=0).detach().cpu().tolist(),
        "wheel_vel_by_index": wheel_vel.mean(dim=0).detach().cpu().tolist(),
        "wheel_torque_by_index": wheel_torque.mean(dim=0).detach().cpu().tolist(),
        "wheel_target_by_index": wheel_target.mean(dim=0).detach().cpu().tolist(),
        "wheel_height_by_index": robot.data.body_pos_w[:, wheel_body_ids, 2]
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "max_wheel_height_by_index": max_wheel_height.max(dim=0).values
        .detach()
        .cpu()
        .tolist(),
    }


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        import gymnasium as gym

        import go2_pvcnn.tasks  # noqa: F401

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        if args.episode_length is not None:
            env_cfg.episode_length_s = args.episode_length
        if args.leg_stiffness is not None:
            env_cfg.scene.robot.actuators["legs"].stiffness = args.leg_stiffness
        if args.leg_damping is not None:
            env_cfg.scene.robot.actuators["legs"].damping = args.leg_damping
        if args.explicit_wheel_actuator:
            from isaaclab.actuators import DCMotorCfg

            explicit_effort = args.wheel_effort_limit if args.wheel_effort_limit is not None else 40.0
            explicit_damping = args.wheel_damping if args.wheel_damping is not None else 2.0
            env_cfg.scene.robot.actuators["wheels"] = DCMotorCfg(
                joint_names_expr=[".*_FOOT_JOINT"],
                effort_limit=explicit_effort,
                saturation_effort=explicit_effort,
                velocity_limit=20.0,
                stiffness=0.0,
                damping=explicit_damping,
                friction=0.0,
            )
        elif args.wheel_effort_limit is not None:
            env_cfg.scene.robot.actuators["wheels"].effort_limit_sim = args.wheel_effort_limit
        if args.wheel_damping is not None and not args.explicit_wheel_actuator:
            env_cfg.scene.robot.actuators["wheels"].damping = args.wheel_damping
        wheel_actions = _parse_wheel_actions(args.wheel_actions)
        leg_actions = _parse_leg_actions(args.leg_actions)
        wheel_pulse_actions = _parse_wheel_actions(args.wheel_pulse_actions)
        if args.stand_height is not None:
            env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.stand_height)
        if args.front_hip is not None:
            env_cfg.scene.robot.init_state.joint_pos["FAR_HIP_JOINT"] = args.front_hip
            env_cfg.scene.robot.init_state.joint_pos["FBL_HIP_JOINT"] = args.front_hip
        if args.front_knee is not None:
            env_cfg.scene.robot.init_state.joint_pos["FAR_KNEE_JOINT"] = args.front_knee
            env_cfg.scene.robot.init_state.joint_pos["FBL_KNEE_JOINT"] = args.front_knee
        if args.rear_hip is not None:
            env_cfg.scene.robot.init_state.joint_pos["RAR_HIP_JOINT"] = args.rear_hip
            env_cfg.scene.robot.init_state.joint_pos["RBL_HIP_JOINT"] = args.rear_hip
        if args.rear_knee is not None:
            env_cfg.scene.robot.init_state.joint_pos["RAR_KNEE_JOINT"] = args.rear_knee
            env_cfg.scene.robot.init_state.joint_pos["RBL_KNEE_JOINT"] = args.rear_knee

        env = gym.make(args.task, cfg=env_cfg)
        print(
            "std reset_count never_reset mean_first_reset_step bad_orientation "
            "base_contact time_out unknown min_height max_tilt_rad mean_dx mean_dy mean_dz "
            "mean_dx_pre_reset mean_dy_pre_reset mean_dz_pre_reset",
            "mean_max_dx mean_wheel_pos mean_wheel_vel",
            flush=True,
        )
        for idx, std in enumerate(_parse_stds(args.stds)):
            stats = _run_one(
                env,
                action_std=std,
                steps=args.steps,
                seed=args.seed + idx,
                clip=args.clip,
                leg_actions=leg_actions,
                wheel_actions=wheel_actions,
                wheel_pulse_actions=wheel_pulse_actions,
                wheel_pulse_period_steps=args.wheel_pulse_period_steps,
                wheel_pulse_steps=args.wheel_pulse_steps,
            )
            print(
                f"{stats['std']:.4f} {stats['reset_count']} {stats['never_reset']} "
                f"{stats['mean_first_reset_step']:.1f} {stats['bad_orientation']} "
                f"{stats['base_contact']} {stats['time_out']} {stats['unknown']} "
                f"{stats['min_height']:.4f} {stats['max_tilt_rad']:.4f} "
                f"{stats['mean_dx']:.4f} {stats['mean_dy']:.4f} {stats['mean_dz']:.4f} "
                f"{stats['mean_dx_pre_reset']:.4f} {stats['mean_dy_pre_reset']:.4f} "
                f"{stats['mean_dz_pre_reset']:.4f} {stats['mean_max_dx']:.4f} "
                f"{stats['mean_wheel_pos']:.4f} {stats['mean_wheel_vel']:.4f} "
                f"wheel_pos_by_index={stats['wheel_pos_by_index']} "
                f"wheel_vel_by_index={stats['wheel_vel_by_index']} "
                f"wheel_torque_by_index={stats['wheel_torque_by_index']} "
                f"wheel_target_by_index={stats['wheel_target_by_index']} "
                f"wheel_height_by_index={stats['wheel_height_by_index']} "
                f"max_wheel_height_by_index={stats['max_wheel_height_by_index']}",
                flush=True,
            )
        env.close()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
