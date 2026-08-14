#!/usr/bin/env python3
"""Play M1 smoke with either open-loop rolling/wave or a trained checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
PVCNN_ROOT = GO2PVCNN_ROOT.parent / "pvcnn"
for path in (GO2PVCNN_ROOT, RSL_RL_ROOT, PVCNN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

configure_pvcnn_cuda(GO2PVCNN_ROOT.parent)

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play M1 smoke rolling/wave or checkpoint policy.")
    parser.add_argument("--task", default="Isaac-M1-Walk-v0")
    parser.add_argument("--mode", default="rolling", choices=["rolling", "wave"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--perception-checkpoint", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rolling-wheel-velocity", type=float, default=0.5)
    parser.add_argument("--wave-wheel-velocity", type=float, default=1.5)
    parser.add_argument("--wave-amplitude", type=float, default=0.0)
    parser.add_argument("--wave-frequency", type=float, default=1.0)
    parser.add_argument("--wheel-signs", default="1,1,1,1")
    parser.add_argument("--clip-actions", dest="clip_actions", type=float, default=0.02)
    parser.add_argument("--lock-legs", action="store_true")
    parser.add_argument("--disable-crossing-reset", action="store_true")
    parser.add_argument("--enable-wave-reference-actions", action="store_true")
    parser.add_argument("--wave-leg-action-limit", type=float, default=None)
    parser.add_argument("--wave-reference-amplitude", type=float, default=None)
    parser.add_argument("--wave-reference-knee-ratio", type=float, default=None)
    parser.add_argument("--rear-amplitude-scale", type=float, default=None)
    parser.add_argument("--front-support-ratio", type=float, default=None)
    parser.add_argument("--rear-support-ratio", type=float, default=None)
    parser.add_argument("--obstacle-wheel-action", type=float, default=None)
    parser.add_argument("--stand-height", type=float, default=None)
    parser.add_argument("--stand-hip", type=float, default=None)
    parser.add_argument("--stand-knee", type=float, default=None)
    parser.add_argument("--front-hip", type=float, default=None)
    parser.add_argument("--front-knee", type=float, default=None)
    parser.add_argument("--rear-hip", type=float, default=None)
    parser.add_argument("--rear-knee", type=float, default=None)
    parser.add_argument("--debug-root-delta", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()
    if args.perception_checkpoint and not args.checkpoint:
        raise ValueError("--perception-checkpoint requires --checkpoint")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        import gymnasium as gym

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_pvcnn_perception import M1PvcnnRslRlEnvWrapper
        from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper
        from go2_pvcnn.tasks.m1_smoke_controller import M1SmokeControllerCfg, build_m1_smoke_action
        from rsl_rl.runners import OnPolicyRunner

        wheel_signs = tuple(float(value.strip()) for value in args.wheel_signs.split(","))
        if len(wheel_signs) != 4:
            raise ValueError("--wheel-signs must contain 4 comma-separated values")

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.control_mode = args.mode
        env_cfg.rolling_wheel_velocity = args.rolling_wheel_velocity
        env_cfg.wave_wheel_velocity = args.wave_wheel_velocity
        env_cfg.wave_amplitude = args.wave_amplitude
        env_cfg.wave_frequency = args.wave_frequency
        env_cfg.sim.device = args.device
        if args.checkpoint and hasattr(env_cfg, "wave_front_wheel_action"):
            env_cfg.wave_front_wheel_action = args.rolling_wheel_velocity
            env_cfg.wave_rear_wheel_action = args.rolling_wheel_velocity
        if args.enable_wave_reference_actions:
            env_cfg.wave_reference_actions = True
        if args.lock_legs:
            if not hasattr(env_cfg, "wave_leg_action_limit"):
                raise ValueError("--lock-legs requires an M1 wave/Pvcnn task")
            env_cfg.wave_leg_action_limit = 0.0
            env_cfg.wave_reference_actions = False
            env_cfg.wave_semantic_obstacle_gating = False
        if args.disable_crossing_reset and hasattr(env_cfg.terminations, "crossing_success"):
            env_cfg.terminations.crossing_success = None
        if args.wave_leg_action_limit is not None:
            env_cfg.wave_leg_action_limit = args.wave_leg_action_limit
        if args.wave_reference_amplitude is not None:
            env_cfg.wave_reference_raw_amplitude = args.wave_reference_amplitude
        if args.wave_reference_knee_ratio is not None:
            env_cfg.wave_reference_knee_ratio = args.wave_reference_knee_ratio
        if args.rear_amplitude_scale is not None:
            env_cfg.wave_rear_amplitude_scale = args.rear_amplitude_scale
        if args.front_support_ratio is not None:
            env_cfg.wave_front_support_ratio = args.front_support_ratio
        if args.rear_support_ratio is not None:
            env_cfg.wave_rear_support_ratio = args.rear_support_ratio
        if args.obstacle_wheel_action is not None:
            env_cfg.wave_obstacle_wheel_action = args.obstacle_wheel_action
        if args.stand_height is not None:
            env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.stand_height)
        if args.stand_hip is not None:
            env_cfg.scene.robot.init_state.joint_pos[".*HIP_JOINT"] = args.stand_hip
        if args.stand_knee is not None:
            env_cfg.scene.robot.init_state.joint_pos[".*KNEE_JOINT"] = args.stand_knee
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
        device = getattr(env.unwrapped, "device", env_cfg.sim.device)
        policy = None
        wrapped_env = None
        if args.checkpoint:
            pvcnn_model = None
            if args.perception_checkpoint is not None:
                from models.s3dis.pvcnn import PVCNN

                perception = torch.load(
                    args.perception_checkpoint, map_location="cpu", weights_only=False
                )
                pvcnn_model = PVCNN(
                    num_classes=3,
                    extra_feature_channels=0,
                    width_multiplier=float(perception["width_multiplier"]),
                ).to(device)
                pvcnn_model.load_state_dict(perception["pvcnn_state_dict"])
                wrapped_env = M1PvcnnRslRlEnvWrapper(
                    env.unwrapped, pvcnn_model, clip_actions=args.clip_actions
                )
            else:
                wrapped_env = M1RslRlEnvWrapper(env.unwrapped, clip_actions=args.clip_actions)
            runner = OnPolicyRunner(wrapped_env, get_m1_train_cfg(), log_dir=None, device=device)
            if pvcnn_model is not None:
                runner.alg.pvcnn_model = pvcnn_model
            runner.load(args.checkpoint, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            obs, _ = wrapped_env.get_observations()
        else:
            env.reset()
            controller_cfg = M1SmokeControllerCfg(
                rolling_wheel_velocity=args.rolling_wheel_velocity,
                wave_wheel_velocity=args.wave_wheel_velocity,
                wave_amplitude=args.wave_amplitude,
                wave_frequency=args.wave_frequency,
                wheel_velocity_signs=wheel_signs,
            )

        debug_robot = env.unwrapped.scene["robot"] if args.debug_root_delta else None
        debug_root_start = debug_robot.data.root_pos_w.clone() if debug_robot is not None else None
        dt = float(env_cfg.sim.dt * env_cfg.decimation)
        for step in range(args.steps):
            with torch.inference_mode():
                if policy is not None:
                    actions = policy(obs.to(device))
                    obs, _, _, _ = wrapped_env.step(actions)
                else:
                    actions = build_m1_smoke_action(
                        num_envs=args.num_envs,
                        time_s=step * dt,
                        mode=args.mode,
                        cfg=controller_cfg,
                        device=device,
                    )
                    env.step(actions)
        if debug_robot is not None:
            debug_root_end = debug_robot.data.root_pos_w.clone()
            print(f"[M1 debug] root_start={debug_root_start.cpu().numpy()}", flush=True)
            print(f"[M1 debug] root_end={debug_root_end.cpu().numpy()}", flush=True)
            print(f"[M1 debug] root_delta={(debug_root_end - debug_root_start).cpu().numpy()}", flush=True)
            print(f"[M1 debug] root_quat={debug_robot.data.root_quat_w.cpu().numpy()}", flush=True)
        env.close()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
