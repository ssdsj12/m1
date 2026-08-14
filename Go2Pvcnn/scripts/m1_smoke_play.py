#!/usr/bin/env python3
"""Play the M1 smoke environment with an open-loop rolling or wave controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run M1 smoke rolling/wave open-loop play.")
    parser.add_argument("--task", default="Isaac-M1-Smoke-v0")
    parser.add_argument("--mode", default="rolling", choices=["rolling", "wave"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rolling-wheel-velocity", type=float, default=4.0)
    parser.add_argument("--wave-wheel-velocity", type=float, default=1.5)
    parser.add_argument("--wave-amplitude", type=float, default=0.08)
    parser.add_argument("--wave-frequency", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym

        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_smoke_controller import M1SmokeControllerCfg, build_m1_smoke_action

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.control_mode = args.mode
        env_cfg.rolling_wheel_velocity = args.rolling_wheel_velocity
        env_cfg.wave_wheel_velocity = args.wave_wheel_velocity
        env_cfg.wave_amplitude = args.wave_amplitude
        env_cfg.wave_frequency = args.wave_frequency

        render_mode = "rgb_array" if getattr(args, "video", False) else None
        env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
        device = getattr(env.unwrapped, "device", "cpu")
        controller_cfg = M1SmokeControllerCfg(
            rolling_wheel_velocity=args.rolling_wheel_velocity,
            wave_wheel_velocity=args.wave_wheel_velocity,
            wave_amplitude=args.wave_amplitude,
            wave_frequency=args.wave_frequency,
        )

        env.reset()
        dt = float(env_cfg.sim.dt * env_cfg.decimation)
        for step in range(args.steps):
            actions = build_m1_smoke_action(
                num_envs=args.num_envs,
                time_s=step * dt,
                mode=args.mode,
                cfg=controller_cfg,
                device=device,
            )
            with torch.inference_mode():
                env.step(actions)
        env.close()
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
