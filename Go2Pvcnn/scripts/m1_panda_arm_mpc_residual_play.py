#!/usr/bin/env python3
"""GUI playback for the 8D Arm-MPC residual policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"


def build_arg_parser(*, include_app_launcher_args: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.num_envs != 1:
        raise ValueError("GUI Play supports exactly one environment")
    if args.steps < 0:
        raise ValueError("steps must be non-negative")

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from agent import get_m1_panda_arm_mpc_residual_train_cfg
        from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
            M1PandaArmMpcResidualEnvWrapper,
        )
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env_cfg.viewer.eye = (2.5, 2.5, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.55)
        env_cfg.viewer.origin_type = "world"
        # Retain the viewport but disable the manager window whose delayed
        # callback otherwise touches a destroyed viewport_camera_controller.
        env_cfg.ui_window_class_type = None
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="human").unwrapped
        wrapper = M1PandaArmMpcResidualEnvWrapper(env, seed=args.seed)
        runner = OnPolicyRunner(
            wrapper,
            get_m1_panda_arm_mpc_residual_train_cfg(),
            log_dir=None,
            device=args.device,
        )
        runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=args.device)
        observations, _ = wrapper.reset()
        step = 0
        with torch.inference_mode():
            while app.is_running() and (args.steps == 0 or step < args.steps):
                actions = policy(observations)
                observations, _, _, _ = wrapper.step(actions)
                env.render()
                step += 1
        return 0
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
