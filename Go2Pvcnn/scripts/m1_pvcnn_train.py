#!/usr/bin/env python3
"""Joint M1 PPO and PVCNN training on the semantic crossing course."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PVCNN_ROOT = ROOT.parent / "pvcnn"
RSL_RL_ROOT = ROOT / "rsl_rl"
for path in (ROOT, PVCNN_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

configure_pvcnn_cuda(ROOT.parent)


def build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-M1-Pvcnn-Crossing-60mm-v0")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--clip-actions", type=float, default=1.0)
    parser.add_argument("--pvcnn-train-interval", type=int, default=10)
    parser.add_argument("--pvcnn-train-epochs", type=int, default=1)
    parser.add_argument("--leg-noise-std", type=float, default=None)
    parser.add_argument("--freeze-leg-noise-std", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _freeze_leg_std_gradient(gradient):
    gradient = gradient.clone()
    gradient[:12] = 0.0
    return gradient


def main() -> int:
    args = build_parser().parse_args()
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    exit_code = 1
    try:
        import gymnasium as gym
        import torch

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_pvcnn_perception import M1PvcnnRslRlEnvWrapper
        from models.s3dis.pvcnn import PVCNN
        from rsl_rl.runners import OnPolicyRunner

        perception = torch.load(args.perception_checkpoint, map_location="cpu", weights_only=False)
        width_multiplier = float(perception["width_multiplier"])
        model = PVCNN(
            num_classes=3,
            extra_feature_channels=0,
            width_multiplier=width_multiplier,
        )
        model.load_state_dict(perception["pvcnn_state_dict"])

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        model = model.to(env_cfg.sim.device)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = M1PvcnnRslRlEnvWrapper(env.unwrapped, model, clip_actions=args.clip_actions)

        train_cfg = get_m1_train_cfg()
        train_cfg["enable_pvcnn_sync_training"] = True
        train_cfg["pvcnn_train_interval"] = args.pvcnn_train_interval
        train_cfg["pvcnn_train_epochs"] = args.pvcnn_train_epochs
        name = args.run_name or datetime.now().strftime("m1_pvcnn_joint_%Y-%m-%d_%H-%M-%S")
        log_dir = ROOT / "logs/m1_walk" / name
        log_dir.mkdir(parents=True, exist_ok=True)
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(log_dir), device=env_cfg.sim.device)
        runner.alg.pvcnn_model = model
        runner.alg.pvcnn_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        runner.load(args.policy_checkpoint, load_optimizer=False)
        if args.leg_noise_std is not None:
            with torch.no_grad():
                runner.alg.actor_critic.std[:12].fill_(args.leg_noise_std)
        if args.freeze_leg_noise_std:
            runner.alg.actor_critic.std.register_hook(_freeze_leg_std_gradient)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        final_path = log_dir / "pvcnn_final.pt"
        torch.save(
            {
                "pvcnn_state_dict": model.state_dict(),
                "width_multiplier": width_multiplier,
                "num_classes": 3,
            },
            final_path,
        )
        env.close()
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
