#!/usr/bin/env python3
"""Train M1 locomotion with RSL-RL PPO."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train M1 locomotion with RSL-RL PPO.")
    parser.add_argument("--task", default="Isaac-M1-Walk-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--reset-optimizer", action="store_true", default=False)
    parser.add_argument("--clip-actions", dest="clip_actions", type=float, default=0.02)
    parser.add_argument("--init-noise-std", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _build_log_dir(run_name: str | None) -> str:
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = GO2PVCNN_ROOT / "logs/m1_walk" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        import gymnasium as gym

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from rsl_rl.runners import OnPolicyRunner

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device

        env = gym.make(args.task, cfg=env_cfg)
        assert isinstance(env.unwrapped, ManagerBasedRLEnv)
        wrapped_env = M1RslRlEnvWrapper(env.unwrapped, clip_actions=args.clip_actions)

        train_cfg = get_m1_train_cfg()
        if args.init_noise_std is not None:
            train_cfg["policy"]["init_noise_std"] = args.init_noise_std
        log_dir = _build_log_dir(args.run_name)
        dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), env_cfg.to_dict())
        dump_yaml(os.path.join(log_dir, "train_cfg.yaml"), train_cfg)

        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=env_cfg.sim.device)
        if args.resume or args.load_checkpoint:
            if not args.load_checkpoint:
                raise ValueError("--resume for m1_train.py requires --load_checkpoint")
            runner.load(args.load_checkpoint, load_optimizer=not args.reset_optimizer)
        if args.init_noise_std is not None:
            runner.alg.actor_critic.std.data.fill_(args.init_noise_std)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        env.close()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
