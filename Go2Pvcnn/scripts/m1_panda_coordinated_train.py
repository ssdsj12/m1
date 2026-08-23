#!/usr/bin/env python3
"""Train the combined 23-effort M1 + Panda coordinated PPO prerequisite."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_arg_parser():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--run_name", default="m1_panda_coordinated_train")
    parser.add_argument("--init-a1-checkpoint", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.init_a1_checkpoint.is_file():
        raise FileNotFoundError(args.init_a1_checkpoint)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from agent import get_m1_panda_teacher_train_cfg
        from go2_pvcnn.tasks.m1_panda_coordinated_wrapper import M1PandaCoordinatedEnvWrapper

        task = "Isaac-M1-Panda-Coordinated-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(task, cfg=cfg).unwrapped
        wrapper = M1PandaCoordinatedEnvWrapper(env)
        train_cfg = deepcopy(get_m1_panda_teacher_train_cfg())
        train_cfg["policy"]["actor_hidden_dims"] = [256, 128]
        train_cfg["policy"]["critic_hidden_dims"] = [256, 128]
        train_cfg["save_interval"] = 100
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(ROOT / "logs/m1_panda_coordinated" / args.run_name), device=args.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        print({"task": task, "action_dim": wrapper.num_actions, "init_a1_checkpoint": str(args.init_a1_checkpoint), "iterations": args.max_iterations})
        return 0
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
