"""Real-Isaac startup probe for inline M1 dual-Panda O6 MPC execution."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--seed", type=int, default=7)
# AppLauncher supplies the standard --headless flag.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import go2_pvcnn.tasks  # noqa: F401, E402
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import (  # noqa: E402
    M1DualPandaO6BimanualEnvCfg,
)
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import (  # noqa: E402
    M1DualPandaO6BimanualWrapper,
)


def main() -> int:
    torch.manual_seed(args.seed)
    cfg = M1DualPandaO6BimanualEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
    env.reset(seed=args.seed)
    wrapper = M1DualPandaO6BimanualWrapper(env)
    unexpected_reset_count = 0
    finite_snapshot = True
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = wrapper.step()
        unexpected_reset_count += int(torch.count_nonzero(terminated | truncated).item())
        snapshot = wrapper.last_snapshot
        tensors = (
            snapshot.base_state,
            snapshot.m1_q,
            snapshot.left_arm.q,
            snapshot.right_arm.q,
            snapshot.left_hand.q,
            snapshot.right_hand.q,
            snapshot.box.pose_b,
        )
        finite_snapshot = finite_snapshot and all(torch.isfinite(value).all().item() for value in tensors)
    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    effort_finite = bool(torch.isfinite(wrapper.last_command.effort).all().item())
    report = {
        "action_dim": action_dim,
        "finite_snapshot": finite_snapshot,
        "finite_effort": effort_finite,
        "unexpected_reset_count": unexpected_reset_count,
        "steps": args.steps,
        "num_envs": args.num_envs,
    }
    passed = action_dim == 43 and finite_snapshot and effort_finite and unexpected_reset_count == 0
    report["passed"] = passed
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    wrapper.close()
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
