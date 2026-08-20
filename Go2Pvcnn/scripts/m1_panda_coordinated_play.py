#!/usr/bin/env python3
"""Run the combined M1 + Panda coordinated mission entrypoint."""

from __future__ import annotations

import argparse
import json

import torch

from go2_pvcnn.control.m1_panda_coordination.coordinated_teacher import CoordinatedTeacherAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-base-pose", nargs=3, type=float, default=(0.5, 0.0, 0.0))
    parser.add_argument("--ee-target-pose", nargs=6, type=float, default=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_envs <= 0 or args.max_steps <= 0:
        raise ValueError("--num_envs and --max_steps must be positive")
    torch.manual_seed(args.seed)
    adapter = CoordinatedTeacherAdapter()
    zero3 = torch.zeros(3)
    zero6 = torch.zeros(6)
    target_base = torch.tensor(args.target_base_pose, dtype=torch.float32)
    target_ee = torch.tensor(args.ee_target_pose, dtype=torch.float32)
    adapter.reset(zero3, zero6, target_base, target_ee, seed=args.seed)
    phase_counts: dict[str, int] = {}
    base_assist_count = 0
    ee_error = 0.0
    base_pose = zero3.clone()
    for _ in range(args.max_steps):
        decision = adapter.step(base_pose, zero6, target_ee, zero6, torch.tensor(0.05), torch.tensor(0.3))
        phase_counts[decision.phase.value] = phase_counts.get(decision.phase.value, 0) + 1
        base_assist_count += int(decision.base_assist_active)
        ee_error = float(torch.linalg.vector_norm(decision.mission.ee_target_pose - target_ee))
        base_pose = base_pose + decision.base_velocity * adapter.mission.cfg.physics_dt
    print(json.dumps({"phase_counts": phase_counts, "base_assist_count": base_assist_count, "ee_error": ee_error}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
