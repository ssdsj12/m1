#!/usr/bin/env python3
"""Supervised PVCNN pretraining from the M1 semantic scanner."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--width-multiplier", type=float, default=0.125)
    parser.add_argument("--target-size", type=int, default=16)
    parser.add_argument("--min-semantic-accuracy", type=float, default=0.97)
    parser.add_argument("--min-obstacle-recall", type=float, default=0.85)
    parser.add_argument("--clip-actions", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    exit_code = 1
    try:
        import gymnasium as gym
        import torch
        import torch.nn.functional as F

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_pvcnn_perception import (
            downsample_perception_maps,
            grid_elevation_to_point_cloud,
        )
        from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper
        from models.s3dis.pvcnn import PVCNN
        from rsl_rl.runners import OnPolicyRunner

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        torch.manual_seed(args.seed)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = M1RslRlEnvWrapper(env.unwrapped, clip_actions=args.clip_actions)
        runner = OnPolicyRunner(wrapped, get_m1_train_cfg(), log_dir=None, device=env_cfg.sim.device)
        runner.load(args.policy_checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env_cfg.sim.device)
        obs, _ = wrapped.get_observations()

        model = PVCNN(num_classes=3, extra_feature_channels=0, width_multiplier=args.width_multiplier).to(env_cfg.sim.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        class_weights = torch.tensor([1.0, 8.0, 8.0], device=env_cfg.sim.device)
        scanner = env.unwrapped.scene["semantic_height_scanner"]
        correct = total = obstacle_true = obstacle_hit = 0
        loss_sum = 0.0

        for step in range(args.steps):
            elevation, labels = downsample_perception_maps(
                scanner.data.elevation_map,
                scanner.data.semantic_map,
                target_size=args.target_size,
            )
            points = grid_elevation_to_point_cloud(elevation)
            model.train()
            logits = model(points.transpose(1, 2).contiguous())
            loss = F.cross_entropy(logits, labels.reshape(labels.shape[0], -1), weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            prediction = logits.detach().argmax(dim=1)
            flat_labels = labels.reshape(labels.shape[0], -1)
            correct += int((prediction == flat_labels).sum().item())
            total += int(flat_labels.numel())
            obstacle_mask = flat_labels > 0
            obstacle_true += int(obstacle_mask.sum().item())
            obstacle_hit += int(((prediction > 0) & obstacle_mask).sum().item())
            loss_sum += float(loss.item())

            model.eval()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = wrapped.step(actions)
            if (step + 1) % 100 == 0:
                semantic_accuracy = correct / max(total, 1)
                obstacle_recall = obstacle_hit / max(obstacle_true, 1)
                print(
                    f"[PVCNN pretrain] step={step + 1} loss={loss_sum / (step + 1):.5f} "
                    f"semantic_accuracy={semantic_accuracy:.5f} obstacle_recall={obstacle_recall:.5f}",
                    flush=True,
                )

        model.eval()
        eval_correct = eval_total = eval_obstacle_true = eval_obstacle_hit = 0
        for _ in range(args.eval_steps):
            elevation, labels = downsample_perception_maps(
                scanner.data.elevation_map,
                scanner.data.semantic_map,
                target_size=args.target_size,
            )
            points = grid_elevation_to_point_cloud(elevation)
            with torch.inference_mode():
                logits = model(points.transpose(1, 2).contiguous())
                prediction = logits.argmax(dim=1)
                actions = policy(obs)
                obs, _, _, _ = wrapped.step(actions)
            flat_labels = labels.reshape(labels.shape[0], -1)
            eval_correct += int((prediction == flat_labels).sum().item())
            eval_total += int(flat_labels.numel())
            obstacle_mask = flat_labels > 0
            eval_obstacle_true += int(obstacle_mask.sum().item())
            eval_obstacle_hit += int(((prediction > 0) & obstacle_mask).sum().item())

        semantic_accuracy = eval_correct / max(eval_total, 1)
        obstacle_recall = eval_obstacle_hit / max(eval_obstacle_true, 1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pvcnn_state_dict": model.state_dict(),
                "num_classes": 3,
                "width_multiplier": args.width_multiplier,
                "target_size": args.target_size,
                "semantic_accuracy": semantic_accuracy,
                "obstacle_recall": obstacle_recall,
                "steps": args.steps,
                "eval_steps": args.eval_steps,
                "seed": args.seed,
            },
            args.output,
        )
        print(
            f"semantic_accuracy={semantic_accuracy:.6f} obstacle_recall={obstacle_recall:.6f} "
            f"checkpoint={args.output.resolve()}",
            flush=True,
        )
        env.close()
        exit_code = int(
            semantic_accuracy < args.min_semantic_accuracy
            or obstacle_recall < args.min_obstacle_recall
        ) * 2
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
