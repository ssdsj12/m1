#!/usr/bin/env python3
"""Collect a deterministic, versioned S1 replay shard for Student preheat.

The collection boundary is intentionally pure PyTorch for the first S1 smoke:
it exercises the frozen observation/action/replay contracts without requiring a
GUI or an RSL-RL runner. Isaac collection can replace the sample generator
behind the same record/manifest contract later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
    StudentActionScaleCfg,
)
from go2_pvcnn.tasks.m1_panda_student_dataset import DaggerRecord, VersionedDaggerReplay


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-M1-Panda-Student-S1-v0")
    parser.add_argument("--accepted-asset-sha", required=True)
    parser.add_argument("--teacher-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", required=True, choices=("teacher-warmup", "dagger-1", "dagger-2", "dagger-3"))
    parser.add_argument("--teacher-probability", type=float, required=True)
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num-envs and steps must be positive")
    if not 0.0 <= args.teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be in [0,1]")
    if args.stage == "teacher-warmup":
        if args.teacher_probability != 1.0:
            raise ValueError("teacher-warmup requires 1.0 teacher_probability")
        if args.student_checkpoint is not None:
            raise ValueError("teacher-warmup does not accept a Student checkpoint")
    elif args.student_checkpoint is None:
        raise ValueError("mixed DAgger stages require --student-checkpoint")
    if args.output_dir.exists():
        raise ValueError(f"output directory must be fresh: {args.output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(args) -> dict[str, object]:
    scales = {
        "leg_position_rad": StudentActionScaleCfg().leg_position_rad,
        "wheel_velocity_radps": StudentActionScaleCfg().wheel_velocity_radps,
        "arm_position_rad": StudentActionScaleCfg().arm_position_rad,
    }
    return {
        "schema_version": 1,
        "asset_sha": args.accepted_asset_sha,
        "teacher_commit": args.teacher_commit,
        "observation_dim": STUDENT_OBSERVATION_DIM,
        "history_length": STUDENT_HISTORY_LENGTH,
        "action_dim": STUDENT_ACTION_DIM,
        "control_dt": 0.005,
        "action_scales": scales,
        "dagger_stage": args.stage,
    }


def collect(args) -> Path:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True)
    replay = VersionedDaggerReplay(
        capacity=max(1024, args.num_envs * args.steps), hard_fraction=0.25, seed=args.seed
    )
    for step in range(args.steps):
        for env_id in range(args.num_envs):
            generator = torch.Generator().manual_seed(args.seed + env_id * 100003 + step)
            history = torch.randn(STUDENT_HISTORY_LENGTH, STUDENT_OBSERVATION_DIM, generator=generator) * 0.02
            teacher_action = torch.tanh(history[-1, :STUDENT_ACTION_DIM])
            executed = teacher_action.clone()
            if args.stage != "teacher-warmup" and torch.rand((), generator=generator).item() > args.teacher_probability:
                executed = torch.zeros_like(teacher_action)
            replay.add(DaggerRecord(
                env_id=env_id, episode_id=0, step=step, history=history,
                teacher_action=teacher_action, executed_action=executed,
                wrench_target=history[-1, 71:77].clone(), safety_target=float(executed.abs().max() < 0.95),
                hard=bool(executed.abs().max() < 0.95),
                metadata={"qp_status": "TRACK", "safety_state": "TRACK", "takeover_reason": "none"},
            ))
    shard = args.output_dir / "shard-00000.pt"
    manifest = _manifest(args)
    replay.save(shard, manifest)
    payload = dict(manifest)
    payload.update({"task": args.task, "num_envs": args.num_envs, "steps": args.steps,
                    "seed": args.seed, "record_count": len(replay), "shards": [{"path": shard.name, "sha256": _sha256(shard)}]})
    (args.output_dir / "dataset.manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shard


def main() -> int:
    args = build_arg_parser().parse_args()
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
