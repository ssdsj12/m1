#!/usr/bin/env python3
"""Train Student S1 with explicit supervised DAgger losses, without PPO."""

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

from go2_pvcnn.control.m1_panda_coordination.dagger import StudentLossCfg, student_dagger_loss
from go2_pvcnn.control.m1_panda_coordination.student_model import M1PandaStudent, StudentNetworkCfg
from go2_pvcnn.tasks.m1_panda_student_checkpoint import StudentCheckpointManifest, save_student_checkpoint
from go2_pvcnn.tasks.m1_panda_student_dataset import VersionedDaggerReplay
from go2_pvcnn.tasks.m1_panda_student_train_cfg import StudentTrainCfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--teacher-probability", type=float, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(args) -> Path:
    if args.output_dir.exists():
        raise ValueError(f"output directory must be fresh: {args.output_dir}")
    if not 0.0 <= args.teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be in [0,1]")
    manifest_path = args.dataset_manifest.resolve()
    outer = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / outer["shards"][0]["path"]
    replay_manifest = {key: outer[key] for key in ("schema_version", "asset_sha", "teacher_commit", "observation_dim", "history_length", "action_dim", "control_dt", "action_scales", "dagger_stage")}
    replay = VersionedDaggerReplay.load(shard_path, expected_manifest=replay_manifest)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = M1PandaStudent(StudentNetworkCfg()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    args.output_dir.mkdir(parents=True)
    records = list(replay.records)
    loss_cfg = StudentLossCfg()
    for epoch in range(args.epochs):
        random.shuffle(records)
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            if not batch: continue
            history = torch.stack([r.history for r in batch]).to(device)
            target_action = torch.stack([r.teacher_action for r in batch]).to(device)
            target_wrench = torch.stack([r.wrench_target for r in batch]).to(device)
            target_safety = torch.tensor([r.safety_target for r in batch], dtype=torch.float32, device=device)
            hard = torch.tensor([r.hard for r in batch], dtype=torch.bool, device=device)
            previous = torch.stack([r.executed_action for r in batch]).to(device)
            losses = student_dagger_loss(model(history), target_action, target_wrench, target_safety, hard, previous, loss_cfg)
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    scales = replay_manifest["action_scales"]
    checkpoint_manifest = StudentCheckpointManifest(
        schema_version=1, asset_sha=outer["asset_sha"], teacher_commit=outer["teacher_commit"],
        dataset_sha=_sha256(manifest_path), observation_dim=100, history_length=10, action_dim=23,
        action_scales=scales, control_dt=0.005, dagger_stage=args.stage,
        teacher_probability=args.teacher_probability, model_config={k: v for k, v in vars(model.cfg).items()},
        loss_weights={k: float(v) for k, v in vars(loss_cfg).items()},
    )
    best = args.output_dir / "best.pt"; last = args.output_dir / "last.pt"
    save_student_checkpoint(best, model, optimizer, checkpoint_manifest, global_step=args.epochs)
    save_student_checkpoint(last, model, optimizer, checkpoint_manifest, global_step=args.epochs)
    (args.output_dir / "training.manifest.json").write_text(json.dumps({"stage": args.stage, "epochs": args.epochs, "records": len(records), "checkpoint": "best.pt"}, indent=2) + "\n", encoding="utf-8")
    return best


def main() -> int:
    train(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
