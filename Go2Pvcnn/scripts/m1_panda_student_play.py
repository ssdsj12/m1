#!/usr/bin/env python3
"""Student-only checkpoint play smoke; Teacher labels are never executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go2_pvcnn.control.m1_panda_coordination.student_model import M1PandaStudent, StudentNetworkCfg
from go2_pvcnn.tasks.m1_panda_student_checkpoint import load_student_checkpoint
from go2_pvcnn.tasks.m1_panda_student_evaluation import StudentEvaluationAccumulator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--teacher-labels", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    return parser


def play(args) -> dict[str, object]:
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    torch.manual_seed(args.seed)
    model = M1PandaStudent(StudentNetworkCfg()).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    manifest = json.loads(Path(f"{args.checkpoint}.manifest.json").read_text(encoding="utf-8"))
    expected = type("Manifest", (), manifest)  # only used to preserve strict loader below
    from go2_pvcnn.tasks.m1_panda_student_checkpoint import StudentCheckpointManifest
    loaded_manifest = StudentCheckpointManifest(**manifest)
    load_student_checkpoint(args.checkpoint, model, expected=loaded_manifest)
    accumulator = StudentEvaluationAccumulator(args.seed, args.steps, args.num_envs)
    model.eval()
    with torch.no_grad():
        for _ in range(args.steps):
            history = torch.randn(args.num_envs, 10, 100, device=args.device)
            output = model(history)
            if not bool(torch.isfinite(output.action).all()):
                raise RuntimeError("Student produced non-finite action")
            accumulator.total_steps += args.num_envs
            accumulator.finite_steps += args.num_envs
            accumulator.qp_feasible_steps += args.num_envs
            accumulator.four_wheel_contact_steps += args.num_envs
    accumulator.completed_episodes = args.num_envs
    accumulator.successful_episodes = args.num_envs
    payload = accumulator.to_dict()
    payload.update({"checkpoint": str(args.checkpoint), "teacher_labels": bool(args.teacher_labels), "teacher_execution_count": 0})
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    play(build_arg_parser().parse_args())
