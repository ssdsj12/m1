#!/usr/bin/env python3
"""Run exact three-seed Student-only CPU evaluation aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAY = PROJECT_ROOT / "scripts/m1_panda_student_play.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go2_pvcnn.tasks.m1_panda_student_evaluation import EXPECTED_STUDENT_SEEDS, EVALUATION_STEPS, MIN_EVALUATION_ENVS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=EVALUATION_STEPS)
    parser.add_argument("--device", default="cpu")
    return parser


def main(args) -> int:
    if tuple(EXPECTED_STUDENT_SEEDS) != (42, 43, 44):
        raise ValueError("evaluation seed contract drifted")
    if args.num_envs < MIN_EVALUATION_ENVS or args.steps != EVALUATION_STEPS:
        raise ValueError("Student evaluation requires at least 64 envs and exactly 4000 steps")
    if args.output_dir.exists():
        raise ValueError("output directory must be fresh")
    args.output_dir.mkdir(parents=True)
    rows = []
    for seed in EXPECTED_STUDENT_SEEDS:
        summary = args.output_dir / f"seed-{seed}.json"
        subprocess.run([sys.executable, str(PLAY), "--checkpoint", str(args.checkpoint), "--steps", str(args.steps), "--num-envs", str(args.num_envs), "--seed", str(seed), "--summary-json", str(summary), "--device", args.device], check=True)
        rows.append(json.loads(summary.read_text(encoding="utf-8")))
    aggregate = {"seeds": list(EXPECTED_STUDENT_SEEDS), "rows": rows, "teacher_execution_count": sum(row["teacher_execution_count"] for row in rows), "success_rate": sum(row["successful_episodes"] for row in rows) / max(sum(row["completed_episodes"] for row in rows), 1)}
    (args.output_dir / "ranking.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_arg_parser().parse_args()))
