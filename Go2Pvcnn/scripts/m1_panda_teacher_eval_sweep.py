#!/usr/bin/env python3
"""Run and rank strict full-scale M1 + Panda A1 checkpoint evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import traceback


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for import_path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

PLAY_SCRIPT = THIS_FILE.parent / "m1_panda_teacher_play.py"
EXPECTED_SEEDS = {42, 43, 44}


def validate_sweep_inputs(
    checkpoints: list[Path], seeds: list[int], output_dir: Path
) -> list[Path]:
    """Resolve immutable sweep inputs before creating any artifact."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    resolved = [Path(path).expanduser().resolve() for path in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint paths must be unique")
    for checkpoint in resolved:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if set(seeds) != EXPECTED_SEEDS or len(seeds) != 3:
        raise ValueError("seeds must be exactly 42, 43, and 44")
    if Path(output_dir).expanduser().resolve().exists():
        raise FileExistsError(
            f"output directory already exists: {Path(output_dir).resolve()}"
        )
    return resolved


def run_play_child(command: list[str], row_path: Path) -> None:
    """Run one strict Play child and require its JSON artifact."""
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Play child exit {result.returncode}: {result.stderr[-2000:]}"
        )
    if not row_path.is_file():
        raise FileNotFoundError(f"Play child did not write row: {row_path}")
    row_path.with_suffix(".stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    row_path.with_suffix(".stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )


def rank_completed_rows(row_paths: list[Path]) -> dict[str, object]:
    """Validate, aggregate, and rank completed row artifacts."""
    from go2_pvcnn.tasks.m1_panda_teacher_evaluation import (
        aggregate_candidate_summaries,
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in row_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation row must be a JSON object: {path}")
        checkpoint_sha = payload.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or not checkpoint_sha:
            raise ValueError(f"row has no checkpoint SHA-256: {path}")
        grouped[checkpoint_sha].append(payload)
    candidates = [
        aggregate_candidate_summaries(rows) for rows in grouped.values()
    ]
    if not candidates:
        raise ValueError("no completed evaluation rows")
    candidates.sort(key=lambda item: tuple(item["rank_key"]), reverse=True)
    return {
        "status": "completed",
        "candidates": candidates,
        "winner": candidates[0],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank A1 checkpoints under strict full-scale disturbance."
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main() -> int:
    try:
        args = build_arg_parser().parse_args()
        if args.num_envs <= 0:
            raise ValueError("--num-envs must be positive")
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        seeds = args.seed if args.seed is not None else [42, 43, 44]
        checkpoints = validate_sweep_inputs(
            args.checkpoint, seeds, args.output_dir
        )
        base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(
                f"base checkpoint does not exist: {base_checkpoint}"
            )

        from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
            atomic_write_manifest,
            file_sha256,
        )

        checkpoint_hashes = {
            checkpoint: file_sha256(checkpoint) for checkpoint in checkpoints
        }
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)

        rows: list[Path] = []
        for checkpoint in checkpoints:
            checkpoint_hash = checkpoint_hashes[checkpoint]
            for seed in sorted(seeds):
                row_path = output_dir / (
                    f"row-{checkpoint.stem}-{checkpoint_hash[:8]}-seed-{seed}.json"
                )
                command = [
                    sys.executable,
                    str(PLAY_SCRIPT),
                    "--stage",
                    "A1",
                    "--base-checkpoint",
                    str(base_checkpoint),
                    "--checkpoint",
                    str(checkpoint),
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--stats-interval",
                    str(args.steps),
                    "--full-scale-disturbance",
                    "--summary-json",
                    str(row_path),
                    "--device",
                    args.device,
                    "--headless",
                ]
                run_play_child(command, row_path)
                rows.append(row_path)

        ranking = rank_completed_rows(rows)
        ranking["base_checkpoint"] = str(base_checkpoint)
        ranking["base_checkpoint_sha256"] = file_sha256(base_checkpoint)
        ranking["num_envs"] = args.num_envs
        ranking["steps"] = args.steps
        ranking_path = output_dir / "ranking.json"
        atomic_write_manifest(ranking_path, ranking)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "winner": ranking["winner"],
                    "artifact": str(ranking_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
