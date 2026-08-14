#!/usr/bin/env python3
"""Run the real four-stage CPU acceptance chain for M1 + Panda Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for import_path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

TRAIN_SCRIPT = THIS_FILE.parent / "m1_panda_teacher_train.py"
STAGE_SEQUENCE = ("a0_initial", "a0_resume", "a1_initial", "a1_resume")
_MODEL_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_iteration(path: Path) -> int:
    """Parse the numeric iteration encoded by an RSL-RL checkpoint name."""
    match = _MODEL_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"checkpoint must match model_<iteration>.pt: {path}")
    return int(match.group(1))


def latest_checkpoint(run_dir: Path) -> Path:
    """Return the checkpoint with the largest numeric model suffix."""
    candidates = []
    for path in Path(run_dir).glob("model_*.pt"):
        match = _MODEL_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            candidates.append((checkpoint_iteration(path), path.resolve()))
    if not candidates:
        raise FileNotFoundError(f"no numeric model checkpoint in {run_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def validate_completed_manifest(
    manifest: dict,
    *,
    expected_stage: str,
    expected_base_sha256: str | None,
    require_frozen_hash: bool,
) -> None:
    """Validate stage completion and frozen/base continuity."""
    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"manifest status must be completed, got {manifest.get('status')!r}"
        )
    if manifest.get("stage") != expected_stage:
        raise RuntimeError(
            f"manifest stage must be {expected_stage}, got {manifest.get('stage')!r}"
        )
    if manifest.get("base_checkpoint_sha256") != expected_base_sha256:
        raise RuntimeError(
            "manifest base_checkpoint_sha256 mismatch: "
            f"expected {expected_base_sha256!r}, got "
            f"{manifest.get('base_checkpoint_sha256')!r}"
        )
    if require_frozen_hash:
        initial = manifest.get("frozen_actor_initial_sha256")
        final = manifest.get("frozen_actor_final_sha256")
        if not initial or initial != final:
            raise RuntimeError(
                f"frozen actor hash mismatch: initial={initial!r}, final={final!r}"
            )


def _load_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"run manifest must contain an object: {path}")
    return payload


def _run_child(label: str, command: list[str], output_root: Path) -> dict:
    stdout_path = output_root / f"{label}.stdout.log"
    stderr_path = output_root / f"{label}.stderr.log"
    try:
        result = subprocess.run(command, check=False, timeout=600, text=True, capture_output=True)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise RuntimeError(
            f"{label} timed out after 600 seconds; see {stdout_path} and {stderr_path}"
        ) from error
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    record = {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} returned {result.returncode}; see {stdout_path} and {stderr_path}"
        )
    return record


def _base_command(args) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--num_envs",
        "1",
        "--max_iterations",
        "1",
        "--num-steps-per-env",
        "4",
        "--learning-epochs",
        "1",
        "--num-mini-batches",
        "1",
        "--save-interval",
        "1",
        "--device",
        args.device,
    ]
    if args.headless:
        command.append("--headless")
    return command


def _validate_checkpoint(path: Path, stage: str, base_sha256: str | None) -> None:
    from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
        TEACHER_ACTION_DIM,
        TEACHER_HIDDEN_DIMS,
        TEACHER_OBSERVATION_DIM,
        validate_teacher_checkpoint,
    )

    validate_teacher_checkpoint(
        path,
        expected_stage=stage,
        expected_observation_dim=TEACHER_OBSERVATION_DIM,
        expected_action_dim=TEACHER_ACTION_DIM,
        expected_actor_hidden_dims=TEACHER_HIDDEN_DIMS,
        expected_base_sha256=base_sha256,
        require_optimizer=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    summary = {"sequence": list(STAGE_SEQUENCE), "runs": []}
    try:
        output_root.mkdir(parents=True, exist_ok=False)
        common = _base_command(args)

        a0_initial_command = common + [
            "--stage",
            "A0",
            "--run_name",
            "smoke_a0",
            "--log-root",
            str(output_root),
        ]
        summary["runs"].append(
            _run_child("a0_initial", a0_initial_command, output_root)
        )
        a0_run_dir = output_root / "a0" / "smoke_a0"
        a0_checkpoint = latest_checkpoint(a0_run_dir)
        a0_manifest = _load_manifest(a0_run_dir)
        validate_completed_manifest(
            a0_manifest,
            expected_stage="A0",
            expected_base_sha256=None,
            require_frozen_hash=False,
        )
        _validate_checkpoint(a0_checkpoint, "A0", None)
        a0_initial_iteration = checkpoint_iteration(a0_checkpoint)

        a0_resume_command = common + [
            "--stage",
            "A0",
            "--resume-checkpoint",
            str(a0_checkpoint),
        ]
        summary["runs"].append(
            _run_child("a0_resume", a0_resume_command, output_root)
        )
        a0_checkpoint = latest_checkpoint(a0_run_dir)
        if checkpoint_iteration(a0_checkpoint) <= a0_initial_iteration:
            raise RuntimeError("A0 resume did not advance the checkpoint iteration")
        a0_manifest = _load_manifest(a0_run_dir)
        validate_completed_manifest(
            a0_manifest,
            expected_stage="A0",
            expected_base_sha256=None,
            require_frozen_hash=False,
        )
        _validate_checkpoint(a0_checkpoint, "A0", None)

        from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import file_sha256

        a0_base_sha256 = file_sha256(a0_checkpoint)
        a1_initial_command = common + [
            "--stage",
            "A1",
            "--base-checkpoint",
            str(a0_checkpoint),
            "--run_name",
            "smoke_a1",
            "--log-root",
            str(output_root),
        ]
        summary["runs"].append(
            _run_child("a1_initial", a1_initial_command, output_root)
        )
        a1_run_dir = output_root / "a1" / "smoke_a1"
        a1_checkpoint = latest_checkpoint(a1_run_dir)
        a1_manifest = _load_manifest(a1_run_dir)
        validate_completed_manifest(
            a1_manifest,
            expected_stage="A1",
            expected_base_sha256=a0_base_sha256,
            require_frozen_hash=True,
        )
        _validate_checkpoint(a1_checkpoint, "A1", a0_base_sha256)
        a1_initial_iteration = checkpoint_iteration(a1_checkpoint)

        a1_resume_command = common + [
            "--stage",
            "A1",
            "--base-checkpoint",
            str(a0_checkpoint),
            "--resume-checkpoint",
            str(a1_checkpoint),
        ]
        summary["runs"].append(
            _run_child("a1_resume", a1_resume_command, output_root)
        )
        a1_checkpoint = latest_checkpoint(a1_run_dir)
        if checkpoint_iteration(a1_checkpoint) <= a1_initial_iteration:
            raise RuntimeError("A1 resume did not advance the checkpoint iteration")
        a1_manifest = _load_manifest(a1_run_dir)
        validate_completed_manifest(
            a1_manifest,
            expected_stage="A1",
            expected_base_sha256=a0_base_sha256,
            require_frozen_hash=True,
        )
        _validate_checkpoint(a1_checkpoint, "A1", a0_base_sha256)

        summary.update(
            {
                "status": "completed",
                "a0_checkpoint": str(a0_checkpoint),
                "a0_checkpoint_sha256": a0_base_sha256,
                "a0_checkpoint_validated": True,
                "a1_checkpoint": str(a1_checkpoint),
                "a1_checkpoint_validated": True,
                "a1_base_checkpoint_sha256": a1_manifest[
                    "base_checkpoint_sha256"
                ],
                "frozen_actor_initial_sha256": a1_manifest[
                    "frozen_actor_initial_sha256"
                ],
                "frozen_actor_final_sha256": a1_manifest[
                    "frozen_actor_final_sha256"
                ],
            }
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(summary, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
