#!/usr/bin/env python3
"""Autonomous Stage 1 -> Stage 2A controller for M1 locomotion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
WORKSPACE = ROOT.parent
LOG_ROOT = ROOT / "logs" / "m1_curriculum"
STATE_FILE = LOG_ROOT / "state.json"
STAGE1_DIR = ROOT / "logs" / "m1_walk" / "m1_roll_stage05_forward_long"
STAGE1_TARGET = STAGE1_DIR / "model_2999.pt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2_pvcnn.tasks.m1_curriculum import discover_latest_checkpoint


def _write_state(stage: str, **details) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_at": time.time(), **details}
    STATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], log_name: str) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / log_name).open("a") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        return result.returncode


def _stage1_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "m1_roll_stage05_forward_long"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _wait_for_stage1() -> Path:
    while not STAGE1_TARGET.exists():
        latest = None
        try:
            latest = discover_latest_checkpoint(STAGE1_DIR)
        except FileNotFoundError:
            pass
        _write_state("waiting_stage1", latest_checkpoint=str(latest) if latest else None)
        if not _stage1_running() and latest is not None:
            _write_state("restarting_stage1", checkpoint=str(latest))
            _run(
                [
                    sys.executable,
                    "Go2Pvcnn/scripts/m1_train.py",
                    "--headless",
                    "--task",
                    "Isaac-M1-Roll-v0",
                    "--num_envs",
                    "64",
                    "--max_iterations",
                    "1000",
                    "--run_name",
                    "m1_roll_stage05_forward_recovery",
                    "--clip-actions",
                    "1.0",
                    "--load_checkpoint",
                    str(latest),
                ],
                "stage1_recovery.console.log",
            )
        time.sleep(15.0)
    return STAGE1_TARGET


def _evaluate(task: str, checkpoint: Path, report_path: Path, promote_path: Path) -> dict:
    _run(
        [
            sys.executable,
            "Go2Pvcnn/scripts/m1_checkpoint_eval.py",
            "--headless",
            "--task",
            task,
            "--checkpoint",
            str(checkpoint),
            "--num_envs",
            "20",
            "--steps",
            "320",
            "--clip-actions",
            "1.0",
            "--report",
            str(report_path),
            "--promote",
            str(promote_path),
        ],
        f"{report_path.stem}.console.log",
    )
    return json.loads(report_path.read_text())


def main() -> int:
    stage1_checkpoint = _wait_for_stage1()
    stage1_report_path = LOG_ROOT / "stage1_roll" / "evaluation.json"
    stage1_accepted = LOG_ROOT / "stage1_roll" / "accepted.pt"
    _write_state("evaluating_stage1", checkpoint=str(stage1_checkpoint))
    report = _evaluate("Isaac-M1-Roll-v0", stage1_checkpoint, stage1_report_path, stage1_accepted)
    if not report["passed"]:
        _write_state("stage1_gate_failed", report=str(stage1_report_path))
        return 2

    stage2a_initialized = LOG_ROOT / "stage2a_wave_flat" / "initialized.pt"
    prepare_code = _run(
        [
            sys.executable,
            "Go2Pvcnn/scripts/m1_prepare_wave_checkpoint.py",
            str(stage1_accepted),
            str(stage2a_initialized),
        ],
        "stage2a_prepare.console.log",
    )
    if prepare_code != 0:
        _write_state("stage2a_prepare_failed", return_code=prepare_code)
        return prepare_code

    _write_state("training_stage2a", checkpoint=str(stage2a_initialized))
    return_code = _run(
        [
            sys.executable,
            "Go2Pvcnn/scripts/m1_train.py",
            "--headless",
            "--task",
            "Isaac-M1-Wave-Flat-v0",
            "--num_envs",
            "64",
            "--max_iterations",
            "300",
            "--run_name",
            "m1_wave_flat_stage2a_reference005",
            "--clip-actions",
            "1.0",
            "--load_checkpoint",
            str(stage2a_initialized),
            "--reset-optimizer",
        ],
        "stage2a_train.console.log",
    )
    if return_code != 0:
        _write_state("stage2a_train_failed", return_code=return_code)
        return return_code

    stage2a_dir = ROOT / "logs" / "m1_walk" / "m1_wave_flat_stage2a_reference005"
    stage2a_checkpoint = discover_latest_checkpoint(stage2a_dir)
    stage2a_report_path = LOG_ROOT / "stage2a_wave_flat" / "evaluation.json"
    stage2a_accepted = LOG_ROOT / "stage2a_wave_flat" / "accepted.pt"
    report = _evaluate("Isaac-M1-Wave-Flat-v0", stage2a_checkpoint, stage2a_report_path, stage2a_accepted)
    if not report["passed"]:
        _write_state("stage2a_gate_failed", report=str(stage2a_report_path))
        return 2
    _write_state("stage2a_complete", checkpoint=str(stage2a_accepted), report=str(stage2a_report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
