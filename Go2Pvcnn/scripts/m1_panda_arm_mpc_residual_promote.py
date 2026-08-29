#!/usr/bin/env python3
"""Calibrate PhysX noise and promote residual candidates in isolated workers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import ResidualEvalMetrics
from go2_pvcnn.training.m1_panda_arm_mpc_residual_promotion import (
    PromotedCandidate,
    calibrate_tolerances,
    evaluate_candidate,
    select_promoted_candidate,
)


SEEDS = (42, 43, 44)
PAIR_INDICES = (0, 1, 2)
CANDIDATE_UPDATES = (0, 25, 50, 75, 100)
EVAL_STEPS = 4000


@dataclass(frozen=True)
class CandidateRecord:
    completed_updates: int
    checkpoint: Path
    checkpoint_sha256: str


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, target: Path) -> None:
    descriptor, raw = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_file(raw: object, *, parent: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_short_manifest(path: Path) -> tuple[dict[str, object], list[CandidateRecord]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("stage") != "short"
        or document.get("status") != "safe_complete"
        or document.get("accepted") is not False
        or document.get("promotion_required") is not True
    ):
        raise ValueError("promotion requires a safe-complete short manifest")
    for label in ("asset", "config", "reward"):
        file_path = _resolve_file(
            document.get(f"{label}_path"), parent=path.parent, label=f"{label}_path"
        )
        if document.get(f"{label}_sha256") != sha256_file(file_path):
            raise ValueError(f"short-run {label} SHA mismatch")
    raw_candidates = document.get("candidate_checkpoints")
    if not isinstance(raw_candidates, list):
        raise ValueError("short manifest must list candidate_checkpoints")
    records: list[CandidateRecord] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise TypeError("candidate records must be objects")
        updates = raw.get("completed_updates")
        if not isinstance(updates, int) or isinstance(updates, bool):
            raise TypeError("candidate completed_updates must be an integer")
        checkpoint = _resolve_file(
            raw.get("checkpoint"), parent=path.parent, label="candidate checkpoint"
        )
        digest = sha256_file(checkpoint)
        if raw.get("checkpoint_sha256") != digest:
            raise ValueError("short candidate checkpoint SHA mismatch")
        records.append(CandidateRecord(updates, checkpoint, digest))
    if {record.completed_updates for record in records} != set(CANDIDATE_UPDATES):
        raise ValueError("short manifest must contain exactly five completed-update candidates")
    return document, sorted(records, key=lambda value: value.completed_updates)


def _metrics(document: dict[str, object]) -> ResidualEvalMetrics:
    return ResidualEvalMetrics(
        hard_failure_count=int(document["hard_failure_count"]),
        mpc_feasible_rate=float(document["mpc_feasible_rate"]),
        qp_feasible_rate=float(document["qp_feasible_rate"]),
        four_contact_rate=float(document["four_contact_rate"]),
        roll_pitch_rms=float(document["roll_pitch_rms"]),
        base_height_rms=float(document["base_height_rms"]),
        ee_position_error=float(document["ee_position_error"]),
        ee_orientation_error=float(document["ee_orientation_error"]),
        wrench_error=float(document["wrench_error"]),
        slip=float(document["slip"]),
        intervention_ratio=float(document["intervention_ratio"]),
        saturation_fraction=tuple(float(value) for value in document["saturation_fraction"]),
    )


def _validate_worker(
    output: Path,
    *,
    mode: str,
    seed: int,
    checkpoint: Path | None,
) -> tuple[ResidualEvalMetrics, ResidualEvalMetrics]:
    document = json.loads(output.read_text(encoding="utf-8"))
    if (
        document.get("status") != "complete"
        or document.get("mode") != mode
        or document.get("seed") != seed
        or document.get("steps") != EVAL_STEPS
    ):
        raise RuntimeError("worker manifest contract mismatch")
    expected_sha = None if checkpoint is None else sha256_file(checkpoint)
    if document.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError("worker checkpoint SHA mismatch")
    baseline = document.get("baseline")
    candidate = document.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise RuntimeError("worker manifest is missing metrics")
    return _metrics(baseline), _metrics(candidate)


def _subprocess_worker(
    *,
    mode: str,
    seed: int,
    output: Path,
    checkpoint: Path | None,
    device: str,
    headless: bool,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/m1_panda_arm_mpc_residual_eval.py"),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--steps",
        str(EVAL_STEPS),
        "--device",
        device,
        "--output_json",
        str(output),
        "--headless" if headless else "--no-headless",
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    subprocess.run(
        command,
        cwd=ROOT.parent,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        check=True,
    )


def run_promotion(
    short_manifest: str | os.PathLike[str],
    *,
    worker_runner: Callable[..., None] | None = None,
    device: str = "cuda:0",
    headless: bool = True,
) -> dict[str, object]:
    manifest_path = Path(short_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    short_document, records = _load_short_manifest(manifest_path)
    run_dir = manifest_path.parent
    worker_runner = worker_runner or _subprocess_worker
    zero_pairs: list[tuple[ResidualEvalMetrics, ResidualEvalMetrics]] = []
    calibration_files: list[str] = []
    for seed in SEEDS:
        for pair_index in PAIR_INDICES:
            output = run_dir / "noise_calibration" / f"seed_{seed}_pair_{pair_index}.json"
            worker_runner(
                mode="zero-pair",
                seed=seed,
                output=output,
                checkpoint=None,
                device=device,
                headless=headless,
            )
            zero_pairs.append(
                _validate_worker(output, mode="zero-pair", seed=seed, checkpoint=None)
            )
            calibration_files.append(str(output))
    tolerances = calibrate_tolerances(zero_pairs)
    calibration_manifest = {
        "schema_version": 1,
        "status": "complete",
        "seeds": list(SEEDS),
        "pairs_per_seed": len(PAIR_INDICES),
        "steps": EVAL_STEPS,
        "worker_manifests": calibration_files,
        "tolerances": tolerances,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_dir / "noise_calibration.json", calibration_manifest)

    promoted: list[PromotedCandidate] = []
    candidate_documents: list[dict[str, object]] = []
    for record in records:
        seed_results = {}
        worker_files = []
        for seed in SEEDS:
            output = (
                run_dir
                / "candidate_eval"
                / f"candidate_u{record.completed_updates:03d}"
                / f"seed_{seed}.json"
            )
            worker_runner(
                mode="candidate",
                seed=seed,
                output=output,
                checkpoint=record.checkpoint,
                device=device,
                headless=headless,
            )
            seed_results[seed] = _validate_worker(
                output,
                mode="candidate",
                seed=seed,
                checkpoint=record.checkpoint,
            )
            worker_files.append(str(output))
        decision = evaluate_candidate(seed_results, tolerances)
        candidate = PromotedCandidate(
            str(record.checkpoint),
            record.completed_updates,
            record.checkpoint_sha256,
            decision,
        )
        promoted.append(candidate)
        candidate_documents.append(
            {
                "checkpoint": str(record.checkpoint),
                "checkpoint_sha256": record.checkpoint_sha256,
                "completed_updates": record.completed_updates,
                "worker_manifests": worker_files,
                "decision": asdict(decision),
            }
        )
    selected = select_promoted_candidate(promoted, tolerances)
    best_path = run_dir / "model_best.pt"
    if selected is not None:
        _atomic_copy(Path(selected.checkpoint), best_path)
        if sha256_file(best_path) != selected.sha256:
            raise RuntimeError("published best checkpoint SHA mismatch")
    result = {
        "schema_version": 1,
        "status": "accepted" if selected is not None else "rejected",
        "accepted": selected is not None,
        "short_manifest": str(manifest_path),
        "short_manifest_sha256": sha256_file(manifest_path),
        "asset_sha256": short_document["asset_sha256"],
        "config_sha256": short_document["config_sha256"],
        "reward_sha256": short_document["reward_sha256"],
        "noise_calibration": str(run_dir / "noise_calibration.json"),
        "tolerances": tolerances,
        "candidates": candidate_documents,
        "best_completed_updates": None if selected is None else selected.completed_updates,
        "best_checkpoint": None if selected is None else str(best_path),
        "best_checkpoint_sha256": None if selected is None else selected.sha256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_dir / "promotion_manifest.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short_manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_promotion(
        args.short_manifest, device=args.device, headless=args.headless
    )
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
