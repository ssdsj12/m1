#!/usr/bin/env python3
"""Audit and prune only numeric checkpoints above update 3500 from long-v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import NamedTuple


EXPECTED_LONG_V4_RUN_DIR = Path(
    "/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_coordinated/"
    "coordinated_teacher_long_v4_64x5000_20260823"
)
_NUMERIC_CHECKPOINT = re.compile(r"model_(\d+)\.pt\Z")


class PruningItem(NamedTuple):
    path: Path
    sha256: str


class PruningPlan(NamedTuple):
    run_dir: Path
    keep_through: int
    original_manifest_sha256: str
    kept_model_3500_sha256: str
    delete: tuple[PruningItem, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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


def build_pruning_plan(
    run_dir: Path,
    keep_through: int,
    *,
    expected_run_dir: Path = EXPECTED_LONG_V4_RUN_DIR,
) -> PruningPlan:
    if not isinstance(keep_through, int) or isinstance(keep_through, bool):
        raise TypeError("keep_through must be an integer")
    if keep_through != 3500:
        raise ValueError("this audit permits only keep_through=3500")
    expected = expected_run_dir.expanduser().resolve(strict=True)
    actual = run_dir.expanduser().resolve(strict=True)
    if actual != expected:
        raise ValueError(f"run directory is not the exact long-v4 directory: {actual}")
    if not actual.is_dir():
        raise ValueError(f"run directory must be a directory: {actual}")

    manifest = actual / "run_manifest.json"
    kept = actual / "model_3500.pt"
    _require_regular_file(manifest, label="original manifest")
    _require_regular_file(kept, label="model_3500.pt")

    delete: list[PruningItem] = []
    for path in actual.iterdir():
        match = _NUMERIC_CHECKPOINT.fullmatch(path.name)
        if match is None or int(match.group(1)) <= keep_through:
            continue
        _require_regular_file(path, label="numeric checkpoint")
        if path.parent.resolve(strict=True) != actual:
            raise ValueError(f"checkpoint escapes exact long-v4 directory: {path}")
        delete.append(PruningItem(path=path, sha256=sha256_file(path)))
    delete.sort(key=lambda item: int(_NUMERIC_CHECKPOINT.fullmatch(item.path.name).group(1)))
    return PruningPlan(
        run_dir=actual,
        keep_through=keep_through,
        original_manifest_sha256=sha256_file(manifest),
        kept_model_3500_sha256=sha256_file(kept),
        delete=tuple(delete),
    )


def _audit_payload(plan: PruningPlan, *, status: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "run_dir": str(plan.run_dir),
        "keep_through": plan.keep_through,
        "original_manifest_sha256": plan.original_manifest_sha256,
        "kept_model_3500_sha256": plan.kept_model_3500_sha256,
        "deleted": [
            {"name": item.path.name, "sha256": item.sha256}
            for item in plan.delete
        ],
    }


def verify_pruning_postcondition(
    run_dir: Path,
    keep_through: int,
    *,
    ignored_names: set[str] | None = None,
) -> None:
    ignored = ignored_names or set()
    remaining: list[str] = []
    for path in run_dir.iterdir():
        match = _NUMERIC_CHECKPOINT.fullmatch(path.name)
        if (
            match is not None
            and int(match.group(1)) > keep_through
            and path.name not in ignored
        ):
            remaining.append(path.name)
    if remaining:
        raise RuntimeError(
            "remaining numeric checkpoints above limit: "
            + ", ".join(sorted(remaining))
        )


def execute_pruning(plan: PruningPlan, *, apply: bool) -> dict[str, object]:
    if not apply:
        return _audit_payload(plan, status="dry_run")

    audit_path = plan.run_dir / "checkpoint_pruning.json"
    _atomic_write_json(audit_path, _audit_payload(plan, status="planned"))
    for item in plan.delete:
        _require_regular_file(item.path, label="numeric checkpoint")
        if item.path.parent.resolve(strict=True) != plan.run_dir:
            raise ValueError(f"checkpoint escapes exact long-v4 directory: {item.path}")
        if sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"checkpoint changed after audit: {item.path.name}")
        item.path.unlink()
    verify_pruning_postcondition(plan.run_dir, plan.keep_through)
    completed = _audit_payload(plan, status="completed")
    _atomic_write_json(audit_path, completed)
    return completed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keep-through", type=int, default=3500)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plan = build_pruning_plan(args.run_dir, args.keep_through)
    result = execute_pruning(plan, apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
