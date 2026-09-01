"""Fail-closed lineage and legacy-normalizer migration for Phase 6 bridge training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import torch

from .m1_panda_arm_mpc_residual_lineage import (
    ResidualSourcePaths,
    sha256_file,
    validate_source_lineage,
)


BRIDGE_PARENT_UPDATES = 100
BRIDGE_ADDITIONAL_UPDATES = 200
BRIDGE_TOTAL_UPDATES = 300
BRIDGE_CANDIDATE_UPDATES = (100, 150, 200, 250, 300)
BRIDGE_PARENT_CANDIDATES = (0, 25, 50, 75, 100)
BRIDGE_LEGACY_SAMPLE_COUNT = 100 * 256 * 8
_NORMALIZER_KEYS = ("obs_norm_state_dict", "critic_obs_norm_state_dict")


@dataclass(frozen=True)
class BridgeParent:
    short_manifest: Path
    short_manifest_sha256: str
    pilot_manifest: Path
    pilot_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    completed_updates: int
    sample_count: int


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


def _load_document(path: Path, *, label: str) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _validate_accepted_pilot(
    raw_path: object,
    expected_sha256: object,
    *,
    parent: Path,
    source_paths: ResidualSourcePaths,
) -> tuple[Path, str]:
    pilot = _resolve_file(raw_path, parent=parent, label="pilot_manifest")
    digest = sha256_file(pilot)
    if expected_sha256 != digest:
        raise ValueError("short-run pilot manifest SHA mismatch")
    document = _load_document(pilot, label="pilot manifest")
    summaries = document.get("optimizer_summaries")
    decision = document.get("pilot_decision")
    if (
        document.get("schema_version") != 2
        or document.get("stage") != "pilot"
        or document.get("status") != "safe_complete"
        or document.get("accepted") is not False
        or document.get("promotion_required") is not False
        or document.get("pilot_accepted") is not True
        or document.get("requested_iterations") != 10
        or document.get("completed_iterations") != 10
        or not isinstance(summaries, list)
        or len(summaries) != 10
        or [value.get("update") for value in summaries if isinstance(value, dict)]
        != list(range(1, 11))
        or not isinstance(decision, dict)
        or decision.get("accepted") is not True
    ):
        raise ValueError("bridge parent pilot is not an accepted 10/10 pilot")
    validate_source_lineage(document, source_paths)
    return pilot, digest


def _load_checkpoint(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    return payload


def _validate_legacy_normalizers(payload: Mapping[str, object]) -> None:
    for key in _NORMALIZER_KEYS:
        state = payload.get(key)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"legacy checkpoint is missing normalizer state {key}")
        if "_count" in state:
            raise ValueError("bridge migration requires a legacy checkpoint without normalizer counts")


def validate_bridge_parent(
    manifest_path: str | os.PathLike[str],
    source_paths: ResidualSourcePaths,
) -> BridgeParent:
    """Validate the immutable v6 short lineage and return its legacy u100 parent."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    document = _load_document(manifest, label="short manifest")
    if (
        document.get("schema_version") != 2
        or document.get("stage") != "short"
        or document.get("status") != "safe_complete"
        or document.get("accepted") is not False
        or document.get("promotion_required") is not True
        or document.get("requested_iterations") != BRIDGE_PARENT_UPDATES
        or document.get("completed_iterations") != BRIDGE_PARENT_UPDATES
    ):
        raise ValueError("bridge requires a safe-complete schema-v2 100/100 short manifest")
    validate_source_lineage(document, source_paths)
    pilot, pilot_sha = _validate_accepted_pilot(
        document.get("pilot_manifest"),
        document.get("pilot_manifest_sha256"),
        parent=manifest.parent,
        source_paths=source_paths,
    )

    raw_candidates = document.get("candidate_checkpoints")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(
        BRIDGE_PARENT_CANDIDATES
    ):
        raise ValueError("bridge parent must contain exactly five candidate checkpoints")
    candidates: dict[int, tuple[Path, str]] = {}
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise TypeError("bridge parent candidate records must be objects")
        updates = raw.get("completed_updates")
        if not isinstance(updates, int) or isinstance(updates, bool):
            raise TypeError("candidate completed_updates must be an integer")
        if updates in candidates:
            raise ValueError("bridge parent contains a duplicate candidate update")
        checkpoint = _resolve_file(
            raw.get("checkpoint"), parent=manifest.parent, label="candidate checkpoint"
        )
        digest = sha256_file(checkpoint)
        if raw.get("checkpoint_sha256") != digest:
            raise ValueError("bridge parent candidate checkpoint SHA mismatch")
        candidates[updates] = (checkpoint, digest)
    if set(candidates) != set(BRIDGE_PARENT_CANDIDATES):
        raise ValueError("bridge parent candidate updates do not match the required set")

    checkpoint, checkpoint_sha = candidates[BRIDGE_PARENT_UPDATES]
    _validate_legacy_normalizers(_load_checkpoint(checkpoint))
    return BridgeParent(
        short_manifest=manifest,
        short_manifest_sha256=sha256_file(manifest),
        pilot_manifest=pilot,
        pilot_manifest_sha256=pilot_sha,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        completed_updates=BRIDGE_PARENT_UPDATES,
        sample_count=BRIDGE_LEGACY_SAMPLE_COUNT,
    )


def validate_counted_checkpoint(
    checkpoint: str | os.PathLike[str], *, expected_count: int | None = None
) -> tuple[int, int]:
    """Require equal, non-negative scalar-int64 actor and critic sample counts."""

    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("expected_count must be a non-negative integer")
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_checkpoint(path)
    counts: list[int] = []
    for key in _NORMALIZER_KEYS:
        state = payload.get(key)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"checkpoint is missing normalizer state {key}")
        count = state.get("_count")
        if (
            not isinstance(count, torch.Tensor)
            or count.dtype != torch.int64
            or count.shape != torch.Size([])
            or int(count.item()) < 0
        ):
            raise ValueError(f"{key} must contain a non-negative scalar-int64 _count")
        counts.append(int(count.item()))
    if counts[0] != counts[1]:
        raise ValueError("actor and critic normalizer counts must match")
    if expected_count is not None and counts[0] != expected_count:
        raise ValueError("normalizer count does not match the expected sample count")
    return counts[0], counts[1]


def migrate_legacy_u100_checkpoint(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    expected_parent_sha256: str,
    sample_count: int,
) -> str:
    """Atomically add the approved legacy count without altering checkpoint values."""

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    if sha256_file(source_path) != expected_parent_sha256:
        raise ValueError("legacy parent checkpoint SHA mismatch")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    payload = _load_checkpoint(source_path)
    _validate_legacy_normalizers(payload)
    migrated = dict(payload)
    for key in _NORMALIZER_KEYS:
        state = dict(payload[key])
        state["_count"] = torch.tensor(sample_count, dtype=torch.int64)
        migrated[key] = state

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=target_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(migrated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    validate_counted_checkpoint(target_path, expected_count=sample_count)
    return sha256_file(target_path)


__all__ = [
    "BRIDGE_ADDITIONAL_UPDATES",
    "BRIDGE_CANDIDATE_UPDATES",
    "BRIDGE_LEGACY_SAMPLE_COUNT",
    "BRIDGE_PARENT_CANDIDATES",
    "BRIDGE_PARENT_UPDATES",
    "BRIDGE_TOTAL_UPDATES",
    "BridgeParent",
    "migrate_legacy_u100_checkpoint",
    "validate_bridge_parent",
    "validate_counted_checkpoint",
]
