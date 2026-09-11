"""Deterministic group splits and atomic, hash-pinned fingertip-prior shards."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

import numpy as np

from .contracts import ExpertWindow, PRIOR_HORIZON


@dataclass(frozen=True)
class GroupSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        values = self.train + self.validation + self.test
        if any(type(value) is not str or not value for value in values):
            raise ValueError("split groups must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError("split groups must be globally unique")


@dataclass(frozen=True)
class ShardRecord:
    path: str
    sha256: str
    split: str
    samples: int
    source_counts: dict[str, int]
    hand_counts: dict[str, int]
    phase_counts: dict[str, int]
    contact_pattern_counts: dict[str, int]


@dataclass(frozen=True)
class AggregateManifest:
    format_version: int
    archive_manifest_sha256: str | None
    source_manifest_sha256: str | None
    split_groups: dict[str, tuple[str, ...]]
    shards: tuple[ShardRecord, ...]
    aggregate_sha256: str


def deterministic_group_split(groups: Sequence[str], seed: int) -> GroupSplit:
    """Return the frozen 80/10/10 assignment over sorted unique group IDs."""

    if isinstance(groups, (str, bytes)):
        raise TypeError("groups must be a sequence of strings")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    unique = sorted(set(groups))
    if any(type(group) is not str or not group for group in groups):
        raise ValueError("groups must contain non-empty strings")
    random.Random(seed).shuffle(unique)
    count = len(unique)
    train_end = int(0.8 * count)
    validation_end = int(0.9 * count)
    return GroupSplit(
        tuple(unique[:train_end]),
        tuple(unique[train_end:validation_end]),
        tuple(unique[validation_end:]),
    )


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _array_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
        ) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _array_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _window_key(window: ExpertWindow) -> tuple[str, str, str]:
    digest = sha256()
    for tensor in (
        window.fingertip_position_palm,
        window.fingertip_velocity_palm,
        window.contact_mask,
        window.future_fingertip_velocity_palm,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(bytes((int(window.phase),)))
    return window.source_group, window.source_sha256, digest.hexdigest()


def _arrays(windows: Sequence[ExpertWindow]) -> dict[str, np.ndarray]:
    return {
        "contact_mask": np.stack(
            [item.contact_mask.detach().cpu().numpy() for item in windows]
        ).astype(np.bool_, copy=False),
        "fingertip_position_palm": np.stack(
            [item.fingertip_position_palm.detach().cpu().numpy() for item in windows]
        ).astype(np.float32, copy=False),
        "fingertip_velocity_palm": np.stack(
            [item.fingertip_velocity_palm.detach().cpu().numpy() for item in windows]
        ).astype(np.float32, copy=False),
        "future_fingertip_velocity_palm": np.stack(
            [item.future_fingertip_velocity_palm.detach().cpu().numpy() for item in windows]
        ).astype(np.float32, copy=False),
        "phase": np.asarray([int(item.phase) for item in windows], dtype=np.uint8),
        "source_group": np.asarray([item.source_group for item in windows], dtype=np.str_),
        "source_sha256": np.asarray([item.source_sha256 for item in windows], dtype="<U64"),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    count = arrays["phase"].shape[0]
    expected = {
        "fingertip_position_palm": ((count, 5, 3), np.dtype(np.float32)),
        "fingertip_velocity_palm": ((count, 5, 3), np.dtype(np.float32)),
        "contact_mask": ((count, 5), np.dtype(np.bool_)),
        "future_fingertip_velocity_palm": (
            (count, PRIOR_HORIZON, 5, 3),
            np.dtype(np.float32),
        ),
        "phase": ((count,), np.dtype(np.uint8)),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"invalid shard array {name}")
        if arrays[name].dtype.kind in "f" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite shard array {name}")


def _audit_row(value: object) -> dict[str, object]:
    if is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError("audit rows must be dataclasses or mappings")
    _canonical_json(result)
    return result


def _sha_or_none(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def write_shards(
    output_root: str | Path,
    windows: Sequence[ExpertWindow],
    split: GroupSplit,
    *,
    shard_size: int = 4096,
    audits: Sequence[object] = (),
    archive_manifest_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
    group_hands: Mapping[str, str] | None = None,
) -> AggregateManifest:
    """Atomically install deterministic NPZ shards, audit JSONL, and manifest."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    if not isinstance(split, GroupSplit):
        raise TypeError("split must be a GroupSplit")
    archive_sha = _sha_or_none("archive_manifest_sha256", archive_manifest_sha256)
    source_sha = _sha_or_none("source_manifest_sha256", source_manifest_sha256)
    assignments: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for group in getattr(split, split_name):
            assignments[group] = split_name

    supplied_windows = tuple(windows)
    if any(not isinstance(window, ExpertWindow) for window in supplied_windows):
        raise TypeError("windows must contain ExpertWindow values")
    ordered = sorted(supplied_windows, key=_window_key)
    group_shas: dict[str, str] = {}
    for window in ordered:
        if window.source_group not in assignments:
            raise ValueError(f"window group is absent from split: {window.source_group}")
        previous_sha = group_shas.setdefault(window.source_group, window.source_sha256)
        if previous_sha != window.source_sha256:
            raise ValueError(f"window group has conflicting provenance: {window.source_group}")
        # Construction validates values, but tensors remain mutable after construction.
        for tensor in (
            window.fingertip_position_palm,
            window.fingertip_velocity_palm,
            window.future_fingertip_velocity_palm,
        ):
            if not bool(tensor.isfinite().all().item()):
                raise ValueError("window contains non-finite geometry")
    hands = {} if group_hands is None else dict(group_hands)
    if any(group not in assignments or type(hand) is not str or not hand for group, hand in hands.items()):
        raise ValueError("group_hands contains invalid provenance")
    audit_rows = sorted(
        (_audit_row(value) for value in audits),
        key=lambda row: _canonical_json(row),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.shards-", dir=destination.parent))
    records: list[ShardRecord] = []
    try:
        for split_name in ("train", "validation", "test"):
            selected = [item for item in ordered if assignments[item.source_group] == split_name]
            for shard_index, begin in enumerate(range(0, len(selected), shard_size)):
                batch = selected[begin : begin + shard_size]
                arrays = _arrays(batch)
                _validate_arrays(arrays)
                relative = Path(split_name) / f"shard-{shard_index:05d}.npz"
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.tmp")
                _write_deterministic_npz(temporary, arrays)
                os.replace(temporary, target)
                phase_counts = Counter(str(int(item.phase)) for item in batch)
                contact_counts = Counter(
                    "".join("1" if value else "0" for value in item.contact_mask.tolist())
                    for item in batch
                )
                source_counts = Counter(item.source_group.split("/", 1)[0] for item in batch)
                hand_counts = Counter(hands[item.source_group] for item in batch if item.source_group in hands)
                records.append(
                    ShardRecord(
                        path=relative.as_posix(),
                        sha256=sha256(target.read_bytes()).hexdigest(),
                        split=split_name,
                        samples=len(batch),
                        source_counts=dict(sorted(source_counts.items())),
                        hand_counts=dict(sorted(hand_counts.items())),
                        phase_counts=dict(sorted(phase_counts.items())),
                        contact_pattern_counts=dict(sorted(contact_counts.items())),
                    )
                )

        (staging / "audit.jsonl").write_bytes(b"".join(_canonical_json(row) for row in audit_rows))
        records.sort(key=lambda item: item.path)
        body = {
            "format_version": 1,
            "archive_manifest_sha256": archive_sha,
            "source_manifest_sha256": source_sha,
            "split_groups": {
                name: sorted(getattr(split, name)) for name in ("train", "validation", "test")
            },
            "shards": [asdict(record) for record in records],
        }
        aggregate_sha = sha256(_canonical_json(body)).hexdigest()
        document = {**body, "aggregate_sha256": aggregate_sha}
        manifest_target = staging / "aggregate_manifest.json"
        manifest_temporary = staging / ".aggregate_manifest.json.tmp"
        manifest_temporary.write_bytes(_canonical_json(document))
        os.replace(manifest_temporary, manifest_target)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return AggregateManifest(
        format_version=1,
        archive_manifest_sha256=archive_sha,
        source_manifest_sha256=source_sha,
        split_groups={
            name: tuple(sorted(getattr(split, name))) for name in ("train", "validation", "test")
        },
        shards=tuple(records),
        aggregate_sha256=aggregate_sha,
    )


__all__ = [
    "AggregateManifest",
    "GroupSplit",
    "ShardRecord",
    "deterministic_group_split",
    "write_shards",
]
