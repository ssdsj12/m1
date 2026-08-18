"""Versioned, hard-sample-aware replay storage for M1 + Panda Student S1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

import torch

from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
)


DATASET_SCHEMA_VERSION = 1
_MANIFEST_FIELDS = (
    "schema_version",
    "asset_sha",
    "teacher_commit",
    "observation_dim",
    "history_length",
    "action_dim",
    "control_dt",
    "action_scales",
    "dagger_stage",
)


@dataclass(frozen=True)
class DaggerRecord:
    env_id: int
    episode_id: int
    step: int
    history: torch.Tensor
    teacher_action: torch.Tensor
    executed_action: torch.Tensor
    wrench_target: torch.Tensor
    safety_target: float
    hard: bool
    metadata: dict[str, object]


def _validate_manifest(manifest: dict[str, object]) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    for field in _MANIFEST_FIELDS:
        if field not in manifest:
            raise ValueError(f"manifest missing {field}")
    fixed = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "observation_dim": STUDENT_OBSERVATION_DIM,
        "history_length": STUDENT_HISTORY_LENGTH,
        "action_dim": STUDENT_ACTION_DIM,
        "control_dt": 0.005,
    }
    for field, expected in fixed.items():
        if manifest[field] != expected:
            raise ValueError(
                f"manifest {field} mismatch: expected {expected!r}, got {manifest[field]!r}"
            )
    for field in ("asset_sha", "teacher_commit", "dagger_stage"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"manifest {field} must be a non-empty string")
    scales = manifest["action_scales"]
    if not isinstance(scales, dict) or not scales:
        raise ValueError("manifest action_scales must be a non-empty dictionary")
    for name, value in scales.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("manifest action_scales must contain positive finite values")
    try:
        canonical = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("manifest must be finite JSON data") from error
    return canonical


def _clone_record(record: DaggerRecord) -> DaggerRecord:
    if not isinstance(record, DaggerRecord):
        raise TypeError("record must be a DaggerRecord")
    for field in ("env_id", "episode_id", "step"):
        value = getattr(record, field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    tensor_contracts = {
        "history": (STUDENT_HISTORY_LENGTH, STUDENT_OBSERVATION_DIM),
        "teacher_action": (STUDENT_ACTION_DIM,),
        "executed_action": (STUDENT_ACTION_DIM,),
        "wrench_target": (6,),
    }
    tensors: dict[str, torch.Tensor] = {}
    for field, shape in tensor_contracts.items():
        value = getattr(record, field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{field} must have shape {shape}, got {tuple(value.shape)}")
        if not value.dtype.is_floating_point:
            raise TypeError(f"{field} must have floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{field} must contain only finite values")
        tensors[field] = value.detach().cpu().clone()
    if (
        not isinstance(record.safety_target, (int, float))
        or isinstance(record.safety_target, bool)
        or not math.isfinite(float(record.safety_target))
        or not 0.0 <= float(record.safety_target) <= 1.0
    ):
        raise ValueError("safety_target must be finite and in [0,1]")
    if not isinstance(record.hard, bool):
        raise TypeError("hard must be boolean")
    if not isinstance(record.metadata, dict):
        raise TypeError("metadata must be a dictionary")
    try:
        metadata = json.loads(json.dumps(record.metadata, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON data") from error
    return replace(
        record,
        **tensors,
        safety_target=float(record.safety_target),
        metadata=metadata,
    )


def _record_to_payload(record: DaggerRecord) -> dict[str, object]:
    return {
        "env_id": record.env_id,
        "episode_id": record.episode_id,
        "step": record.step,
        "history": record.history,
        "teacher_action": record.teacher_action,
        "executed_action": record.executed_action,
        "wrench_target": record.wrench_target,
        "safety_target": record.safety_target,
        "hard": record.hard,
        "metadata": record.metadata,
    }


def _payload_to_record(payload: object) -> DaggerRecord:
    if not isinstance(payload, dict):
        raise ValueError("serialized record must be a dictionary")
    try:
        record = DaggerRecord(**payload)
    except TypeError as error:
        raise ValueError("serialized record fields do not match DaggerRecord") from error
    return _clone_record(record)


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json_save(path: Path, payload: dict[str, object]) -> None:
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


class VersionedDaggerReplay:
    """Two deterministic reservoirs with a fixed hard-sample reservation."""

    def __init__(self, capacity: int, hard_fraction: float, seed: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            not isinstance(hard_fraction, (int, float))
            or isinstance(hard_fraction, bool)
            or not math.isfinite(float(hard_fraction))
            or not 0.0 <= float(hard_fraction) <= 1.0
        ):
            raise ValueError("hard_fraction must be finite and in [0,1]")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.capacity = capacity
        self.hard_fraction = float(hard_fraction)
        self.seed = seed
        self._hard_capacity = math.ceil(capacity * self.hard_fraction)
        self._normal_capacity = capacity - self._hard_capacity
        self._normal: list[DaggerRecord] = []
        self._hard: list[DaggerRecord] = []
        self._normal_seen = 0
        self._hard_seen = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._normal) + len(self._hard)

    @property
    def records(self) -> tuple[DaggerRecord, ...]:
        return tuple(_clone_record(record) for record in (*self._normal, *self._hard))

    def add(self, record: DaggerRecord) -> None:
        stored = _clone_record(record)
        reservoir = self._hard if stored.hard else self._normal
        capacity = self._hard_capacity if stored.hard else self._normal_capacity
        if stored.hard:
            self._hard_seen += 1
            seen = self._hard_seen
        else:
            self._normal_seen += 1
            seen = self._normal_seen
        if capacity == 0:
            return
        if len(reservoir) < capacity:
            reservoir.append(stored)
            return
        replacement = self._rng.randrange(seen)
        if replacement < capacity:
            reservoir[replacement] = stored

    def extend(self, records: Iterable[DaggerRecord]) -> None:
        for record in records:
            self.add(record)

    def sample(self, count: int, *, hard_weight: float = 2.0) -> list[DaggerRecord]:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")
        if (
            not isinstance(hard_weight, (int, float))
            or isinstance(hard_weight, bool)
            or not math.isfinite(float(hard_weight))
            or float(hard_weight) < 1.0
        ):
            raise ValueError("hard_weight must be finite and at least 1")
        records = [*self._normal, *self._hard]
        if not records:
            raise ValueError("cannot sample an empty replay")
        weights = [float(hard_weight) if record.hard else 1.0 for record in records]
        selected = self._rng.choices(records, weights=weights, k=count)
        return [_clone_record(record) for record in selected]

    def save(self, path: str | os.PathLike[str], manifest: dict[str, object]) -> None:
        target = Path(path).expanduser().resolve()
        canonical_manifest = _validate_manifest(manifest)
        payload: dict[str, object] = {
            "capacity": self.capacity,
            "hard_fraction": self.hard_fraction,
            "seed": self.seed,
            "normal_seen": self._normal_seen,
            "hard_seen": self._hard_seen,
            "rng_state": self._rng.getstate(),
            "normal": [_record_to_payload(record) for record in self._normal],
            "hard": [_record_to_payload(record) for record in self._hard],
        }
        _atomic_torch_save(target, payload)
        _atomic_json_save(Path(f"{target}.manifest.json"), canonical_manifest)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_manifest: dict[str, object],
    ) -> "VersionedDaggerReplay":
        target = Path(path).expanduser().resolve()
        manifest_path = Path(f"{target}.manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing replay manifest: {manifest_path}")
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("replay manifest is not valid JSON") from error
        if not isinstance(actual, dict):
            raise ValueError("replay manifest must contain a JSON object")
        expected = _validate_manifest(expected_manifest)
        for field in _MANIFEST_FIELDS:
            if actual.get(field) != expected[field]:
                raise ValueError(
                    f"manifest {field} mismatch: expected {expected[field]!r}, got {actual.get(field)!r}"
                )
        if actual != expected:
            raise ValueError("manifest contains unexpected compatibility fields")
        if not target.is_file():
            raise FileNotFoundError(f"missing replay shard: {target}")
        try:
            payload = torch.load(target, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(target, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("replay shard must contain a dictionary")
        try:
            replay = cls(
                capacity=payload["capacity"],
                hard_fraction=payload["hard_fraction"],
                seed=payload["seed"],
            )
            replay._normal = [_payload_to_record(item) for item in payload["normal"]]
            replay._hard = [_payload_to_record(item) for item in payload["hard"]]
            replay._normal_seen = payload["normal_seen"]
            replay._hard_seen = payload["hard_seen"]
            replay._rng.setstate(payload["rng_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay shard fields are invalid") from error
        if len(replay._normal) > replay._normal_capacity:
            raise ValueError("normal replay reservoir exceeds capacity")
        if len(replay._hard) > replay._hard_capacity:
            raise ValueError("hard replay reservoir exceeds capacity")
        if any(record.hard for record in replay._normal) or any(
            not record.hard for record in replay._hard
        ):
            raise ValueError("replay shard contains records in the wrong reservoir")
        return replay


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DaggerRecord",
    "VersionedDaggerReplay",
]
