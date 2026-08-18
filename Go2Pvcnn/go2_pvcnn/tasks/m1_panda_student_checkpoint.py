"""Strict, atomic Student S1 checkpoint and compatibility manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
)


STUDENT_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StudentCheckpointManifest:
    schema_version: int
    asset_sha: str
    teacher_commit: str
    dataset_sha: str
    observation_dim: int
    history_length: int
    action_dim: int
    action_scales: dict[str, float]
    control_dt: float
    dagger_stage: str
    teacher_probability: float
    model_config: dict[str, int]
    loss_weights: dict[str, float]


@dataclass(frozen=True)
class LoadedStudentCheckpoint:
    global_step: int
    manifest: StudentCheckpointManifest


_MANIFEST_FIELDS = tuple(StudentCheckpointManifest.__dataclass_fields__)


def _validate_numeric_mapping(
    value: object, *, label: str, positive: bool
) -> dict[str, int | float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty dictionary")
    normalized: dict[str, int | float] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or (positive and float(item) <= 0.0)
            or (not positive and float(item) < 0.0)
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{label} must contain finite {qualifier} numeric values")
        normalized[name] = item
    return normalized


def _validated_manifest(
    manifest: StudentCheckpointManifest, *, label: str = "manifest"
) -> dict[str, object]:
    if not isinstance(manifest, StudentCheckpointManifest):
        raise TypeError(f"{label} must be a StudentCheckpointManifest")
    payload = asdict(manifest)
    fixed = {
        "schema_version": STUDENT_CHECKPOINT_SCHEMA_VERSION,
        "observation_dim": STUDENT_OBSERVATION_DIM,
        "history_length": STUDENT_HISTORY_LENGTH,
        "action_dim": STUDENT_ACTION_DIM,
        "control_dt": 0.005,
    }
    for field, expected in fixed.items():
        if payload[field] != expected:
            raise ValueError(
                f"{label} {field} mismatch: expected {expected!r}, got {payload[field]!r}"
            )
    for field in ("asset_sha", "teacher_commit", "dataset_sha", "dagger_stage"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{label} {field} must be a non-empty string")
    probability = payload["teacher_probability"]
    if (
        not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ValueError(f"{label} teacher_probability must be finite and in [0,1]")
    payload["action_scales"] = _validate_numeric_mapping(
        payload["action_scales"], label=f"{label} action_scales", positive=True
    )
    model_config = _validate_numeric_mapping(
        payload["model_config"], label=f"{label} model_config", positive=True
    )
    if any(not isinstance(item, int) or isinstance(item, bool) for item in model_config.values()):
        raise ValueError(f"{label} model_config values must be integers")
    payload["model_config"] = model_config
    payload["loss_weights"] = _validate_numeric_mapping(
        payload["loss_weights"], label=f"{label} loss_weights", positive=False
    )
    try:
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON data") from error


def _validate_finite_tree(value: object, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains non-finite tensor values")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _validate_finite_tree(item, label=f"{label}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, label=f"{label}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")


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


def _load_payload(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return payload


def _load_actual_manifest(path: Path) -> tuple[StudentCheckpointManifest, dict[str, object]]:
    manifest_path = Path(f"{path}.manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing checkpoint manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must contain a JSON object")
    for field in _MANIFEST_FIELDS:
        if field not in payload:
            raise ValueError(f"checkpoint manifest missing {field}")
    extra = set(payload) - set(_MANIFEST_FIELDS)
    if extra:
        raise ValueError(f"checkpoint manifest contains unexpected fields: {sorted(extra)}")
    try:
        manifest = StudentCheckpointManifest(**payload)
    except TypeError as error:
        raise ValueError("checkpoint manifest fields are invalid") from error
    return manifest, _validated_manifest(manifest, label="checkpoint manifest")


def _validate_model_state(
    serialized: object, model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    if not isinstance(serialized, dict):
        raise ValueError("model_state_dict must be a dictionary")
    expected = model.state_dict()
    if set(serialized) != set(expected):
        missing = sorted(set(expected) - set(serialized))
        unexpected = sorted(set(serialized) - set(expected))
        raise ValueError(
            f"model_state_dict keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    normalized: dict[str, torch.Tensor] = {}
    for name, target in expected.items():
        value = serialized[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model_state_dict {name} must be a tensor")
        if value.shape != target.shape:
            raise ValueError(
                f"model_state_dict {name} shape mismatch: expected {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"model_state_dict {name} contains non-finite values")
        normalized[name] = value
    return normalized


def save_student_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    manifest: StudentCheckpointManifest,
    *,
    global_step: int,
) -> None:
    """Validate and atomically publish a resumable Student checkpoint."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 0
    ):
        raise ValueError("global_step must be a nonnegative integer")
    canonical = _validated_manifest(manifest)
    model_cfg = getattr(model, "cfg", None)
    if model_cfg is not None and hasattr(model_cfg, "__dataclass_fields__"):
        actual_model_config = json.loads(json.dumps(asdict(model_cfg), sort_keys=True))
        if canonical["model_config"] != actual_model_config:
            raise ValueError(
                "manifest model_config mismatch with the model being saved"
            )
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    _validate_finite_tree(model_state, label="model_state_dict")
    _validate_finite_tree(optimizer_state, label="optimizer_state_dict")
    target = Path(path).expanduser().resolve()
    _atomic_torch_save(
        target,
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "global_step": global_step,
        },
    )
    _atomic_json_save(Path(f"{target}.manifest.json"), canonical)


def load_student_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    expected: StudentCheckpointManifest,
) -> LoadedStudentCheckpoint:
    """Strictly restore a compatible Student checkpoint for inference/resume."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer or None")
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"missing Student checkpoint: {target}")
    actual_manifest, actual = _load_actual_manifest(target)
    wanted = _validated_manifest(expected, label="expected manifest")
    for field in _MANIFEST_FIELDS:
        if actual[field] != wanted[field]:
            raise ValueError(
                f"manifest {field} mismatch: expected {wanted[field]!r}, got {actual[field]!r}"
            )

    payload = _load_payload(target)
    global_step = payload.get("global_step")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 0
    ):
        raise ValueError("checkpoint global_step must be a nonnegative integer")
    model_state = _validate_model_state(payload.get("model_state_dict"), model)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None:
        if not isinstance(optimizer_state, dict):
            raise ValueError(
                "checkpoint optimizer_state_dict must be present for resume"
            )
        _validate_finite_tree(optimizer_state, label="optimizer_state_dict")

    model.load_state_dict(model_state, strict=True)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("optimizer_state_dict is incompatible") from error
    return LoadedStudentCheckpoint(
        global_step=global_step,
        manifest=actual_manifest,
    )


__all__ = [
    "LoadedStudentCheckpoint",
    "STUDENT_CHECKPOINT_SCHEMA_VERSION",
    "StudentCheckpointManifest",
    "load_student_checkpoint",
    "save_student_checkpoint",
]
