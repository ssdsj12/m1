"""Offline distillation of a compact, SHA-pinned fingertip mixture student."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import (
    DISTILLATION_LOSS_CONFIG,
    DISTILLATION_TRAINING_DEFAULTS,
    save_student_artifact,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    FINGER_ORDER, LEFT_REFLECTION, MODEL_INPUT_FIELD_ORDER, MIXTURE_OUTPUT_AXIS_ORDER,
    PHASE_ORDER, PRIOR_DT, MixtureDistribution, StudentArtifactMetadata,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.download import sha256_file
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.ensemble_artifact import (
    PRODUCTION_MEMBER_SEEDS,
    adamw_contract,
    ensemble_manifest_sha256,
    evaluate_ensemble_metrics,
    production_ensemble_approved,
    validate_adamw_state,
    validate_ensemble_artifact,
    validate_model_state,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import (
    FingertipMixtureNet, mixture_log_prob, mixture_nll, temporal_regularizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import verify_aggregate_manifest

DISTILLATION_CONFIG = DISTILLATION_LOSS_CONFIG
_ARRAY_NAMES = frozenset({
    "fingertip_position_palm", "fingertip_velocity_palm", "contact_mask", "phase",
    "future_fingertip_velocity_palm", "source_group", "source_sha256",
})
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
_SYNTHETIC_MEMBER_SEEDS = (1701, 2718)


@dataclass(frozen=True)
class GroupShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    inputs: torch.Tensor
    targets: torch.Tensor
    groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.inputs.dtype != torch.float32 or self.inputs.ndim != 2 or self.inputs.shape[1] != 42:
            raise ValueError("inputs must have shape (N, 42) and float32 dtype")
        if self.targets.dtype != torch.float32 or self.targets.shape != (self.inputs.shape[0], 20, 5, 3):
            raise ValueError("targets must have shape (N, 20, 5, 3) and float32 dtype")
        if len(self.groups) != self.inputs.shape[0] or any(type(group) is not str for group in self.groups):
            raise ValueError("groups must align with inputs")
        if not bool(torch.isfinite(self.inputs).all() and torch.isfinite(self.targets).all()):
            raise ValueError("dataset geometry must be finite")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _ensemble_manifest_sha256(body: dict[str, object]) -> str:
    return ensemble_manifest_sha256(body)


def _lexical_absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if ".." in value.parts:
        raise ValueError(f"unsafe parent traversal in path: {value}")
    return value if value.is_absolute() else Path.cwd() / value


def _walk_safe_directory(path: Path, *, create: bool) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            if not create:
                raise ValueError(f"required directory is missing: {current}") from error
            os.mkdir(current)
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"safe directory chain contains symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"safe directory chain contains non-directory: {current}")


def _ensure_safe_directory(path: Path) -> None:
    _walk_safe_directory(path, create=True)


def _require_safe_directory(path: Path) -> None:
    _walk_safe_directory(path, create=False)


def _require_regular_file(path: Path, *, label: str) -> None:
    _require_safe_directory(path.parent)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")


def _atomic_bytes(path: Path, value: bytes) -> None:
    _ensure_safe_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    _ensure_safe_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _configure_cuda_determinism(device: str | torch.device) -> None:
    if torch.device(device).type != "cuda":
        return
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialized; CUBLAS_WORKSPACE_CONFIG preflight is too late")
    value = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if value not in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG is unsupported for deterministic CUDA distillation")


def _device_fingerprint(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        return {"type": "cpu"}
    if device.type != "cuda":
        raise ValueError(f"unsupported training device type: {device.type}")
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda", "uuid": str(properties.uuid), "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _set_seed(seed: int, device: str | torch.device = "cpu") -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _finite_state(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _resolve_manifest(value: str) -> Path:
    path = _lexical_absolute(value)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and stat.S_ISDIR(metadata.st_mode):
        _require_safe_directory(path)
        path = path / "aggregate_manifest.json"
    if path.name != "aggregate_manifest.json":
        raise ValueError("dataset manifest must be a regular aggregate_manifest.json file")
    _require_regular_file(path, label="dataset aggregate_manifest.json")
    return path


def _phase_one_hot(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8 or values.ndim != 1 or np.any(values >= len(PHASE_ORDER)):
        raise ValueError("phase is invalid")
    return np.eye(len(PHASE_ORDER), dtype=np.float32)[values]


def _load_group_split(manifest_path: Path, split_name: str) -> tuple[GroupShardDataset, dict[str, object]]:
    document = verify_aggregate_manifest(manifest_path.parent)
    split_groups = document.get("split_groups")
    if type(split_groups) is not dict or set(split_groups) != {"train", "validation", "test"}:
        raise ValueError("aggregate split groups are invalid")
    groups_by_split: list[set[str]] = []
    for name in ("train", "validation", "test"):
        values = split_groups[name]
        if type(values) is not list or any(type(group) is not str or not group for group in values):
            raise ValueError("aggregate split groups are invalid")
        groups_by_split.append(set(values))
        if len(groups_by_split[-1]) != len(values):
            raise ValueError("aggregate split groups contain duplicates")
    if any(groups_by_split[left] & groups_by_split[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("aggregate split groups overlap")
    expected_groups = split_groups.get(split_name)
    if type(expected_groups) is not list:
        raise ValueError("aggregate split groups are invalid")
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[str] = []
    for record in document["shards"]:
        if record.get("split") != split_name:
            continue
        with np.load(manifest_path.parent / record["path"], allow_pickle=False) as shard:
            if set(shard.files) != _ARRAY_NAMES:
                raise ValueError("shard keys do not match the frozen geometry contract")
            position, velocity = shard["fingertip_position_palm"], shard["fingertip_velocity_palm"]
            contact, phase = shard["contact_mask"], shard["phase"]
            future, shard_groups = shard["future_fingertip_velocity_palm"], shard["source_group"]
            source_sha = shard["source_sha256"]
        count = phase.shape[0]
        if (
            position.dtype != np.float32 or position.shape != (count, 5, 3)
            or velocity.dtype != np.float32 or velocity.shape != (count, 5, 3)
            or contact.dtype != np.bool_ or contact.shape != (count, 5)
            or future.dtype != np.float32 or future.shape != (count, 20, 5, 3)
            or shard_groups.ndim != 1 or shard_groups.shape[0] != count
            or source_sha.ndim != 1 or source_sha.shape[0] != count
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not np.isfinite(future).all()
        ):
            raise ValueError("shard arrays violate the frozen geometry contract")
        shard_group_values = tuple(str(group) for group in shard_groups.tolist())
        if any(group not in expected_groups for group in shard_group_values):
            raise ValueError("a source group crosses aggregate split boundaries")
        network_input = np.concatenate((
            position.reshape(count, -1), velocity.reshape(count, -1),
            contact.astype(np.float32, copy=False), _phase_one_hot(phase),
        ), axis=1)
        inputs.append(network_input.astype(np.float32, copy=False))
        targets.append(future)
        groups.extend(shard_group_values)
    if not inputs:
        raise ValueError(f"aggregate has no {split_name} samples")
    return GroupShardDataset(
        torch.from_numpy(np.concatenate(inputs, axis=0)),
        torch.from_numpy(np.concatenate(targets, axis=0)), tuple(groups),
    ), document


def sample_mixture(
    distribution: MixtureDistribution, *, samples_per_state: int, seed: int
) -> torch.Tensor:
    """Draw canonical fixed-seed samples, preserving no component index correspondence."""

    if distribution.logits.ndim != 2:
        raise ValueError("teacher distribution must have a batch dimension")
    if type(samples_per_state) is not int or samples_per_state <= 0 or type(seed) is not int:
        raise ValueError("samples_per_state and seed must be valid integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = distribution.logits.detach().cpu().softmax(-1)
    component = torch.multinomial(probabilities, samples_per_state, replacement=True, generator=generator)
    batch, components = component.shape[0], distribution.logits.shape[-1]
    mean = distribution.mean.detach().cpu().unsqueeze(1).expand(batch, samples_per_state, components, 20, 5, 3)
    log_std = distribution.log_std.detach().cpu().unsqueeze(1).expand_as(mean)
    index = component[:, :, None, None, None, None].expand(batch, samples_per_state, 1, 20, 5, 3)
    selected_mean = mean.gather(2, index).squeeze(2)
    selected_log_std = log_std.gather(2, index).squeeze(2)
    return selected_mean + selected_log_std.exp() * torch.randn(selected_mean.shape, generator=generator, dtype=torch.float32)


def distribution_distillation_loss(student: MixtureDistribution, teacher_samples: torch.Tensor) -> torch.Tensor:
    """Score fixed teacher samples under the student distribution, never aligning components."""

    if not isinstance(teacher_samples, torch.Tensor) or teacher_samples.dtype != torch.float32:
        raise TypeError("teacher_samples must be a float32 tensor")
    if student.logits.ndim != 2 or teacher_samples.ndim != 5:
        raise ValueError("teacher_samples must have shape (batch, samples, 20, 5, 3)")
    batch, samples = teacher_samples.shape[:2]
    if teacher_samples.shape != (student.logits.shape[0], samples, 20, 5, 3) or samples <= 0:
        raise ValueError("teacher_samples must have shape (batch, samples, 20, 5, 3)")
    if teacher_samples.device != student.mean.device or not torch.isfinite(teacher_samples).all().item():
        raise ValueError("teacher_samples must be finite and on the student device")
    components = student.logits.shape[-1]
    expanded = MixtureDistribution(
        logits=student.logits.unsqueeze(1).expand(batch, samples, components),
        mean=student.mean.unsqueeze(1).expand(batch, samples, components, 20, 5, 3),
        log_std=student.log_std.unsqueeze(1).expand(batch, samples, components, 20, 5, 3),
    )
    return -mixture_log_prob(expanded, teacher_samples).mean()


def student_distillation_objective(student: MixtureDistribution, target: torch.Tensor, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label, sampled = mixture_nll(student, target), distribution_distillation_loss(student, samples)
    acceleration, jerk = temporal_regularizer(student, dt=PRIOR_DT)
    total = DISTILLATION_CONFIG["label_nll_weight"] * label + DISTILLATION_CONFIG["teacher_sample_weight"] * sampled + DISTILLATION_CONFIG["acceleration_weight"] * acceleration + DISTILLATION_CONFIG["jerk_weight"] * jerk
    return total, label, sampled, acceleration, jerk


@dataclass(frozen=True)
class _Ensemble:
    models: tuple[FingertipMixtureNet, ...]
    manifest_sha256: str
    dataset_sha256: str
    hidden: tuple[int, ...]
    teacher_seed: int
    synthetic: bool
    reported_metrics: dict[str, float] = field(default_factory=dict)


def _regular(path: Path, label: str) -> None:
    _require_regular_file(path, label=label)


def _load_ensemble(
    path: str | Path, *, expected_dataset_sha: str, allow_synthetic: bool,
    nonproduction_synthetic: bool = False,
) -> _Ensemble:
    root = _lexical_absolute(path)
    _require_safe_directory(root)
    expected_seeds = _SYNTHETIC_MEMBER_SEEDS if allow_synthetic else PRODUCTION_MEMBER_SEEDS
    validated = validate_ensemble_artifact(
        root, expected_dataset_sha256=expected_dataset_sha,
        expected_member_seeds=expected_seeds, expected_synthetic=allow_synthetic,
        expected_nonproduction_synthetic=nonproduction_synthetic,
    )
    manifest = validated.manifest
    if not allow_synthetic and manifest["production_deployable"] is not True:
        raise ValueError("ensemble is not deployable")
    return _Ensemble(
        validated.models, str(manifest["ensemble_manifest_sha256"]), expected_dataset_sha,
        tuple(manifest["hidden"]), expected_seeds[0], bool(manifest["synthetic_smoke"]),
        dict(manifest["metrics"]),
    )


def _validate_recomputed_ensemble_metrics(
    models: Sequence[FingertipMixtureNet], dataset: GroupShardDataset,
    reported_metrics: Mapping[str, object], *, batch_size: int, device: torch.device,
    require_production: bool = False,
) -> dict[str, float]:
    recomputed = evaluate_ensemble_metrics(
        tuple(models), dataset.inputs, dataset.targets, batch_size=batch_size, device=device,
    )
    if set(recomputed) != set(reported_metrics):
        raise ValueError("reported held-out ensemble metric schema differs from recomputed metrics")
    for key, actual in recomputed.items():
        reported = reported_metrics[key]
        if (
            type(reported) not in (int, float) or type(reported) is bool
            or not math.isclose(actual, float(reported), rel_tol=1e-5, abs_tol=1e-7)
        ):
            raise ValueError(f"reported held-out ensemble metric {key} differs from recomputed inference")
    if require_production and not production_ensemble_approved(recomputed):
        raise ValueError("recomputed held-out ensemble metrics do not pass the production gate")
    return recomputed


def _ensemble_outputs(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor) -> tuple[MixtureDistribution, ...]:
    distributions = [model(inputs) for model in models]
    if not distributions:
        raise ValueError("ensemble must contain a member")
    return tuple(distributions)


def _ensemble_log_prob(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact member-average density while each member remains a frozen 4-mixture."""

    values = torch.stack([mixture_log_prob(distribution, target) for distribution in _ensemble_outputs(models, inputs)], dim=1)
    return torch.logsumexp(values, dim=1) - math.log(values.shape[1])


def _ensemble_mean(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor) -> torch.Tensor:
    distributions = _ensemble_outputs(models, inputs)
    count = len(distributions)
    means = torch.stack([distribution.mean for distribution in distributions], dim=1)
    weights = torch.stack([distribution.logits.softmax(-1) for distribution in distributions], dim=1) / count
    return (weights[..., None, None, None] * means).sum(dim=(1, 2))


def _sample_ensemble(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor, *, samples_per_state: int, seed: int) -> torch.Tensor:
    distributions = _ensemble_outputs(models, inputs)
    if type(samples_per_state) is not int or samples_per_state <= 0 or type(seed) is not int:
        raise ValueError("samples_per_state and seed must be valid integers")
    probabilities = torch.stack(
        [distribution.logits.detach().cpu().softmax(-1) for distribution in distributions], dim=1
    ) / len(distributions)
    means = torch.stack([distribution.mean.detach().cpu() for distribution in distributions], dim=1)
    log_stds = torch.stack([distribution.log_std.detach().cpu() for distribution in distributions], dim=1)
    batch = probabilities.shape[0]
    flat_probabilities = probabilities.reshape(batch, -1)
    flat_means = means.reshape(batch, -1, 20, 5, 3)
    flat_log_stds = log_stds.reshape_as(flat_means)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.multinomial(flat_probabilities, samples_per_state, replacement=True, generator=generator)
    index = selected[:, :, None, None, None, None].expand(batch, samples_per_state, 1, 20, 5, 3)
    means_by_sample = flat_means.unsqueeze(1).expand(batch, samples_per_state, flat_means.shape[1], 20, 5, 3)
    stds_by_sample = flat_log_stds.unsqueeze(1).expand_as(means_by_sample)
    return means_by_sample.gather(2, index).squeeze(2) + stds_by_sample.gather(2, index).squeeze(2).exp() * torch.randn((batch, samples_per_state, 20, 5, 3), generator=generator, dtype=torch.float32)


def _ensemble_component_probabilities(
    models: Sequence[FingertipMixtureNet], inputs: torch.Tensor
) -> torch.Tensor:
    """Return the exact flattened weights of the equal-member ensemble density."""

    distributions = _ensemble_outputs(models, inputs)
    return torch.cat(
        [distribution.logits.softmax(-1) / len(distributions) for distribution in distributions],
        dim=-1,
    )


@dataclass(frozen=True)
class _TeacherSampleStore:
    root: Path
    samples: np.memmap
    manifest: dict[str, object]


class _DistillationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Join resident geometry tensors to mmap-backed teacher samples per batch item."""

    def __init__(self, dataset: GroupShardDataset, store: _TeacherSampleStore) -> None:
        if len(dataset) != store.samples.shape[0]:
            raise ValueError("teacher sample store does not align with training dataset")
        self.dataset = dataset
        self.store = store

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        network_input, target = self.dataset[index]
        # Copy one sample record because the mmap is deliberately read-only.
        samples = torch.from_numpy(np.array(self.store.samples[index], copy=True))
        return network_input, target, samples


def _teacher_store_identity(
    *, sample_count: int, samples_per_state: int, seed: int,
    dataset_sha256: str, ensemble_sha256: str, chunk_size: int,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "sampling_algorithm": "equal-member-mixture-v1",
        "sample_count": sample_count,
        "samples_per_state": samples_per_state,
        "seed": seed,
        "dataset_aggregate_sha256": dataset_sha256,
        "teacher_ensemble_manifest_sha256": ensemble_sha256,
        "chunk_size": chunk_size,
        "shape": [sample_count, samples_per_state, 20, 5, 3],
        "dtype": "float32",
    }


def _load_teacher_sample_store(root: Path, identity: dict[str, object]) -> _TeacherSampleStore:
    _require_safe_directory(root)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise ValueError("teacher sample store must be a safe regular directory") from error
    manifest_descriptor = samples_descriptor = None
    try:
        if set(os.listdir(root_descriptor)) != {"manifest.json", "samples.npy"}:
            raise ValueError("teacher sample store must contain exactly manifest.json and samples.npy")
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            manifest_descriptor = os.open("manifest.json", file_flags, dir_fd=root_descriptor)
            samples_descriptor = os.open("samples.npy", file_flags, dir_fd=root_descriptor)
        except OSError as error:
            raise ValueError("teacher sample store entries must be regular non-symlink files") from error
        for descriptor, label in (
            (manifest_descriptor, "teacher sample manifest"),
            (samples_descriptor, "teacher sample array"),
        ):
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{label} must be a regular file")
        with os.fdopen(manifest_descriptor, "rb", closefd=False) as handle:
            manifest_bytes = handle.read()
        try:
            document = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("teacher sample manifest is invalid") from error
        if type(document) is not dict or set(document) != set(identity) | {"samples_sha256", "manifest_sha256"}:
            raise ValueError("teacher sample manifest schema is invalid")
        if {key: document.get(key) for key in identity} != identity:
            raise ValueError("teacher sample store identity mismatch")
        body = dict(document)
        declared_manifest = body.pop("manifest_sha256", None)
        if declared_manifest != sha256(_canonical_json(body)).hexdigest():
            raise ValueError("teacher sample manifest SHA mismatch")
        digest = sha256()
        os.lseek(samples_descriptor, 0, os.SEEK_SET)
        while chunk := os.read(samples_descriptor, 1024 * 1024):
            digest.update(chunk)
        if document.get("samples_sha256") != digest.hexdigest():
            raise ValueError("teacher sample array SHA mismatch")
        samples = np.load(
            f"/proc/self/fd/{samples_descriptor}", mmap_mode="r", allow_pickle=False,
        )
    finally:
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        if samples_descriptor is not None:
            os.close(samples_descriptor)
        os.close(root_descriptor)
    if not isinstance(samples, np.memmap) or list(samples.shape) != identity["shape"] or samples.dtype != np.float32:
        raise ValueError("teacher sample array violates frozen shape/dtype")
    for start in range(0, samples.shape[0], 128):
        if not np.isfinite(samples[start:start + 128]).all():
            raise ValueError("teacher sample array must be finite")
    return _TeacherSampleStore(root, samples, document)


def _teacher_staging_paths(destination: Path) -> tuple[Path, Path]:
    return (
        destination.parent / f".{destination.name}.owner-v1.json",
        destination.parent / f".{destination.name}.stage-v1",
    )


def _teacher_owner_document(
    destination: Path, identity: dict[str, object], *, completed_count: int,
    chunks: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    identity_sha = sha256(_canonical_json(identity)).hexdigest()
    body: dict[str, object] = {
        "format_version": 1,
        "destination": str(destination),
        "stage_name": f".{destination.name}.stage-v1",
        "identity": identity,
        "identity_sha256": identity_sha,
        "completed_count": completed_count,
        "chunks": [dict(chunk) for chunk in chunks],
    }
    return {**body, "progress_sha256": sha256(_canonical_json(body)).hexdigest()}


def _load_teacher_owner(
    owner_path: Path, destination: Path, identity: dict[str, object],
) -> dict[str, object]:
    _require_regular_file(owner_path, label="teacher sample staging owner")
    try:
        document = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher sample staging owner is invalid") from error
    expected_fields = {
        "format_version", "destination", "stage_name", "identity", "identity_sha256",
        "completed_count", "chunks", "progress_sha256",
    }
    if type(document) is not dict or set(document) != expected_fields:
        raise ValueError("teacher sample staging owner schema is invalid")
    body = dict(document)
    declared_progress = body.pop("progress_sha256")
    if declared_progress != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("teacher sample staging owner progress SHA mismatch")
    if (
        document["format_version"] != 1
        or document["destination"] != str(destination)
        or document["stage_name"] != f".{destination.name}.stage-v1"
        or document["identity"] != identity
        or document["identity_sha256"] != sha256(_canonical_json(identity)).hexdigest()
    ):
        raise ValueError("teacher sample staging owner identity mismatch")
    completed = document["completed_count"]
    if type(completed) is not int or not 0 <= completed <= int(identity["sample_count"]):
        raise ValueError("teacher sample staging progress is invalid")
    chunks = document["chunks"]
    if type(chunks) is not list:
        raise ValueError("teacher sample staging chunk progress is invalid")
    cursor = 0
    for chunk in chunks:
        expected_stop = min(cursor + int(identity["chunk_size"]), int(identity["sample_count"]))
        if (
            type(chunk) is not dict or set(chunk) != {"start", "stop", "sha256"}
            or chunk["start"] != cursor or chunk["stop"] != expected_stop
            or type(chunk["sha256"]) is not str or len(chunk["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in chunk["sha256"])
        ):
            raise ValueError("teacher sample staging chunk progress is invalid")
        cursor = expected_stop
    if cursor != completed:
        raise ValueError("teacher sample staging chunk progress does not match completed count")
    return document


def _write_teacher_owner(
    owner_path: Path, destination: Path, identity: dict[str, object], *, completed_count: int,
    chunks: Sequence[dict[str, object]] = (),
) -> None:
    if os.path.lexists(owner_path):
        _load_teacher_owner(owner_path, destination, identity)
    _atomic_bytes(
        owner_path,
        _canonical_json(_teacher_owner_document(
            destination, identity, completed_count=completed_count, chunks=chunks,
        )),
    )


def _remove_owned_teacher_staging(
    owner_path: Path, stage: Path, destination: Path, identity: dict[str, object],
    *, stage_may_be_missing: bool = False,
) -> None:
    """Delete only the exact, regular staging tree named by a valid owner checkpoint."""

    _load_teacher_owner(owner_path, destination, identity)
    if os.path.lexists(stage):
        _require_safe_directory(stage)
        shutil.rmtree(stage)
    elif not stage_may_be_missing:
        raise ValueError("owned teacher sample staging directory is missing")
    _require_regular_file(owner_path, label="teacher sample staging owner")
    owner_path.unlink()


def _open_partial_teacher_samples(
    stage: Path, identity: dict[str, object], *, completed_count: int,
    chunks: Sequence[dict[str, object]],
) -> np.memmap:
    sample_path = stage / "samples.npy"
    if os.path.lexists(sample_path):
        _require_regular_file(sample_path, label="partial teacher sample array")
        output = np.load(sample_path, mmap_mode="r+", allow_pickle=False)
        if (
            not isinstance(output, np.memmap)
            or list(output.shape) != identity["shape"]
            or output.dtype != np.float32
        ):
            raise ValueError("partial teacher sample array violates frozen shape/dtype")
        for chunk in chunks:
            start, stop = int(chunk["start"]), int(chunk["stop"])
            values = np.asarray(output[start:stop])
            if not np.isfinite(values).all():
                raise ValueError("completed teacher samples must be finite")
            if sha256(values.tobytes(order="C")).hexdigest() != chunk["sha256"]:
                raise ValueError("completed teacher sample chunk SHA mismatch")
        return output
    if completed_count != 0:
        raise ValueError("partial teacher sample array is missing after recorded progress")
    return np.lib.format.open_memmap(
        sample_path, mode="w+", dtype=np.float32, shape=tuple(identity["shape"]),
    )


def _validate_teacher_stage_files(stage: Path) -> None:
    _require_safe_directory(stage)
    entries = {entry.name: entry for entry in stage.iterdir()}
    if set(entries) != {"manifest.json", "samples.npy"}:
        raise ValueError("teacher sample staging must contain exactly manifest.json and samples.npy")
    for name, path in entries.items():
        _require_regular_file(path, label=f"teacher sample staging {name}")


def _prepare_teacher_sample_store(
    root: str | Path,
    models: Sequence[FingertipMixtureNet],
    dataset: GroupShardDataset,
    *,
    samples_per_state: int,
    seed: int,
    dataset_sha256: str,
    ensemble_sha256: str,
    chunk_size: int = 128,
) -> _TeacherSampleStore:
    """Create or verify deterministic samples without materializing the full array in RAM."""

    destination = _lexical_absolute(root)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("teacher sample chunk size must be positive")
    identity = _teacher_store_identity(
        sample_count=len(dataset), samples_per_state=samples_per_state, seed=seed,
        dataset_sha256=dataset_sha256, ensemble_sha256=ensemble_sha256,
        chunk_size=chunk_size,
    )
    if os.path.lexists(destination):
        if stat.S_ISLNK(os.lstat(destination).st_mode):
            raise ValueError("teacher sample store must not be a symlink")
        store = _load_teacher_sample_store(destination, identity)
        owner_path, stage = _teacher_staging_paths(destination)
        if os.path.lexists(owner_path):
            _remove_owned_teacher_staging(
                owner_path, stage, destination, identity, stage_may_be_missing=True,
            )
        elif os.path.lexists(stage):
            raise ValueError("unowned teacher sample staging directory must not be removed")
        return store
    _ensure_safe_directory(destination.parent)
    owner_path, stage = _teacher_staging_paths(destination)
    if os.path.lexists(owner_path):
        owner = _load_teacher_owner(owner_path, destination, identity)
        completed_count = int(owner["completed_count"])
        chunks = list(owner["chunks"])
        if os.path.lexists(stage):
            _require_safe_directory(stage)
        elif completed_count == 0:
            os.mkdir(stage)
        else:
            raise ValueError("owned teacher sample staging directory is missing")
    else:
        if os.path.lexists(stage):
            raise ValueError("unowned teacher sample staging directory must not be removed")
        _write_teacher_owner(owner_path, destination, identity, completed_count=0)
        os.mkdir(stage)
        completed_count = 0
        chunks = []
    try:
        sample_path = stage / "samples.npy"
        output = _open_partial_teacher_samples(
            stage, identity, completed_count=completed_count, chunks=chunks,
        )
        with torch.no_grad():
            for start in range(completed_count, len(dataset), chunk_size):
                stop = min(start + chunk_size, len(dataset))
                output[start:stop] = _sample_ensemble(
                    models, dataset.inputs[start:stop], samples_per_state=samples_per_state,
                    seed=seed + start,
                ).numpy()
                output.flush()
                with sample_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                chunks.append({
                    "start": start,
                    "stop": stop,
                    "sha256": sha256(np.asarray(output[start:stop]).tobytes(order="C")).hexdigest(),
                })
                _write_teacher_owner(
                    owner_path, destination, identity, completed_count=stop, chunks=chunks,
                )
        output.flush()
        del output
        with sample_path.open("rb") as handle:
            os.fsync(handle.fileno())
        body = {**identity, "samples_sha256": sha256_file(sample_path)}
        manifest = {**body, "manifest_sha256": sha256(_canonical_json(body)).hexdigest()}
        _atomic_bytes(stage / "manifest.json", _canonical_json(manifest))
        _load_teacher_sample_store(stage, identity)
        _validate_teacher_stage_files(stage)
        os.replace(stage, destination)
        store = _load_teacher_sample_store(destination, identity)
        _remove_owned_teacher_staging(
            owner_path, stage, destination, identity, stage_may_be_missing=True,
        )
        return store
    except Exception:
        if not os.path.lexists(destination):
            _remove_owned_teacher_staging(owner_path, stage, destination, identity)
        raise


def _predictive_mean(distribution: MixtureDistribution) -> torch.Tensor:
    return (distribution.logits.softmax(-1)[..., None, None, None] * distribution.mean).sum(dim=1)


def _metrics(
    model: FingertipMixtureNet, ensemble: _Ensemble, dataset: GroupShardDataset,
    *, device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    first_sq = endpoint_sq = teacher_endpoint_sq = zero_first_sq = zero_endpoint_sq = student_sum = teacher_sum = 0.0
    samples = 0
    model.to(device).eval()
    for member in ensemble.models:
        member.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            inputs = dataset.inputs[start:start + 128].to(device)
            target = dataset.targets[start:start + 128].to(device)
            student = model(inputs)
            teacher_log_prob = _ensemble_log_prob(ensemble.models, inputs, target)
            student_sum += float((-mixture_log_prob(student, target)).sum().item())
            teacher_sum += float((-teacher_log_prob).sum().item())
            mean = _predictive_mean(student)
            first_sq += float((mean[:, 0] - target[:, 0]).square().sum().item())
            zero_first_sq += float(target[:, 0].square().sum().item())
            endpoint = mean.sum(dim=1) * PRIOR_DT
            teacher_endpoint_mean = _ensemble_mean(ensemble.models, inputs).sum(dim=1) * PRIOR_DT
            target_endpoint = target.sum(dim=1) * PRIOR_DT
            endpoint_sq += float((endpoint - target_endpoint).square().sum().item())
            teacher_endpoint_sq += float((teacher_endpoint_mean - target_endpoint).square().sum().item())
            zero_endpoint_sq += float(target_endpoint.square().sum().item())
            samples += target.shape[0]
    if samples == 0 or zero_first_sq <= 0.0 or zero_endpoint_sq <= 0.0:
        raise ValueError("evaluation set cannot produce zero baselines")
    dimensions = samples * 20 * 5 * 3
    first_count = samples * 5 * 3
    student_nll, teacher_nll = student_sum / samples, teacher_sum / samples
    endpoint_rmse = math.sqrt(endpoint_sq / first_count)
    teacher_endpoint_rmse = math.sqrt(teacher_endpoint_sq / first_count)
    zero_endpoint = math.sqrt(zero_endpoint_sq / first_count)
    first_rmse, zero_first = math.sqrt(first_sq / first_count), math.sqrt(zero_first_sq / first_count)
    return {
        "student_nll": student_nll,
        "teacher_nll": teacher_nll,
        "nll_delta_per_dim": (student_nll - teacher_nll) / 300.0,
        "first_step_velocity_rmse": first_rmse,
        "first_step_zero_rmse": zero_first,
        "first_step_improvement": (zero_first - first_rmse) / zero_first,
        "endpoint_rmse": endpoint_rmse,
        "teacher_endpoint_rmse": teacher_endpoint_rmse,
        "endpoint_zero_rmse": zero_endpoint,
        "endpoint_improvement": (zero_endpoint - endpoint_rmse) / zero_endpoint,
    }


def _latency(model: FingertipMixtureNet) -> dict[str, object]:
    model.eval()
    inputs = torch.zeros(1, 42, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(100):
            model(inputs)
        timings: list[int] = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            model(inputs)
            timings.append(time.perf_counter_ns() - start)
    return {"warmups": 100, "measurements": 1000, "p99_ms": float(np.quantile(np.asarray(timings, dtype=np.float64), 0.99, method="higher") / 1_000_000.0)}


def _git_commit() -> str:
    completed = subprocess.run(("git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"), check=True, capture_output=True, text=True)
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve code commit")
    return value


def _canonical_source_bytes(path: Path) -> bytes:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _distillation_source_sha256() -> dict[str, str]:
    package = Path(__file__).parents[1] / "go2_pvcnn" / "control" / "m1_bimanual_coordination" / "expert_fingertip_prior"
    paths = {
        "trainer": Path(__file__),
        "model": package / "model.py",
        "contracts": package / "contracts.py",
        "artifact": package / "artifact.py",
    }
    return {name: sha256(_canonical_source_bytes(path)).hexdigest() for name, path in paths.items()}


def _identity_sha256(identity: dict[str, object]) -> str:
    return sha256(_canonical_json(identity)).hexdigest()


def _distillation_identity(
    *, aggregate_sha: str, ensemble_sha: str, hidden: tuple[int, ...], epochs: int,
    batch_size: int, learning_rate: float, samples_per_state: int, seed: int,
    device: torch.device, synthetic_smoke: bool,
) -> dict[str, object]:
    source = _distillation_source_sha256()
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None
    if device.type == "cuda" and cublas not in {":4096:8", ":16:8"}:
        raise ValueError("CUDA distillation identity requires a supported CUBLAS_WORKSPACE_CONFIG")
    return {
        "format_version": 1,
        "dataset_aggregate_sha256": aggregate_sha,
        "teacher_ensemble_manifest_sha256": ensemble_sha,
        "seed": seed,
        "samples_per_state": samples_per_state,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden": list(hidden),
        "optimizer": adamw_contract(learning_rate),
        "distillation_loss": DISTILLATION_CONFIG,
        "software": {
            "torch_version": str(torch.__version__),
            "torch_cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
            "numpy_version": str(np.__version__),
        },
        "device": _device_fingerprint(device),
        "cublas_workspace_config": cublas,
        "training_semantics": {
            "source_sha256": source,
            "aggregate_sha256": sha256(_canonical_json(source)).hexdigest(),
        },
        "synthetic_smoke": synthetic_smoke,
    }


def _read_student_progress(path: Path) -> dict[str, object]:
    progress = path / "progress.json"
    _regular(progress, "student resume progress")
    document = json.loads(progress.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise ValueError("student resume progress is invalid")
    if set(document) != {
        "format_version", "identity", "identity_sha256", "completed_epochs", "checkpoint",
        "progress_sha256",
    } or document.get("format_version") != 1:
        raise ValueError("student resume progress schema is invalid")
    body = dict(document)
    declared = body.pop("progress_sha256", None)
    if declared != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("student resume progress SHA mismatch")
    return document


class _StudentResumeWorkspace:
    """Strict epoch-boundary state, isolated by the requested output path."""

    def __init__(self, path: str | Path, identity: dict[str, object]) -> None:
        self.path = _lexical_absolute(path)
        self.identity = identity
        self.identity_sha256 = _identity_sha256(identity)
        _ensure_safe_directory(self.path.parent)
        if os.path.lexists(path):
            _require_safe_directory(self.path)
            document = _read_student_progress(self.path)
            if document.get("identity") != identity or document.get("identity_sha256") != self.identity_sha256:
                raise ValueError("student resume identity does not match requested distillation")
            self.document = document
        else:
            body = {
                "format_version": 1,
                "identity": identity,
                "identity_sha256": self.identity_sha256,
                "completed_epochs": 0,
                "checkpoint": None,
            }
            document = {**body, "progress_sha256": sha256(_canonical_json(body)).hexdigest()}
            stage = Path(tempfile.mkdtemp(prefix=f".{self.path.name}.stage-", dir=self.path.parent))
            try:
                _ensure_safe_directory(stage / "checkpoints")
                _atomic_bytes(stage / "progress.json", _canonical_json(document))
                os.replace(stage, self.path)
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            self.document = document
        _ensure_safe_directory(self.path / "checkpoints")
        self._validate()

    def _validate(self) -> None:
        epochs = self.identity["epochs"]
        completed = self.document.get("completed_epochs")
        if type(completed) is not int or type(epochs) is not int or not 0 <= completed <= epochs:
            raise ValueError("student resume completed epoch is invalid")
        record = self.document.get("checkpoint")
        if completed == 0:
            if record is not None:
                raise ValueError("empty student resume has a checkpoint")
            return
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ValueError("student resume checkpoint record is invalid")
        name = record["path"]
        if name not in {"checkpoints/student-0.pt", "checkpoints/student-1.pt"}:
            raise ValueError("student resume checkpoint path is invalid")
        checkpoint = self.path / name
        _regular(checkpoint, "student resume checkpoint")
        if sha256_file(checkpoint) != record["sha256"]:
            raise ValueError("student resume checkpoint SHA mismatch")

    def load(self, *, hidden: tuple[int, ...], steps_per_epoch: int) -> dict[str, object] | None:
        if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
            raise ValueError("student resume steps per epoch are invalid")
        if self.document["completed_epochs"] == 0:
            return None
        checkpoint = self.path / self.document["checkpoint"]["path"]
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise ValueError("student resume checkpoint could not be safely loaded") from error
        if (
            type(state) is not dict
            or set(state) != {
                "format_version", "identity_sha256", "epoch", "hidden", "model_state",
                "optimizer_state",
            }
            or state.get("format_version") != 1
            or state.get("identity_sha256") != self.identity_sha256
            or state.get("epoch") != self.document["completed_epochs"]
            or state.get("hidden") != hidden
            or not isinstance(state.get("model_state"), dict)
            or not isinstance(state.get("optimizer_state"), dict)
            or not _finite_state(state)
        ):
            raise ValueError("student resume checkpoint state is invalid")
        model = FingertipMixtureNet(hidden=hidden)
        validate_model_state(model, state["model_state"], label="student resume checkpoint")
        validate_adamw_state(
            model, state["optimizer_state"], contract=self.identity.get("optimizer"),
            expected_step=int(state["epoch"]) * steps_per_epoch,
            label="student resume checkpoint",
        )
        return state

    def commit_epoch(self, state: dict[str, object]) -> None:
        epoch = state.get("epoch")
        if type(epoch) is not int or epoch != self.document["completed_epochs"] + 1:
            raise ValueError("student resume epochs must be committed in sequence")
        current = self.document.get("checkpoint")
        slot = 1 if type(current) is dict and current.get("path") == "checkpoints/student-0.pt" else 0
        checkpoint = self.path / "checkpoints" / f"student-{slot}.pt"
        _atomic_torch_save(checkpoint, state)
        next_document = copy.deepcopy(self.document)
        next_document.update({
            "completed_epochs": epoch,
            "checkpoint": {
                "path": checkpoint.relative_to(self.path).as_posix(),
                "sha256": sha256_file(checkpoint),
            },
        })
        body = dict(next_document)
        body.pop("progress_sha256", None)
        next_document["progress_sha256"] = sha256(_canonical_json(body)).hexdigest()
        _atomic_bytes(self.path / "progress.json", _canonical_json(next_document))
        self.document = next_document

    def remove_after_publish(self) -> None:
        _require_safe_directory(self.path)
        current = _read_student_progress(self.path)
        if current.get("identity_sha256") != self.identity_sha256 or current.get("identity") != self.identity:
            raise ValueError("refusing to remove changed student resume workspace")
        suffix = ".resume-v1"
        output_name = self.path.name[1 : -len(suffix)] if self.path.name.startswith(".") else ""
        if (
            not output_name or self.path.with_name(output_name).with_name(f".{output_name}{suffix}") != self.path
        ):
            raise ValueError("refusing to remove unexpected student resume workspace")
        shutil.rmtree(self.path)


def _train_student(
    train: GroupShardDataset,
    teacher_store: _TeacherSampleStore,
    *, hidden: tuple[int, ...], epochs: int, batch_size: int, learning_rate: float,
    seed: int, device: torch.device, workspace: _StudentResumeWorkspace,
    interrupt_after_epoch: int | None = None,
) -> FingertipMixtureNet:
    _set_seed(seed, device)
    model = FingertipMixtureNet(hidden=hidden).to(device)
    optimizer_contract = adamw_contract(learning_rate)
    if workspace.identity.get("optimizer") != optimizer_contract:
        raise ValueError("student workspace AdamW identity does not match requested optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_contract["learning_rate"],
        betas=tuple(optimizer_contract["betas"]), eps=optimizer_contract["eps"],
        weight_decay=optimizer_contract["weight_decay"], amsgrad=optimizer_contract["amsgrad"],
        maximize=optimizer_contract["maximize"], foreach=optimizer_contract["foreach"],
        capturable=optimizer_contract["capturable"], differentiable=optimizer_contract["differentiable"],
        fused=optimizer_contract["fused"],
    )
    state = workspace.load(hidden=hidden, steps_per_epoch=math.ceil(len(train) / batch_size))
    start_epoch = 0
    if state is not None:
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"])
    dataset = _DistillationDataset(train, teacher_store)
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        model.train()
        for inputs, target, samples in loader:
            inputs, target, samples = inputs.to(device), target.to(device), samples.to(device)
            loss, _, _, _, _ = student_distillation_objective(model(inputs), target, samples)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("student distillation loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        state = {
            "format_version": 1,
            "identity_sha256": workspace.identity_sha256,
            "epoch": epoch + 1,
            "hidden": hidden,
            "model_state": {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
        }
        if not _finite_state(state):
            raise FloatingPointError("student resume state is non-finite")
        workspace.commit_epoch(state)
        if interrupt_after_epoch == epoch + 1:
            raise RuntimeError("test interruption")
    model.eval()
    return model


def _synthetic_ensemble(stage: Path, *, epochs: int) -> tuple[Path, Path]:
    teacher = stage / "synthetic-teacher"
    command = [sys.executable, str(Path(__file__).with_name("m1_dual_panda_o6_train_fingertip_expert.py")), "--synthetic-smoke", "--output-dir", str(teacher), "--epochs", str(epochs), "--hidden", "16,16", "--member-seeds", "1701,2718"]
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})
    return teacher / "synthetic-shards" / "aggregate_manifest.json", teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline compact fingertip-prior distillation; no network or downloads.")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--ensemble-dir")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DISTILLATION_TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DISTILLATION_TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DISTILLATION_TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument("--samples-per-state", type=int, default=DISTILLATION_TRAINING_DEFAULTS["samples_per_state"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--device", default=DISTILLATION_TRAINING_DEFAULTS["device"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.epochs <= 0 or args.batch_size <= 0 or args.samples_per_state <= 0
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0
    ):
        raise ValueError("distillation values are invalid")
    if args.synthetic_smoke and (args.dataset_manifest or args.ensemble_dir):
        raise ValueError("synthetic smoke cannot consume caller-provided inputs")
    if not args.synthetic_smoke and (not args.dataset_manifest or not args.ensemble_dir):
        raise ValueError("real distillation requires dataset manifest and deployable ensemble")
    try:
        hidden = tuple(int(value) for value in args.hidden.split(","))
    except ValueError as error:
        raise ValueError("hidden must be comma-separated positive integer widths") from error
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden widths are invalid")
    _configure_cuda_determinism(args.device)
    device = torch.device(args.device)
    destination = _lexical_absolute(args.output_dir)
    _ensure_safe_directory(destination.parent)
    if os.path.lexists(destination):
        if stat.S_ISLNK(os.lstat(destination).st_mode):
            raise ValueError(f"output directory must not be a symlink: {destination}")
        raise FileExistsError(f"output directory already exists: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.distill-", dir=destination.parent))
    try:
        if args.synthetic_smoke:
            manifest_path, ensemble_dir = _synthetic_ensemble(stage, epochs=min(args.epochs, 2))
        else:
            manifest_path, ensemble_dir = _resolve_manifest(args.dataset_manifest), _lexical_absolute(args.ensemble_dir)
        document = verify_aggregate_manifest(manifest_path.parent)
        train, train_doc = _load_group_split(manifest_path, "train")
        test, test_doc = _load_group_split(manifest_path, "test")
        if train_doc["aggregate_sha256"] != document["aggregate_sha256"] or test_doc["aggregate_sha256"] != document["aggregate_sha256"]:
            raise ValueError("dataset aggregate changed during verified loading")
        nonproduction_synthetic = bool(
            args.synthetic_smoke
            or document.get("verified_inputs", {}).get("nonproduction_synthetic") is True
        )
        ensemble = _load_ensemble(
            ensemble_dir, expected_dataset_sha=document["aggregate_sha256"],
            allow_synthetic=bool(args.synthetic_smoke),
            nonproduction_synthetic=nonproduction_synthetic,
        )
        _validate_recomputed_ensemble_metrics(
            ensemble.models, test, ensemble.reported_metrics,
            batch_size=args.batch_size, device=device,
            require_production=not bool(args.synthetic_smoke),
        )
        for member in ensemble.models:
            member.cpu().eval()
        identity = _distillation_identity(
            aggregate_sha=document["aggregate_sha256"], ensemble_sha=ensemble.manifest_sha256,
            hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=float(args.learning_rate), samples_per_state=args.samples_per_state,
            seed=args.seed, device=device, synthetic_smoke=bool(args.synthetic_smoke),
        )
        workspace = _StudentResumeWorkspace(
            destination.with_name(f".{destination.name}.resume-v1"), identity,
        )
        teacher_store = _prepare_teacher_sample_store(
            workspace.path / "teacher-samples", ensemble.models, train,
            samples_per_state=args.samples_per_state, seed=args.seed,
            dataset_sha256=document["aggregate_sha256"], ensemble_sha256=ensemble.manifest_sha256,
        )
        model = _train_student(
            train, teacher_store, hidden=hidden, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
            device=device, workspace=workspace,
        )
        metrics = _metrics(model, ensemble, test, device=device)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("student metrics are non-finite")
        repeat_workspace = _StudentResumeWorkspace(stage / ".repeat.resume-v1", identity)
        repeated_model = _train_student(
            train, teacher_store, hidden=hidden, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
            device=device, workspace=repeat_workspace,
        )
        repeated_metrics = _metrics(repeated_model, ensemble, test, device=device)
        if metrics != repeated_metrics or any(
            not torch.equal(value.detach().cpu(), repeated_model.state_dict()[key].detach().cpu())
            for key, value in model.state_dict().items()
        ):
            raise RuntimeError("independent deterministic student identity mismatch")
        cpu_model = FingertipMixtureNet(hidden=hidden)
        cpu_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
        cpu_model.eval()
        latency = _latency(cpu_model)
        production_approved = (
            not args.synthetic_smoke and not ensemble.synthetic
            and metrics["nll_delta_per_dim"] <= 0.05
            and metrics["endpoint_rmse"] <= 1.05 * metrics["teacher_endpoint_rmse"]
            and metrics["first_step_improvement"] >= 0.10
            and metrics["endpoint_improvement"] >= 0.10
            and latency["p99_ms"] < 2.0
        )
        metrics = {**metrics, "production_approved": production_approved, "deterministic_repeat_verified": True}
        training_metadata = {
            "samples_per_state": args.samples_per_state,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": float(args.learning_rate),
            "device": identity["device"],
            "software": identity["software"],
            "cublas_workspace_config": identity["cublas_workspace_config"],
            "source_semantic_sha256": identity["training_semantics"]["aggregate_sha256"],
        }
        if not args.synthetic_smoke and not production_approved:
            diagnostic_body = {
                "deployable": False,
                "reason": "student_quality_gate_failed",
                "distillation_identity_sha256": workspace.identity_sha256,
                "metrics": metrics,
                "latency": latency,
                "distillation_training": training_metadata,
            }
            diagnostic = {
                **diagnostic_body,
                "report_sha256": sha256(_canonical_json(diagnostic_body)).hexdigest(),
            }
            diagnostics = workspace.path / "diagnostics"
            _ensure_safe_directory(diagnostics)
            _atomic_bytes(diagnostics / "nondeployable.json", _canonical_json(diagnostic))
            print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
            return 2
        metadata = StudentArtifactMetadata(
            format_version=1, input_dim=42, mixture_components=4, horizon=20, dt=PRIOR_DT,
            finger_order=FINGER_ORDER, phase_order=PHASE_ORDER,
            mirror_matrix=torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
            dataset_aggregate_sha256=document["aggregate_sha256"], teacher_ensemble_manifest_sha256=ensemble.manifest_sha256,
            teacher_seed=ensemble.teacher_seed, distillation_seed=args.seed, code_commit=_git_commit(), weight_sha256="0" * 64,
            hidden=hidden, input_field_order=MODEL_INPUT_FIELD_ORDER, output_axis_order=MIXTURE_OUTPUT_AXIS_ORDER,
        )
        provenance = {"nonproduction_synthetic": nonproduction_synthetic, "dataset_aggregate_sha256": document["aggregate_sha256"], "teacher_ensemble_manifest_sha256": ensemble.manifest_sha256}
        first = stage / "student"
        pinned = save_student_artifact(
            first, model=cpu_model, metadata=metadata, metrics=metrics, latency=latency,
            provenance=provenance, distillation_training=training_metadata,
        )
        os.replace(first, destination)
        workspace.remove_after_publish()
        print(json.dumps({"artifact": str(destination), "weight_sha256": pinned.weight_sha256, "metrics": metrics, "latency": latency, "production_approved": production_approved}, sort_keys=True))
        return 0
    except BaseException:
        if destination.exists() and destination.parent == stage.parent:
            # The destination is only created by the successful final replace; never remove a caller path.
            pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
