"""Train a hash-pinned, offline ensemble over fingertip geometry shards."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    ExpertWindow,
    PRIOR_DT,
    PriorPhase,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import (
    FingertipMixtureNet,
    mixture_log_prob,
    mixture_nll,
    temporal_regularizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import (
    deterministic_group_split,
    verify_aggregate_manifest,
    write_shards,
)


_ARRAY_NAMES = frozenset(
    {
        "fingertip_position_palm",
        "fingertip_velocity_palm",
        "contact_mask",
        "phase",
        "future_fingertip_velocity_palm",
        "source_group",
        "source_sha256",
    }
)
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


def _configure_cuda_determinism(device: str | torch.device) -> None:
    """Validate cuBLAS determinism before any CUDA operation is attempted."""

    if torch.device(device).type != "cuda":
        return
    value = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if value not in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
        supported = ", ".join(sorted(_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS))
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG must be one of {supported} for deterministic CUDA training; got {value!r}"
        )


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
        if len(self.groups) != self.inputs.shape[0] or any(not isinstance(group, str) for group in self.groups):
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
    """Hash exactly the canonical manifest body used by the atomic writer."""

    return sha256(_canonical_json(body)).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _resolve_manifest(value: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "aggregate_manifest.json"
    if path.name != "aggregate_manifest.json" or not path.is_file() or path.is_symlink():
        raise ValueError("dataset manifest must be a regular aggregate_manifest.json file")
    return path


def _phase_one_hot(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8 or values.ndim != 1 or np.any(values >= len(PriorPhase)):
        raise ValueError("phase is invalid")
    return np.eye(len(PriorPhase), dtype=np.float32)[values]


def _validate_group_assignments(document: dict[str, object]) -> None:
    split_groups = document.get("split_groups")
    if type(split_groups) is not dict or set(split_groups) != {"train", "validation", "test"}:
        raise ValueError("aggregate split groups are invalid")
    groups_by_split: list[set[str]] = []
    for split_name in ("train", "validation", "test"):
        groups = split_groups[split_name]
        if type(groups) is not list or any(type(group) is not str or not group for group in groups):
            raise ValueError("aggregate split groups are invalid")
        groups_by_split.append(set(groups))
        if len(groups_by_split[-1]) != len(groups):
            raise ValueError("aggregate split groups contain duplicates")
    if groups_by_split[0] & groups_by_split[1] or groups_by_split[0] & groups_by_split[2] or groups_by_split[1] & groups_by_split[2]:
        raise ValueError("aggregate split groups overlap")


def _load_group_split(manifest_path: Path, split_name: str) -> tuple[GroupShardDataset, dict[str, object]]:
    document = verify_aggregate_manifest(manifest_path.parent)
    _validate_group_assignments(document)
    expected_groups = document.get("split_groups", {}).get(split_name)
    if type(expected_groups) is not list or any(type(group) is not str for group in expected_groups):
        raise ValueError("aggregate split groups are invalid")
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[str] = []
    records = document["shards"]
    for record in records:
        if record.get("split") != split_name:
            continue
        relative = record["path"]
        with np.load(manifest_path.parent / relative, allow_pickle=False) as shard:
            if set(shard.files) != _ARRAY_NAMES:
                raise ValueError("shard keys do not match the frozen geometry contract")
            position = shard["fingertip_position_palm"]
            velocity = shard["fingertip_velocity_palm"]
            contact = shard["contact_mask"]
            phase = shard["phase"]
            future = shard["future_fingertip_velocity_palm"]
            shard_groups = shard["source_group"]
            source_sha = shard["source_sha256"]
        count = phase.shape[0]
        if (
            position.dtype != np.float32 or position.shape != (count, 5, 3)
            or velocity.dtype != np.float32 or velocity.shape != (count, 5, 3)
            or contact.dtype != np.bool_ or contact.shape != (count, 5)
            or future.dtype != np.float32 or future.shape != (count, 20, 5, 3)
            or shard_groups.ndim != 1 or shard_groups.shape[0] != count
            or source_sha.ndim != 1 or source_sha.shape[0] != count
            or not np.isfinite(position).all() or not np.isfinite(velocity).all() or not np.isfinite(future).all()
        ):
            raise ValueError("shard arrays violate the frozen geometry contract")
        shard_group_values = tuple(str(group) for group in shard_groups.tolist())
        if any(group not in expected_groups for group in shard_group_values):
            raise ValueError("a source group crosses aggregate split boundaries")
        network_input = np.concatenate(
            (
                position.reshape(count, -1),
                velocity.reshape(count, -1),
                contact.astype(np.float32, copy=False),
                _phase_one_hot(phase),
            ),
            axis=1,
        )
        inputs.append(network_input.astype(np.float32, copy=False))
        targets.append(future)
        groups.extend(shard_group_values)
    if not inputs:
        raise ValueError(f"aggregate has no {split_name} samples")
    return (
        GroupShardDataset(
            inputs=torch.from_numpy(np.concatenate(inputs, axis=0)),
            targets=torch.from_numpy(np.concatenate(targets, axis=0)),
            groups=tuple(groups),
        ),
        document,
    )


def _loader(dataset: GroupShardDataset, *, batch_size: int, seed: int, shuffle: bool) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _set_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _checkpoint_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resume_workspace_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.resume-v1")


def _training_identity(
    *,
    aggregate_sha: str,
    member_seeds: tuple[int, ...],
    hidden: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    acceleration_weight: float,
    jerk_weight: float,
    device: torch.device,
    synthetic_smoke: bool,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "dataset_aggregate_sha256": aggregate_sha,
        "member_seeds": list(member_seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "model": {
            "class": "FingertipMixtureNet",
            "format_version": 1,
            "hidden": list(hidden),
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": learning_rate,
        },
        "regularization": {
            "acceleration_weight": acceleration_weight,
            "jerk_weight": jerk_weight,
        },
        "software": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "device": str(device),
        "synthetic_smoke": synthetic_smoke,
    }


def _identity_sha256(identity: dict[str, object]) -> str:
    return sha256(_canonical_json(identity)).hexdigest()


def _progress_document(identity: dict[str, object]) -> dict[str, object]:
    seeds = identity["member_seeds"]
    assert isinstance(seeds, list)
    body: dict[str, object] = {
        "format_version": 1,
        "identity": identity,
        "identity_sha256": _identity_sha256(identity),
        "members": [
            {
                "member_index": index,
                "seed": seed,
                "completed_epochs": 0,
                "last": None,
                "best": None,
            }
            for index, seed in enumerate(seeds)
        ],
    }
    return {**body, "progress_sha256": sha256(_canonical_json(body)).hexdigest()}


def _read_progress(path: Path) -> dict[str, object]:
    progress_path = path / "progress.json"
    if not progress_path.is_file() or progress_path.is_symlink():
        raise ValueError("resume workspace is missing a regular progress.json identity")
    try:
        document = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("resume workspace progress is invalid") from error
    if type(document) is not dict:
        raise ValueError("resume workspace progress is invalid")
    declared = document.get("progress_sha256")
    body = dict(document)
    body.pop("progress_sha256", None)
    if type(declared) is not str or declared != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("resume workspace progress SHA-256 mismatch")
    identity = document.get("identity")
    if (
        document.get("format_version") != 1
        or type(identity) is not dict
        or document.get("identity_sha256") != _identity_sha256(identity)
        or type(document.get("members")) is not list
    ):
        raise ValueError("resume workspace identity is invalid")
    return document


def _write_progress(path: Path, document: dict[str, object]) -> None:
    body = dict(document)
    body.pop("progress_sha256", None)
    _atomic_bytes(
        path / "progress.json",
        _canonical_json({**body, "progress_sha256": sha256(_canonical_json(body)).hexdigest()}),
    )


def _record_file(path: Path, value: object, *, label: str) -> Path:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"resume {label} checkpoint record is invalid")
    relative, declared = value.get("path"), value.get("sha256")
    pure = PurePosixPath(relative) if type(relative) is str else None
    if (
        pure is None
        or pure.is_absolute()
        or len(pure.parts) != 2
        or pure.parts[0] != "checkpoints"
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"resume {label} checkpoint path is invalid")
    checkpoint = path.joinpath(*pure.parts)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError(f"resume {label} checkpoint is missing")
    if type(declared) is not str or _checkpoint_sha(checkpoint) != declared:
        raise ValueError(f"resume {label} checkpoint SHA-256 mismatch")
    return checkpoint


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


class _ResumeWorkspace:
    """Persist epoch-boundary state without ever presenting it as final output."""

    def __init__(self, path: Path, identity: dict[str, object]) -> None:
        self.path = path
        self.identity = identity
        self.identity_sha256 = _identity_sha256(identity)
        if path.exists():
            if not path.is_dir() or path.is_symlink():
                raise ValueError("resume workspace must be a regular directory")
            document = _read_progress(path)
            if document["identity_sha256"] != self.identity_sha256 or document["identity"] != identity:
                raise ValueError("resume identity does not match the requested training configuration")
            self.document = document
            self._validate_records()
        else:
            path.mkdir()
            self.document = _progress_document(identity)
            _write_progress(path, self.document)

    def _member_record(self, member_index: int, seed: int) -> dict[str, object]:
        records = self.document["members"]
        if type(records) is not list or member_index >= len(records):
            raise ValueError("resume member records are invalid")
        record = records[member_index]
        if (
            type(record) is not dict
            or record.get("member_index") != member_index
            or record.get("seed") != seed
        ):
            raise ValueError("resume member identity is invalid")
        return record

    def _validate_records(self) -> None:
        seeds = self.identity["member_seeds"]
        epochs = self.identity["epochs"]
        records = self.document["members"]
        if type(seeds) is not list or type(epochs) is not int or type(records) is not list or len(records) != len(seeds):
            raise ValueError("resume member records are invalid")
        for index, seed in enumerate(seeds):
            record = self._member_record(index, seed)
            completed = record.get("completed_epochs")
            if type(completed) is not int or not 0 <= completed <= epochs:
                raise ValueError("resume completed epoch is invalid")
            if completed == 0:
                if record.get("last") is not None or record.get("best") is not None:
                    raise ValueError("empty resume member has checkpoint records")
                continue
            _record_file(self.path, record.get("last"), label="last")
            _record_file(self.path, record.get("best"), label="best")

    def load_member(self, *, member_index: int, seed: int, hidden: tuple[int, ...]) -> _ResumeState | None:
        record = self._member_record(member_index, seed)
        if record["completed_epochs"] == 0:
            return None
        last_path = _record_file(self.path, record["last"], label="last")
        best_path = _record_file(self.path, record["best"], label="best")
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise ValueError("resume last checkpoint could not be safely loaded") from error
        if (
            type(state) is not dict
            or state.get("format_version") != 1
            or state.get("training_identity_sha256") != self.identity_sha256
            or state.get("member_index") != member_index
            or state.get("seed") != seed
            or state.get("hidden") != hidden
            or state.get("dataset_aggregate_sha256") != self.identity["dataset_aggregate_sha256"]
            or state.get("epoch") != record["completed_epochs"]
            or type(state.get("best_validation_nll")) is not float
            or not isinstance(state.get("model_state"), dict)
            or not isinstance(state.get("optimizer_state"), dict)
            or not _finite_state(state)
        ):
            raise ValueError("resume checkpoint metadata or state is invalid")
        return _ResumeState(state=state, selected_checkpoint=best_path.read_bytes())

    def commit_epoch(
        self,
        *,
        member_index: int,
        seed: int,
        state: dict[str, object],
        best_changed: bool,
    ) -> None:
        """Publish checkpoints first, then atomically advance their progress pointer."""

        record = self._member_record(member_index, seed)
        epoch = state["epoch"]
        if type(epoch) is not int or epoch != record["completed_epochs"] + 1:
            raise ValueError("resume epochs must be committed in sequence")
        next_slot = epoch % 2
        last_path = self.path / "checkpoints" / f"member-{member_index:02d}-last-{next_slot}.pt"
        _atomic_torch_save(last_path, state)
        last_record = {
            "path": last_path.relative_to(self.path).as_posix(),
            "sha256": _checkpoint_sha(last_path),
        }
        best_record = record["best"]
        if best_changed:
            best_path = self.path / "checkpoints" / f"member-{member_index:02d}-best-{next_slot}.pt"
            _atomic_torch_save(best_path, state)
            best_record = {
                "path": best_path.relative_to(self.path).as_posix(),
                "sha256": _checkpoint_sha(best_path),
            }
        if best_record is None:
            raise RuntimeError("first committed epoch did not select a best checkpoint")
        record.update(
            {
                "completed_epochs": epoch,
                "last": last_record,
                "best": best_record,
            }
        )
        _write_progress(self.path, self.document)

    def import_member(
        self,
        *,
        member_index: int,
        seed: int,
        state: dict[str, object],
        selected_checkpoint: bytes,
    ) -> None:
        record = self._member_record(member_index, seed)
        if record["completed_epochs"] != 0:
            raise ValueError("cannot import into a non-empty resume member")
        epoch = state.get("epoch")
        if type(epoch) is not int or not 1 <= epoch <= self.identity["epochs"]:
            raise ValueError("resume checkpoint epoch is incompatible with requested epochs")
        state = dict(state)
        state["training_identity_sha256"] = self.identity_sha256
        last_path = self.path / "checkpoints" / f"member-{member_index:02d}-last-0.pt"
        best_path = self.path / "checkpoints" / f"member-{member_index:02d}-best-0.pt"
        _atomic_torch_save(last_path, state)
        _atomic_bytes(best_path, selected_checkpoint)
        record.update(
            {
                "completed_epochs": state["epoch"],
                "last": {"path": last_path.relative_to(self.path).as_posix(), "sha256": _checkpoint_sha(last_path)},
                "best": {"path": best_path.relative_to(self.path).as_posix(), "sha256": _checkpoint_sha(best_path)},
            }
        )
        _write_progress(self.path, self.document)

    def best_checkpoint(self, *, member_index: int, seed: int) -> Path:
        record = self._member_record(member_index, seed)
        return _record_file(self.path, record["best"], label="best")

    def remove_after_publish(self) -> None:
        document = _read_progress(self.path)
        if document["identity_sha256"] != self.identity_sha256 or document["identity"] != self.identity:
            raise ValueError("refusing to remove a resume workspace with changed identity")
        suffix = ".resume-v1"
        output_name = self.path.name[1 : -len(suffix)] if self.path.name.startswith(".") else ""
        if not output_name or _resume_workspace_path(self.path.parent / output_name) != self.path or self.path.is_symlink():
            raise ValueError("refusing to remove an unexpected resume workspace path")
        shutil.rmtree(self.path)


@dataclass(frozen=True)
class _ResumeState:
    state: dict[str, object]
    selected_checkpoint: bytes


def _load_resume(
    path: Path | None,
    *,
    member_index: int,
    seed: int,
    hidden: tuple[int, ...],
    aggregate_sha: str,
    training_identity_sha: str | None = None,
) -> _ResumeState | None:
    if path is None:
        return None
    if not path.is_dir() or path.is_symlink():
        raise ValueError("resume checkpoint must be an ensemble output directory")
    manifest_path = path / "ensemble_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("resume output is missing ensemble_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("resume ensemble manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("resume ensemble manifest is invalid")
    declared_manifest_sha = manifest.get("ensemble_manifest_sha256")
    manifest_body = dict(manifest)
    manifest_body.pop("ensemble_manifest_sha256", None)
    if (
        type(declared_manifest_sha) is not str
        or len(declared_manifest_sha) != 64
        or declared_manifest_sha != _ensemble_manifest_sha256(manifest_body)
    ):
        raise ValueError("resume ensemble manifest SHA-256 mismatch")
    if (
        manifest.get("format_version") != 1
        or manifest.get("dataset_aggregate_sha256") != aggregate_sha
        or manifest.get("hidden") != list(hidden)
    ):
        raise ValueError("resume ensemble manifest does not match the verified aggregate and model")
    manifest_identity = manifest.get("training_identity")
    manifest_identity_sha = manifest.get("training_identity_sha256")
    if training_identity_sha is not None and (
        type(manifest_identity) is not dict
        or type(manifest_identity_sha) is not str
        or manifest_identity_sha != _identity_sha256(manifest_identity)
        or manifest_identity_sha != training_identity_sha
    ):
        raise ValueError("resume ensemble training identity does not match the requested configuration")
    records = manifest.get("members")
    if type(records) is not list:
        raise ValueError("resume ensemble member records are invalid")
    record = next((item for item in records if isinstance(item, dict) and item.get("member_index") == member_index), None)
    expected_relative = f"checkpoints/member-{member_index:02d}-best.pt"
    if (
        record is None
        or record.get("seed") != seed
        or record.get("checkpoint") != expected_relative
        or type(record.get("checkpoint_sha256")) is not str
    ):
        raise ValueError("resume ensemble member does not match the requested seed")
    checkpoint_path = path / expected_relative
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(f"missing resume checkpoint: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.read_bytes()
    if sha256(checkpoint_bytes).hexdigest() != record["checkpoint_sha256"]:
        raise ValueError("resume checkpoint SHA-256 does not match ensemble manifest")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format_version") != 1
        or state.get("member_index") != member_index
        or state.get("seed") != seed
        or state.get("hidden") != hidden
        or state.get("dataset_aggregate_sha256") != aggregate_sha
        or (
            training_identity_sha is not None
            and state.get("training_identity_sha256") != training_identity_sha
        )
        or type(state.get("epoch")) is not int
        or type(state.get("best_validation_nll")) is not float
        or not isinstance(state.get("model_state"), dict)
        or not isinstance(state.get("optimizer_state"), dict)
    ):
        raise ValueError("resume checkpoint metadata is invalid")
    return _ResumeState(state=state, selected_checkpoint=checkpoint_bytes)


def _train_member(
    *,
    member_index: int,
    seed: int,
    hidden: tuple[int, ...],
    train_data: GroupShardDataset,
    validation_data: GroupShardDataset,
    aggregate_sha: str,
    workspace: _ResumeWorkspace,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    acceleration_weight: float,
    jerk_weight: float,
    device: torch.device,
    resume: Path | None,
    epoch_commit_hook: Callable[[int, int], None] | None = None,
) -> tuple[FingertipMixtureNet, dict[str, object]]:
    _configure_cuda_determinism(device)
    _set_seed(seed)
    model = FingertipMixtureNet(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    resume_state = workspace.load_member(member_index=member_index, seed=seed, hidden=hidden)
    if resume_state is None:
        resume_state = _load_resume(
            resume,
            member_index=member_index,
            seed=seed,
            hidden=hidden,
            aggregate_sha=aggregate_sha,
            training_identity_sha=workspace.identity_sha256,
        )
        if resume_state is not None:
            workspace.import_member(
                member_index=member_index,
                seed=seed,
                state=resume_state.state,
                selected_checkpoint=resume_state.selected_checkpoint,
            )
            resume_state = workspace.load_member(member_index=member_index, seed=seed, hidden=hidden)
    start_epoch = 0
    best_validation_nll = math.inf
    if resume_state is not None:
        model.load_state_dict(resume_state.state["model_state"])
        optimizer.load_state_dict(resume_state.state["optimizer_state"])
        start_epoch = int(resume_state.state["epoch"])
        best_validation_nll = float(resume_state.state["best_validation_nll"])
    validation_loader = _loader(validation_data, batch_size=batch_size, seed=seed, shuffle=False)
    for epoch in range(start_epoch, epochs):
        model.train()
        for network_input, target in _loader(train_data, batch_size=batch_size, seed=seed + epoch, shuffle=True):
            network_input, target = network_input.to(device), target.to(device)
            distribution = model(network_input)
            acceleration, jerk = temporal_regularizer(distribution, dt=PRIOR_DT)
            loss = mixture_nll(distribution, target) + acceleration_weight * acceleration + jerk_weight * jerk
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite ensemble training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_nll = _mean_nll(model, validation_loader, device)
        if not math.isfinite(validation_nll):
            raise FloatingPointError("non-finite validation NLL")
        state = {
            "format_version": 1,
            "member_index": member_index,
            "seed": seed,
            "hidden": hidden,
            "epoch": epoch + 1,
            "best_validation_nll": min(best_validation_nll, validation_nll),
            "dataset_aggregate_sha256": aggregate_sha,
            "training_identity_sha256": workspace.identity_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        best_changed = validation_nll < best_validation_nll
        if best_changed:
            best_validation_nll = validation_nll
            state["best_validation_nll"] = best_validation_nll
        workspace.commit_epoch(
            member_index=member_index,
            seed=seed,
            state=state,
            best_changed=best_changed,
        )
        if epoch_commit_hook is not None:
            epoch_commit_hook(member_index, epoch + 1)
    best_path = workspace.best_checkpoint(member_index=member_index, seed=seed)
    if not best_path.exists():
        raise RuntimeError("no best validation checkpoint was selected")
    selected = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state"])
    model.eval()
    return model, {
        "member_index": member_index,
        "seed": seed,
        "best_validation_nll": float(selected["best_validation_nll"]),
        "checkpoint": f"checkpoints/member-{member_index:02d}-best.pt",
        "checkpoint_sha256": _checkpoint_sha(best_path),
    }


def _mean_nll(model: FingertipMixtureNet, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for network_input, target in loader:
            log_prob = mixture_log_prob(model(network_input.to(device)), target.to(device))
            total += float((-log_prob.sum()).item())
            count += log_prob.numel()
    return total / count


def _ensemble_outputs(models: Sequence[FingertipMixtureNet], network_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distributions = [model(network_input) for model in models]
    return (
        torch.stack([distribution.logits for distribution in distributions], dim=1),
        torch.stack([distribution.mean for distribution in distributions], dim=1),
        torch.stack([distribution.log_std for distribution in distributions], dim=1),
    )


def _mixture_quantile(means: torch.Tensor, std: torch.Tensor, weights: torch.Tensor, probability: float) -> torch.Tensor:
    if not 0.0 < probability < 1.0:
        raise ValueError("interval probability must be strictly between zero and one")
    lower = (means - 10.0 * std).amin(dim=-1)
    upper = (means + 10.0 * std).amax(dim=-1)
    target = torch.full_like(lower, probability)
    for _ in range(36):
        midpoint = (lower + upper) * 0.5
        cdf = (0.5 * (1.0 + torch.erf((midpoint.unsqueeze(-1) - means) / (std * math.sqrt(2.0)))) * weights).sum(dim=-1)
        lower = torch.where(cdf < target, midpoint, lower)
        upper = torch.where(cdf < target, upper, midpoint)
    return (lower + upper) * 0.5


def _ensemble_metrics(models: Sequence[FingertipMixtureNet], dataset: GroupShardDataset, *, batch_size: int, device: torch.device) -> dict[str, float]:
    first_squared = endpoint_squared = zero_first_squared = zero_endpoint_squared = 0.0
    covered = dimensions = nll_total = 0.0
    samples = 0
    with torch.no_grad():
        for network_input, target in _loader(dataset, batch_size=batch_size, seed=0, shuffle=False):
            network_input, target = network_input.to(device), target.to(device)
            logits, means, log_std = _ensemble_outputs(models, network_input)
            member_count = len(models)
            component_weights = logits.softmax(-1) / member_count
            predictive_mean = (component_weights[..., None, None, None] * means).sum(dim=(1, 2))
            member_log_probs = []
            for member in range(member_count):
                from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import MixtureDistribution
                member_log_probs.append(mixture_log_prob(MixtureDistribution(logits[:, member], means[:, member], log_std[:, member]), target))
            nll_total += float((-torch.logsumexp(torch.stack(member_log_probs, 1) - math.log(member_count), dim=1).sum()).item())
            first_squared += float((predictive_mean[:, 0] - target[:, 0]).square().sum().item())
            zero_first_squared += float(target[:, 0].square().sum().item())
            integration_dt = 0.01
            if PRIOR_DT != integration_dt:
                raise RuntimeError("endpoint integration no longer matches the frozen horizon rate")
            predicted_endpoint = predictive_mean.sum(dim=1) * integration_dt
            target_endpoint = target.sum(dim=1) * integration_dt
            endpoint_squared += float((predicted_endpoint - target_endpoint).square().sum().item())
            zero_endpoint_squared += float(target_endpoint.square().sum().item())
            flat_means = means.permute(0, 3, 4, 5, 1, 2).reshape(*target.shape, -1)
            flat_std = log_std.exp().permute(0, 3, 4, 5, 1, 2).reshape(*target.shape, -1)
            flat_weights = component_weights[:, None, None, None, :, :].expand(
                target.shape[0], target.shape[1], target.shape[2], target.shape[3], member_count, logits.shape[-1]
            ).reshape(*target.shape, -1)
            lower = _mixture_quantile(flat_means, flat_std, flat_weights, 0.1)
            upper = _mixture_quantile(flat_means, flat_std, flat_weights, 0.9)
            covered += float(((target >= lower) & (target <= upper)).sum().item())
            dimensions += target.numel()
            samples += target.shape[0]
    if samples == 0:
        raise ValueError("held-out group split is empty")
    first_count = samples * 5 * 3
    endpoint_count = samples * 5 * 3
    first_rmse = math.sqrt(first_squared / first_count)
    endpoint_rmse = math.sqrt(endpoint_squared / endpoint_count)
    baseline_first_rmse = math.sqrt(zero_first_squared / first_count)
    baseline_endpoint_rmse = math.sqrt(zero_endpoint_squared / endpoint_count)
    return {
        "test_nll": nll_total / samples,
        "first_step_velocity_rmse": first_rmse,
        "first_step_zero_rmse": baseline_first_rmse,
        "first_step_improvement": (baseline_first_rmse - first_rmse) / baseline_first_rmse,
        "endpoint_rmse": endpoint_rmse,
        "endpoint_zero_rmse": baseline_endpoint_rmse,
        "endpoint_improvement": (baseline_endpoint_rmse - endpoint_rmse) / baseline_endpoint_rmse,
        "interval_80_coverage": covered / dimensions,
    }


def _synthetic_manifest(stage: Path) -> Path:
    groups = [f"synthetic/sequence-{index:02d}/rh" for index in range(20)]
    windows: list[ExpertWindow] = []
    for group_index, group in enumerate(groups):
        for frame in range(3):
            base = (group_index * 3 + frame) / 100.0
            position = torch.full((5, 3), base, dtype=torch.float32)
            velocity = torch.full((5, 3), 0.02 + base, dtype=torch.float32)
            horizon = torch.arange(1, 21, dtype=torch.float32).reshape(20, 1, 1)
            future = velocity.unsqueeze(0) + 0.001 * horizon
            windows.append(ExpertWindow(position, velocity, torch.zeros(5, dtype=torch.bool), PriorPhase.APPROACH, future.expand(-1, 5, 3).clone(), group, f"{group_index + 1:064x}"))
    root = stage / "synthetic-shards"
    write_shards(
        root,
        windows,
        deterministic_group_split(groups, seed=17),
        shard_size=16,
        verified_inputs={"nonproduction_synthetic": True},
    )
    return root / "aggregate_manifest.json"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("member seeds must be comma-separated integers") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("member seeds must be distinct and contain at least two values")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--acceleration-weight", type=float, default=1e-5)
    parser.add_argument("--jerk-weight", type=float, default=1e-7)
    parser.add_argument("--member-seeds", type=_parse_seeds, default=(1701, 2718, 3141))
    parser.add_argument("--hidden", default="512,512,512")
    parser.add_argument("--device", default="cpu")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    epoch_commit_hook: Callable[[int, int], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.synthetic_smoke) == bool(args.dataset_manifest):
        raise ValueError("choose exactly one of --dataset-manifest and --synthetic-smoke")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0 or args.acceleration_weight < 0.0 or args.jerk_weight < 0.0:
        raise ValueError("training values are invalid")
    try:
        hidden = tuple(int(value) for value in args.hidden.split(","))
    except ValueError as error:
        raise ValueError("hidden must be comma-separated integer widths") from error
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden widths are invalid")
    _configure_cuda_determinism(args.device)
    device = torch.device(args.device)
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.train-", dir=destination.parent))
    try:
        manifest_path = _synthetic_manifest(stage) if args.synthetic_smoke else _resolve_manifest(args.dataset_manifest)
        train_data, document = _load_group_split(manifest_path, "train")
        validation_data, validation_document = _load_group_split(manifest_path, "validation")
        test_data, test_document = _load_group_split(manifest_path, "test")
        if document["aggregate_sha256"] != validation_document["aggregate_sha256"] or document["aggregate_sha256"] != test_document["aggregate_sha256"]:
            raise ValueError("aggregate changed during verified split loading")
        identity = _training_identity(
            aggregate_sha=document["aggregate_sha256"],
            member_seeds=tuple(args.member_seeds),
            hidden=hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            acceleration_weight=args.acceleration_weight,
            jerk_weight=args.jerk_weight,
            device=device,
            synthetic_smoke=bool(args.synthetic_smoke),
        )
        resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
        if resume is not None:
            _load_resume(
                resume,
                member_index=0,
                seed=args.member_seeds[0],
                hidden=hidden,
                aggregate_sha=document["aggregate_sha256"],
                training_identity_sha=_identity_sha256(identity),
            )
        workspace = _ResumeWorkspace(_resume_workspace_path(destination), identity)
        members: list[FingertipMixtureNet] = []
        member_records: list[dict[str, object]] = []
        for index, seed in enumerate(args.member_seeds):
            member, record = _train_member(
                member_index=index,
                seed=seed,
                hidden=hidden,
                train_data=train_data,
                validation_data=validation_data,
                aggregate_sha=document["aggregate_sha256"],
                workspace=workspace,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                acceleration_weight=args.acceleration_weight,
                jerk_weight=args.jerk_weight,
                device=device,
                resume=resume,
                epoch_commit_hook=epoch_commit_hook,
            )
            members.append(member)
            member_records.append(record)
        for record in member_records:
            index = int(record["member_index"])
            seed = int(record["seed"])
            source = workspace.best_checkpoint(member_index=index, seed=seed)
            destination_checkpoint = stage / str(record["checkpoint"])
            _atomic_bytes(destination_checkpoint, source.read_bytes())
            if _checkpoint_sha(destination_checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError("published checkpoint copy changed SHA-256")
        metrics = _ensemble_metrics(members, test_data, batch_size=args.batch_size, device=device)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("held-out metrics are non-finite")
        nonproduction_synthetic = document.get("verified_inputs", {}).get("nonproduction_synthetic") is True
        production_deployable = (
            not args.synthetic_smoke
            and not nonproduction_synthetic
            and metrics["first_step_improvement"] >= 0.10
            and metrics["endpoint_improvement"] >= 0.10
            and 0.65 <= metrics["interval_80_coverage"] <= 0.95
        )
        body = {
            "format_version": 1,
            "dataset_aggregate_sha256": document["aggregate_sha256"],
            "training_identity": identity,
            "training_identity_sha256": workspace.identity_sha256,
            "member_seeds": list(args.member_seeds),
            "hidden": list(hidden),
            "members": member_records,
            "metrics": metrics,
            "synthetic_smoke": bool(args.synthetic_smoke),
            "production_deployable": production_deployable,
        }
        manifest_sha = _ensemble_manifest_sha256(body)
        _atomic_bytes(stage / "ensemble_manifest.json", _canonical_json({**body, "ensemble_manifest_sha256": manifest_sha}))
        os.replace(stage, destination)
        workspace.remove_after_publish()
        print(json.dumps({**body, "ensemble_manifest_sha256": manifest_sha}, sort_keys=True))
        return 0
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
