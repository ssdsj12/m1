"""Strict validation for expert ensemble artifacts shared by producer and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import pickle
import stat

import torch

from .model import FingertipMixtureNet


_MANIFEST_FIELDS = {
    "format_version", "dataset_aggregate_sha256", "training_identity",
    "training_identity_sha256", "member_seeds", "hidden", "members", "metrics",
    "synthetic_smoke", "production_deployable", "ensemble_manifest_sha256",
}
_METRIC_FIELDS = {
    "test_nll", "first_step_velocity_rmse", "first_step_zero_rmse",
    "first_step_improvement", "endpoint_rmse", "endpoint_zero_rmse",
    "endpoint_improvement", "interval_80_coverage",
}
_IDENTITY_FIELDS = {
    "format_version", "dataset_aggregate_sha256", "member_seeds", "epochs",
    "batch_size", "model", "optimizer", "regularization", "software",
    "training_semantics", "cublas_workspace_config", "device", "synthetic_smoke",
}
_CHECKPOINT_FIELDS = {
    "format_version", "member_index", "seed", "hidden", "epoch",
    "best_validation_nll", "dataset_aggregate_sha256", "training_identity_sha256",
    "model_state", "optimizer_state",
}
PRODUCTION_MEMBER_SEEDS = (42, 43, 44, 45, 46)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def ensemble_manifest_sha256(body: dict[str, object]) -> str:
    return sha256(_canonical_json(body)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    value = path.expanduser()
    if ".." in value.parts:
        raise ValueError(f"unsafe parent traversal in ensemble path: {value}")
    return value if value.is_absolute() else Path.cwd() / value


def _require_safe_directory(path: Path) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise ValueError(f"ensemble directory is missing: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"ensemble directory chain contains symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"ensemble directory chain contains non-directory: {current}")
    return absolute


def _require_regular_file(path: Path, *, label: str) -> None:
    _require_safe_directory(path.parent)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file and not a symlink")


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and type(value) is not bool and math.isfinite(float(value))


def adamw_contract(learning_rate: float) -> dict[str, object]:
    """Return the exact AdamW semantics pinned by resumable offline training."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("AdamW learning rate must be finite and positive")
    return {
        "class": "torch.optim.AdamW",
        "learning_rate": float(learning_rate),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }


def validate_model_state(model: FingertipMixtureNet, state: object, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or set(state) != set(model.state_dict()):
        raise ValueError(f"{label} model state schema is invalid")
    clean: dict[str, torch.Tensor] = {}
    for key, reference in model.state_dict().items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor) or value.shape != reference.shape
            or value.dtype != reference.dtype or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{label} model state is invalid")
        clean[key] = value
    return clean


def validate_adamw_state(
    model: FingertipMixtureNet, state: object, *, contract: object,
    expected_step: int, label: str,
) -> dict[str, object]:
    """Reject optimizer state that cannot be proven to match the pinned AdamW contract."""

    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError(f"{label} expected AdamW step is invalid")
    if type(contract) is not dict or contract != adamw_contract(contract.get("learning_rate", math.nan)):
        raise ValueError(f"{label} AdamW identity is invalid")
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError(f"{label} optimizer state schema is invalid")
    groups, slots = state["param_groups"], state["state"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict or type(slots) is not dict:
        raise ValueError(f"{label} optimizer state schema is invalid")
    group = groups[0]
    expected_group_keys = (set(contract) - {"class", "learning_rate"}) | {"lr", "params"}
    parameters = list(model.parameters())
    expected_ids = list(range(len(parameters)))
    if set(group) != expected_group_keys or group.get("params") != expected_ids:
        raise ValueError(f"{label} optimizer parameter roster is invalid")
    expected_values = {key: value for key, value in contract.items() if key not in {"class", "learning_rate"}}
    expected_values["lr"] = contract["learning_rate"]
    for key, expected in expected_values.items():
        actual = group[key]
        if key == "betas":
            if type(actual) not in (tuple, list) or list(actual) != expected:
                raise ValueError(f"{label} AdamW parameter group is invalid")
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{label} AdamW parameter group is invalid")
    if set(slots) != set(expected_ids):
        raise ValueError(f"{label} optimizer state parameter IDs are invalid")
    for parameter_id, parameter in enumerate(parameters):
        values = slots[parameter_id]
        if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"{label} optimizer moment schema is invalid")
        step, average, square_average = values["step"], values["exp_avg"], values["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.shape != ()
            or not torch.isfinite(step).item() or float(step.item()) != float(expected_step)
            or not isinstance(average, torch.Tensor) or average.dtype != parameter.dtype
            or average.shape != parameter.shape or not torch.isfinite(average).all().item()
            or not isinstance(square_average, torch.Tensor) or square_average.dtype != parameter.dtype
            or square_average.shape != parameter.shape or not torch.isfinite(square_average).all().item()
        ):
            raise ValueError(f"{label} optimizer moment state is invalid")
    return state


def _validate_training_identity(
    identity: object, *, dataset_sha256: str, member_seeds: tuple[int, ...],
    hidden: tuple[int, ...], synthetic: bool,
) -> dict[str, object]:
    if type(identity) is not dict or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("ensemble training identity schema is invalid")
    if (
        identity["format_version"] != 1
        or identity["dataset_aggregate_sha256"] != dataset_sha256
        or identity["member_seeds"] != list(member_seeds)
        or type(identity["epochs"]) is not int or identity["epochs"] <= 0
        or type(identity["batch_size"]) is not int or identity["batch_size"] <= 0
        or identity["synthetic_smoke"] is not synthetic
    ):
        raise ValueError("ensemble training identity does not match dataset or roster")
    model = identity["model"]
    if type(model) is not dict or model != {
        "class": "FingertipMixtureNet", "format_version": 1, "hidden": list(hidden),
    }:
        raise ValueError("ensemble training identity model is invalid")
    optimizer = identity["optimizer"]
    if (
        type(optimizer) is not dict or set(optimizer) != {"class", "learning_rate"}
        or optimizer.get("class") != "torch.optim.AdamW"
        or not _finite_number(optimizer.get("learning_rate"))
        or float(optimizer["learning_rate"]) <= 0.0
    ):
        raise ValueError("ensemble training identity optimizer is invalid")
    regularization = identity["regularization"]
    if (
        type(regularization) is not dict
        or set(regularization) != {"acceleration_weight", "jerk_weight"}
        or any(not _finite_number(regularization.get(key)) or float(regularization[key]) < 0.0
               for key in ("acceleration_weight", "jerk_weight"))
    ):
        raise ValueError("ensemble training identity regularization is invalid")
    software = identity["software"]
    if (
        type(software) is not dict
        or set(software) != {"torch_version", "torch_cuda_build", "numpy_version"}
        or type(software["torch_version"]) is not str or not software["torch_version"]
        or type(software["numpy_version"]) is not str or not software["numpy_version"]
        or software["torch_cuda_build"] is not None and type(software["torch_cuda_build"]) is not str
    ):
        raise ValueError("ensemble training identity software is invalid")
    semantics = identity["training_semantics"]
    if type(semantics) is not dict or set(semantics) != {"source_sha256", "aggregate_sha256"}:
        raise ValueError("ensemble training identity semantics are invalid")
    sources = semantics["source_sha256"]
    if (
        type(sources) is not dict or set(sources) != {"trainer", "model", "contracts"}
        or any(not _sha(value) for value in sources.values())
        or semantics["aggregate_sha256"] != sha256(_canonical_json(sources)).hexdigest()
    ):
        raise ValueError("ensemble training identity semantics SHA is invalid")
    device, cublas = identity["device"], identity["cublas_workspace_config"]
    if type(device) is not dict or device.get("type") not in {"cpu", "cuda"}:
        raise ValueError("ensemble training identity device is invalid")
    if device["type"] == "cpu":
        if set(device) != {"type"} or cublas is not None:
            raise ValueError("ensemble training identity device/CUBLAS is invalid")
    elif (
        set(device) != {"type", "uuid", "name", "compute_capability"}
        or type(device["uuid"]) is not str or not device["uuid"]
        or type(device["name"]) is not str or not device["name"]
        or type(device["compute_capability"]) is not list or len(device["compute_capability"]) != 2
        or any(type(number) is not int or number < 0 for number in device["compute_capability"])
        or cublas not in {":4096:8", ":16:8"}
    ):
        raise ValueError("ensemble training identity device/CUBLAS is invalid")
    return identity


def _validate_metrics(
    metrics: object, *, synthetic: bool, nonproduction_synthetic: bool,
) -> tuple[dict[str, float], bool]:
    if type(synthetic) is not bool or type(nonproduction_synthetic) is not bool:
        raise ValueError("ensemble production classification is invalid")
    if type(metrics) is not dict or set(metrics) != _METRIC_FIELDS:
        raise ValueError("ensemble metrics schema is invalid")
    if any(not _finite_number(metrics[key]) for key in _METRIC_FIELDS):
        raise ValueError("ensemble metrics must be finite")
    values = {key: float(metrics[key]) for key in _METRIC_FIELDS}
    if (
        values["first_step_velocity_rmse"] < 0.0 or values["endpoint_rmse"] < 0.0
        or values["first_step_zero_rmse"] <= 0.0 or values["endpoint_zero_rmse"] <= 0.0
        or not 0.0 <= values["interval_80_coverage"] <= 1.0
        or not math.isclose(
            values["first_step_improvement"],
            1.0 - values["first_step_velocity_rmse"] / values["first_step_zero_rmse"],
            abs_tol=1e-8,
        )
        or not math.isclose(
            values["endpoint_improvement"],
            1.0 - values["endpoint_rmse"] / values["endpoint_zero_rmse"],
            abs_tol=1e-8,
        )
    ):
        raise ValueError("ensemble metrics are inconsistent")
    deployable = bool(
        not synthetic and not nonproduction_synthetic
        and values["first_step_improvement"] >= 0.10
        and values["endpoint_improvement"] >= 0.10
        and 0.65 <= values["interval_80_coverage"] <= 0.95
    )
    return values, deployable


@dataclass(frozen=True)
class ValidatedEnsemble:
    manifest: dict[str, object]
    models: tuple[FingertipMixtureNet, ...]


def validate_ensemble_artifact(
    root: Path, *, expected_dataset_sha256: str, expected_member_seeds: tuple[int, ...],
    expected_synthetic: bool, expected_nonproduction_synthetic: bool = False,
) -> ValidatedEnsemble:
    """Validate exact manifest/checkpoint semantics before exposing teacher models."""

    if (
        type(expected_member_seeds) is not tuple or len(expected_member_seeds) < 2
        or any(type(seed) is not int for seed in expected_member_seeds)
        or len(set(expected_member_seeds)) != len(expected_member_seeds)
    ):
        raise ValueError("expected ensemble member roster is invalid")
    if not expected_synthetic and expected_member_seeds != PRODUCTION_MEMBER_SEEDS:
        raise ValueError(
            "production member roster must be exactly 42,43,44,45,46"
        )
    root = _require_safe_directory(root)
    _require_safe_directory(root / "checkpoints")
    manifest_path = root / "ensemble_manifest.json"
    _require_regular_file(manifest_path, label="ensemble manifest")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ensemble manifest is invalid") from error
    if type(document) is not dict or set(document) != _MANIFEST_FIELDS:
        raise ValueError("ensemble manifest schema is invalid")
    body = dict(document)
    declared = body.pop("ensemble_manifest_sha256")
    if not _sha(declared) or declared != ensemble_manifest_sha256(body):
        raise ValueError("ensemble manifest SHA-256 mismatch")
    if (
        document["format_version"] != 1
        or document["dataset_aggregate_sha256"] != expected_dataset_sha256
        or document["synthetic_smoke"] is not expected_synthetic
        or document["member_seeds"] != list(expected_member_seeds)
        or len(set(document["member_seeds"])) != len(expected_member_seeds)
    ):
        raise ValueError("ensemble manifest does not match dataset, synthetic mode, or member roster")
    hidden_value = document["hidden"]
    if type(hidden_value) is not list or not hidden_value or any(type(width) is not int or width <= 0 for width in hidden_value):
        raise ValueError("ensemble architecture is invalid")
    hidden = tuple(hidden_value)
    identity = _validate_training_identity(
        document["training_identity"], dataset_sha256=expected_dataset_sha256,
        member_seeds=expected_member_seeds, hidden=hidden, synthetic=expected_synthetic,
    )
    identity_sha = sha256(_canonical_json(identity)).hexdigest()
    if document["training_identity_sha256"] != identity_sha:
        raise ValueError("ensemble training identity SHA mismatch")
    _, deployable = _validate_metrics(
        document["metrics"], synthetic=expected_synthetic,
        nonproduction_synthetic=expected_nonproduction_synthetic,
    )
    if type(document["production_deployable"]) is not bool or document["production_deployable"] != deployable:
        raise ValueError("ensemble production deployable flag does not match gates")
    records = document["members"]
    if type(records) is not list or len(records) != len(expected_member_seeds):
        raise ValueError("ensemble member roster is invalid")
    models: list[FingertipMixtureNet] = []
    for index, seed in enumerate(expected_member_seeds):
        record = records[index]
        expected_relative = f"checkpoints/member-{index:02d}-best.pt"
        if (
            type(record) is not dict
            or set(record) != {"member_index", "seed", "best_validation_nll", "checkpoint", "checkpoint_sha256"}
            or record["member_index"] != index or record["seed"] != seed
            or record["checkpoint"] != expected_relative
            or not _finite_number(record["best_validation_nll"])
            or not _sha(record["checkpoint_sha256"])
        ):
            raise ValueError("ensemble member roster is invalid or out of order")
        checkpoint = root / expected_relative
        _require_regular_file(checkpoint, label="ensemble checkpoint")
        if _sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("ensemble checkpoint SHA-256 mismatch")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
            raise ValueError("ensemble checkpoint could not be safely loaded") from error
        if (
            type(state) is not dict or set(state) != _CHECKPOINT_FIELDS
            or state["format_version"] != 1 or state["member_index"] != index
            or state["seed"] != seed or state["hidden"] != hidden
            or type(state["epoch"]) is not int or not 1 <= state["epoch"] <= identity["epochs"]
            or state["dataset_aggregate_sha256"] != expected_dataset_sha256
            or state["training_identity_sha256"] != identity_sha
            or not _finite_number(state["best_validation_nll"])
            or float(state["best_validation_nll"]) != float(record["best_validation_nll"])
            or type(state["optimizer_state"]) is not dict
        ):
            raise ValueError("ensemble checkpoint metadata is invalid")
        model = FingertipMixtureNet(hidden=hidden)
        model.load_state_dict(validate_model_state(model, state["model_state"], label="ensemble checkpoint"), strict=True)
        # Published ensemble checkpoints are inference artifacts.  Legacy completed
        # ensembles may carry an empty optimizer state and are never resumed; only
        # resume workspaces require optimizer semantics.  The checkpoint SHA still
        # binds this inert field, while strict model validation protects inference.
        model.eval()
        models.append(model)
    return ValidatedEnsemble(document, tuple(models))


__all__ = [
    "PRODUCTION_MEMBER_SEEDS", "ValidatedEnsemble", "adamw_contract", "ensemble_manifest_sha256",
    "validate_adamw_state", "validate_ensemble_artifact", "validate_model_state",
]
