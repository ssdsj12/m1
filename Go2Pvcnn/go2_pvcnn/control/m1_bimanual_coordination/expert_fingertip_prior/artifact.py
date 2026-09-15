"""Strict, SHA-pinned serialization for the deployable fingertip student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Final

import numpy as np
import torch

from .contracts import PriorPhase, StudentArtifactMetadata
from .model import FingertipMixtureNet


_ARTIFACT_FILES = frozenset({"metadata.json", "student.pt", "metrics.json", "latency.json"})
DISTILLATION_LOSS_CONFIG = {"label_nll_weight": 0.5, "teacher_sample_weight": 0.5, "acceleration_weight": 1e-5, "jerk_weight": 1e-7}
DISTILLATION_TRAINING_DEFAULTS = {
    "samples_per_state": 8,
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "device": "cpu",
}

RUN_E_AGGREGATE_SHA256: Final = "dfaa213a89d8a87b267ffd7ed9dc69d5a3f8582204e79a11d575d30140a57c7c"
RUN_E_SHARD_COUNT: Final = 241
RUN_E_SAMPLE_COUNT: Final = 984_641


def validate_run_e_corpus_identity(verified_manifest: object) -> None:
    """Check frozen run_e facts AFTER storage.verify_aggregate_manifest(root).

    This deliberately does not replace aggregate/audit/all-shard hashing and
    does not hard-bind the generic student loader to one training corpus.
    """

    if type(verified_manifest) is not dict or verified_manifest.get("aggregate_sha256") != RUN_E_AGGREGATE_SHA256:
        raise ValueError("run_e aggregate SHA-256 does not match the frozen corpus")
    shards = verified_manifest.get("shards")
    if type(shards) is not list or len(shards) != RUN_E_SHARD_COUNT:
        raise ValueError("run_e shard count does not match the frozen corpus")
    if any(type(row) is not dict or type(row.get("samples")) is not int or row["samples"] <= 0 for row in shards):
        raise ValueError("run_e shard samples must be positive integers")
    if sum(row["samples"] for row in shards) != RUN_E_SAMPLE_COUNT:
        raise ValueError("run_e sample count does not match the frozen corpus")


def validate_run_e_training_qualification(verified_manifest: object) -> None:
    """Require frozen identity AND both accepted production source corpora.

    The caller must first use the existing storage verifier. Identity alone
    never grants permission to train or waive complete-data provenance gates.
    """

    validate_run_e_corpus_identity(verified_manifest)
    totals = {"favor": 0, "oakinkv2": 0}
    for row in verified_manifest["shards"]:
        counts = row.get("source_counts")
        if (
            type(counts) is not dict or not counts or not set(counts) <= set(totals)
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or sum(counts.values()) != row["samples"]
        ):
            raise ValueError("run_e source accounting is invalid")
        for source, count in counts.items():
            totals[source] += count
    if any(count <= 0 for count in totals.values()):
        raise ValueError("run_e production training requires both favor and oakinkv2 accepted samples")


def default_distillation_training() -> dict[str, object]:
    """Return the one frozen default training contract used by CLI and artifact tests."""

    return {
        "samples_per_state": DISTILLATION_TRAINING_DEFAULTS["samples_per_state"],
        "epochs": DISTILLATION_TRAINING_DEFAULTS["epochs"],
        "batch_size": DISTILLATION_TRAINING_DEFAULTS["batch_size"],
        "learning_rate": DISTILLATION_TRAINING_DEFAULTS["learning_rate"],
        "device": {"type": DISTILLATION_TRAINING_DEFAULTS["device"]},
        "software": {
            "torch_version": str(torch.__version__),
            "torch_cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
            "numpy_version": str(np.__version__),
        },
        "cublas_workspace_config": None,
        "source_semantic_sha256": "0" * 64,
    }


def sha256_file(path: str | Path) -> str:
    """Hash a regular artifact file without importing any fetch/extraction code."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _regular(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    """Open once without following symlinks and return that immutable file snapshot."""

    _safe_directory(path.parent, create=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _safe_directory(path: str | Path, *, create: bool) -> Path:
    value = Path(path).expanduser()
    if ".." in value.parts:
        raise ValueError(f"unsafe parent traversal in artifact path: {value}")
    absolute = value if value.is_absolute() else Path.cwd() / value
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            if not create:
                raise ValueError(f"artifact directory is missing: {current}") from error
            os.mkdir(current)
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("student artifact root must be a regular directory")
    return absolute


def _metadata_document(metadata: StudentArtifactMetadata) -> dict[str, object]:
    return {
        "format_version": metadata.format_version,
        "input_dim": metadata.input_dim,
        "mixture_components": metadata.mixture_components,
        "horizon": metadata.horizon,
        "dt": metadata.dt,
        "finger_order": list(metadata.finger_order),
        "phase_order": [phase.name for phase in metadata.phase_order],
        "mirror_matrix": metadata.mirror_matrix.detach().cpu().tolist(),
        "dataset_aggregate_sha256": metadata.dataset_aggregate_sha256,
        "teacher_ensemble_manifest_sha256": metadata.teacher_ensemble_manifest_sha256,
        "teacher_seed": metadata.teacher_seed,
        "distillation_seed": metadata.distillation_seed,
        "code_commit": metadata.code_commit,
        "weight_sha256": metadata.weight_sha256,
        "hidden": list(metadata.hidden),
        "input_field_order": list(metadata.input_field_order),
        "output_axis_order": list(metadata.output_axis_order),
    }


def _metadata_from_document(document: object) -> StudentArtifactMetadata:
    if type(document) is not dict:
        raise ValueError("artifact metadata must be a JSON object")
    body = dict(document)
    declared = body.pop("metadata_sha256", None)
    required = set(_metadata_document(_metadata_for_schema()).keys()) | {
        "metrics_sha256", "metrics_bytes", "latency_sha256", "latency_bytes",
        "distillation_loss", "distillation_training", "deterministic_identity_sha256",
        "qualification_sha256",
    }
    if set(body) != required:
        raise ValueError("artifact metadata fields do not match the frozen schema")
    if type(declared) is not str or declared != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("artifact metadata SHA mismatch")
    try:
        phase_order = tuple(PriorPhase[name] for name in body["phase_order"])
        return StudentArtifactMetadata(
            format_version=body["format_version"],
            input_dim=body["input_dim"],
            mixture_components=body["mixture_components"],
            horizon=body["horizon"],
            dt=body["dt"],
            finger_order=tuple(body["finger_order"]),
            phase_order=phase_order,
            mirror_matrix=torch.tensor(body["mirror_matrix"], dtype=torch.float32),
            dataset_aggregate_sha256=body["dataset_aggregate_sha256"],
            teacher_ensemble_manifest_sha256=body["teacher_ensemble_manifest_sha256"],
            teacher_seed=body["teacher_seed"],
            distillation_seed=body["distillation_seed"],
            code_commit=body["code_commit"],
            weight_sha256=body["weight_sha256"],
            hidden=tuple(body["hidden"]),
            input_field_order=tuple(body["input_field_order"]),
            output_axis_order=tuple(body["output_axis_order"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact metadata violates frozen contracts") from error


def _metadata_for_schema() -> StudentArtifactMetadata:
    """A private valid instance used only to keep the exact JSON field set centralized."""

    from .contracts import (
        FINGER_ORDER, LEFT_REFLECTION, MODEL_INPUT_FIELD_ORDER, MIXTURE_OUTPUT_AXIS_ORDER,
        PHASE_ORDER, PRIOR_DT,
    )

    return StudentArtifactMetadata(
        format_version=1, input_dim=42, mixture_components=4, horizon=20, dt=PRIOR_DT,
        finger_order=FINGER_ORDER, phase_order=PHASE_ORDER,
        mirror_matrix=torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
        dataset_aggregate_sha256="0" * 64, teacher_ensemble_manifest_sha256="0" * 64,
        teacher_seed=0, distillation_seed=0, code_commit="0" * 40, weight_sha256="0" * 64,
        hidden=(1,), input_field_order=MODEL_INPUT_FIELD_ORDER, output_axis_order=MIXTURE_OUTPUT_AXIS_ORDER,
    )


def _json_document_from_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_json(value: object) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (str, bool, int) or value is None:
        return True
    if type(value) is list:
        return all(_finite_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _finite_json(item) for key, item in value.items())
    return False


def _validate_reports(metrics: dict[str, object], latency: dict[str, object]) -> None:
    if set(metrics) != {"metrics", "provenance"} or type(metrics["metrics"]) is not dict or type(metrics["provenance"]) is not dict:
        raise ValueError("artifact metrics/provenance schema is invalid")
    values = metrics["metrics"]
    provenance = metrics["provenance"]
    required_values = {"student_nll", "teacher_nll", "nll_delta_per_dim", "first_step_velocity_rmse", "first_step_zero_rmse", "first_step_improvement", "endpoint_rmse", "teacher_endpoint_rmse", "endpoint_zero_rmse", "endpoint_improvement", "deterministic_repeat_verified"}
    if set(values) != required_values or not _finite_json(metrics):
        raise ValueError("artifact metrics/provenance is invalid")
    numeric = required_values - {"deterministic_repeat_verified"}
    if type(values["deterministic_repeat_verified"]) is not bool or any(type(values[key]) not in (int, float) or type(values[key]) is bool or not math.isfinite(float(values[key])) for key in numeric):
        raise ValueError("artifact metrics are invalid")
    if values["deterministic_repeat_verified"] is not True or any(float(values[key]) < 0.0 for key in ("first_step_velocity_rmse", "endpoint_rmse", "teacher_endpoint_rmse")) or any(float(values[key]) <= 0.0 for key in ("first_step_zero_rmse", "endpoint_zero_rmse")) or not math.isclose(values["nll_delta_per_dim"], (values["student_nll"] - values["teacher_nll"]) / 300.0, abs_tol=1e-8) or not math.isclose(values["first_step_improvement"], 1 - values["first_step_velocity_rmse"] / values["first_step_zero_rmse"], abs_tol=1e-8) or not math.isclose(values["endpoint_improvement"], 1 - values["endpoint_rmse"] / values["endpoint_zero_rmse"], abs_tol=1e-8):
        raise ValueError("artifact derived metrics are inconsistent")
    if set(provenance) != {"nonproduction_synthetic", "dataset_aggregate_sha256", "teacher_ensemble_manifest_sha256"} or type(provenance.get("nonproduction_synthetic")) is not bool:
        raise ValueError("artifact provenance is invalid")
    if set(latency) != {"warmups", "measurements", "p99_ms", "production_approved"}:
        raise ValueError("artifact latency schema is invalid")
    if type(latency["production_approved"]) is not bool:
        raise ValueError("artifact latency production approval is invalid")
    if type(latency["warmups"]) is not int or latency["warmups"] < 100:
        raise ValueError("artifact latency warmups are invalid")
    if type(latency["measurements"]) is not int or latency["measurements"] < 1000:
        raise ValueError("artifact latency measurement count is invalid")
    if type(latency["p99_ms"]) is not float or not math.isfinite(latency["p99_ms"]) or latency["p99_ms"] < 0.0:
        raise ValueError("artifact latency p99 is invalid")


def _validate_provenance(reports: dict[str, object], metadata: StudentArtifactMetadata) -> None:
    provenance = reports["provenance"]
    assert isinstance(provenance, dict)
    if (
        provenance["dataset_aggregate_sha256"] != metadata.dataset_aggregate_sha256
        or provenance["teacher_ensemble_manifest_sha256"] != metadata.teacher_ensemble_manifest_sha256
    ):
        raise ValueError("artifact provenance does not match frozen metadata pins")


def _production_approved(values: Mapping[str, object], provenance: Mapping[str, object], p99_ms: object) -> bool:
    return bool(
        provenance.get("nonproduction_synthetic") is False
        and float(values["nll_delta_per_dim"]) <= 0.05
        and float(values["endpoint_rmse"]) <= 1.05 * float(values["teacher_endpoint_rmse"])
        and float(values["first_step_improvement"]) >= 0.10
        and float(values["endpoint_improvement"]) >= 0.10
        and float(p99_ms) < 2.0
    )


def _validate_distillation_training(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "samples_per_state", "epochs", "batch_size", "learning_rate", "device", "software",
        "cublas_workspace_config", "source_semantic_sha256",
    }:
        raise ValueError("artifact distillation training configuration is invalid")
    for key in ("samples_per_state", "epochs", "batch_size"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("artifact distillation training configuration is invalid")
    if type(value["learning_rate"]) is not float or not math.isfinite(value["learning_rate"]) or value["learning_rate"] <= 0.0:
        raise ValueError("artifact distillation training configuration is invalid")
    device = value["device"]
    if type(device) is not dict or device.get("type") not in {"cpu", "cuda"}:
        raise ValueError("artifact distillation training device is invalid")
    if device["type"] == "cpu":
        if set(device) != {"type"}:
            raise ValueError("artifact distillation training device is invalid")
    elif (
        set(device) != {"type", "uuid", "name", "compute_capability"}
        or type(device["uuid"]) is not str or not device["uuid"]
        or type(device["name"]) is not str or not device["name"]
        or type(device["compute_capability"]) is not list
        or len(device["compute_capability"]) != 2
        or any(type(number) is not int or number < 0 for number in device["compute_capability"])
    ):
        raise ValueError("artifact distillation training device is invalid")
    software = value["software"]
    if (
        type(software) is not dict
        or set(software) != {"torch_version", "torch_cuda_build", "numpy_version"}
        or type(software["torch_version"]) is not str or not software["torch_version"]
        or type(software["numpy_version"]) is not str or not software["numpy_version"]
        or software["torch_cuda_build"] is not None and type(software["torch_cuda_build"]) is not str
        or not _finite_json(software)
    ):
        raise ValueError("artifact distillation software identity is invalid")
    cublas = value["cublas_workspace_config"]
    if (
        device["type"] == "cpu" and cublas is not None
        or device["type"] == "cuda" and cublas not in {":4096:8", ":16:8"}
    ):
        raise ValueError("artifact distillation CUBLAS identity is invalid")
    semantic = value["source_semantic_sha256"]
    if type(semantic) is not str or len(semantic) != 64 or any(ch not in "0123456789abcdef" for ch in semantic):
        raise ValueError("artifact distillation source semantic SHA is invalid")
    return dict(value)


def _metadata_integrity(document: dict[str, object], metrics_bytes: bytes, latency_bytes: bytes, reports: dict[str, object], metadata: StudentArtifactMetadata) -> None:
    if document.get("distillation_loss") != DISTILLATION_LOSS_CONFIG:
        raise ValueError("artifact distillation loss configuration is invalid")
    training = _validate_distillation_training(document.get("distillation_training"))
    for label, raw in (("metrics", metrics_bytes), ("latency", latency_bytes)):
        if document.get(f"{label}_sha256") != sha256(raw).hexdigest() or document.get(f"{label}_bytes") != len(raw):
            raise ValueError(f"artifact {label} SHA mismatch")
    values, provenance = reports["metrics"], reports["provenance"]
    latency = json.loads(latency_bytes)
    expected = _production_approved(values, provenance, latency["p99_ms"])
    if latency["production_approved"] != expected:
        raise ValueError("artifact production approval does not match gates")
    identity_body = {
        "metadata": _metadata_document(metadata),
        "distillation_loss": DISTILLATION_LOSS_CONFIG,
        "distillation_training": training,
        "metrics": reports,
    }
    identity_sha = sha256(_canonical_json(identity_body)).hexdigest()
    if document.get("deterministic_identity_sha256") != identity_sha:
        raise ValueError("artifact deterministic identity mismatch")
    qualification = {
        "deterministic_identity_sha256": identity_sha,
        "latency_sha256": document["latency_sha256"],
        "production_approved": latency["production_approved"],
    }
    if document.get("qualification_sha256") != sha256(_canonical_json(qualification)).hexdigest():
        raise ValueError("artifact qualification SHA mismatch")


def _state_for_save(model: FingertipMixtureNet) -> dict[str, torch.Tensor]:
    if not isinstance(model, FingertipMixtureNet):
        raise TypeError("student model must be a FingertipMixtureNet")
    result: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all().item():
            raise ValueError("student state dict must contain finite float32 tensors")
        result[key] = value.detach().cpu().contiguous()
    return result


def _validate_state(model: FingertipMixtureNet, state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError("student weights must be a tensor state dictionary")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("student weights state_dict keys do not match architecture")
    clean: dict[str, torch.Tensor] = {}
    for key, reference in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != reference.dtype or value.shape != reference.shape:
            raise ValueError("student weights state_dict shapes/dtypes do not match architecture")
        if not torch.isfinite(value).all().item():
            raise ValueError("student weights contain non-finite values")
        clean[key] = value
    return clean


@dataclass(frozen=True)
class LoadedStudent:
    model: FingertipMixtureNet
    metadata: StudentArtifactMetadata
    metrics: dict[str, object]
    latency: dict[str, object]


def save_student_artifact(
    root: str | Path,
    *,
    model: FingertipMixtureNet,
    metadata: StudentArtifactMetadata,
    metrics: Mapping[str, object],
    latency: Mapping[str, object],
    provenance: Mapping[str, object],
    distillation_training: Mapping[str, object] | None = None,
) -> StudentArtifactMetadata:
    """Atomically save exactly the four self-validating student artifact files."""

    if not isinstance(metadata, StudentArtifactMetadata):
        raise TypeError("metadata must be StudentArtifactMetadata")
    lexical = Path(root).expanduser()
    if ".." in lexical.parts:
        raise ValueError(f"unsafe parent traversal in artifact path: {lexical}")
    destination = lexical if lexical.is_absolute() else Path.cwd() / lexical
    metric_document = dict(metrics)
    latency_document = dict(latency)
    approval = metric_document.pop("production_approved", latency_document.get("production_approved"))
    if approval is None:
        approval = _production_approved(metric_document, provenance, latency_document.get("p99_ms"))
    if "production_approved" in latency_document and latency_document["production_approved"] != approval:
        raise ValueError("conflicting artifact production approval")
    latency_document["production_approved"] = approval
    reports = {"metrics": metric_document, "provenance": dict(provenance)}
    _validate_reports(reports, latency_document)
    _validate_provenance(reports, metadata)
    expected_approval = _production_approved(metric_document, provenance, latency_document["p99_ms"])
    if approval is not expected_approval:
        raise ValueError("artifact production approval does not match recomputed gates")
    training = _validate_distillation_training(
        default_distillation_training() if distillation_training is None else dict(distillation_training)
    )
    state = _state_for_save(model)
    if model.hidden != metadata.hidden:
        raise ValueError("student model architecture does not match metadata hidden widths")
    _safe_directory(destination.parent, create=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"student artifact destination already exists: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        weights_path = stage / "student.pt"
        with weights_path.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        pinned = replace(metadata, weight_sha256=sha256_file(weights_path))
        metrics_bytes, latency_bytes = _canonical_json(reports), _canonical_json(latency_document)
        metadata_body = {
            **_metadata_document(pinned),
            "metrics_sha256": sha256(metrics_bytes).hexdigest(),
            "metrics_bytes": len(metrics_bytes),
            "latency_sha256": sha256(latency_bytes).hexdigest(),
            "latency_bytes": len(latency_bytes),
            "distillation_loss": DISTILLATION_LOSS_CONFIG,
            "distillation_training": training,
        }
        identity_body = {
            "metadata": _metadata_document(pinned),
            "distillation_loss": DISTILLATION_LOSS_CONFIG,
            "distillation_training": training,
            "metrics": reports,
        }
        metadata_body["deterministic_identity_sha256"] = sha256(_canonical_json(identity_body)).hexdigest()
        metadata_body["qualification_sha256"] = sha256(_canonical_json({
            "deterministic_identity_sha256": metadata_body["deterministic_identity_sha256"],
            "latency_sha256": metadata_body["latency_sha256"],
            "production_approved": latency_document["production_approved"],
        })).hexdigest()
        metadata_document = {**metadata_body, "metadata_sha256": sha256(_canonical_json(metadata_body)).hexdigest()}
        for path, value in (
            (stage / "metadata.json", metadata_document),
            (stage / "metrics.json", reports), (stage / "latency.json", latency_document),
        ):
            with path.open("xb") as handle:
                handle.write(_canonical_json(value))
                handle.flush()
                os.fsync(handle.fileno())
        if {path.name for path in stage.iterdir()} != _ARTIFACT_FILES:
            raise RuntimeError("student artifact staging file set is invalid")
        os.replace(stage, destination)
        return pinned
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_student_artifact(
    root: str | Path, *, expected_metadata_sha256: str | None = None,
) -> LoadedStudent:
    """Load only a regular, complete, SHA-verified artifact using weights-only Torch loading."""

    artifact = _safe_directory(root, create=False)
    if {path.name for path in artifact.iterdir()} != _ARTIFACT_FILES:
        raise ValueError("student artifact must contain exactly the required files")
    metadata_bytes = _read_regular_bytes(artifact / "metadata.json", label="artifact metadata")
    if expected_metadata_sha256 is not None:
        if (
            type(expected_metadata_sha256) is not str or len(expected_metadata_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_metadata_sha256)
        ):
            raise ValueError("expected metadata SHA-256 pin is invalid")
        if sha256(metadata_bytes).hexdigest() != expected_metadata_sha256:
            raise ValueError("artifact metadata does not match expected metadata SHA-256 pin")
    metadata_document = _json_document_from_bytes(metadata_bytes, label="artifact metadata")
    metadata = _metadata_from_document(metadata_document)
    weights_bytes = _read_regular_bytes(artifact / "student.pt", label="student weights")
    if sha256(weights_bytes).hexdigest() != metadata.weight_sha256:
        raise ValueError("student weight SHA mismatch")
    metrics_bytes = _read_regular_bytes(artifact / "metrics.json", label="artifact metrics")
    latency_bytes = _read_regular_bytes(artifact / "latency.json", label="artifact latency")
    metrics = _json_document_from_bytes(metrics_bytes, label="artifact metrics")
    latency = _json_document_from_bytes(latency_bytes, label="artifact latency")
    _validate_reports(metrics, latency)
    _validate_provenance(metrics, metadata)
    _metadata_integrity(metadata_document, metrics_bytes, latency_bytes, metrics, metadata)
    model = FingertipMixtureNet(hidden=metadata.hidden)
    try:
        state = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError("student weights could not be safely loaded") from error
    model.load_state_dict(_validate_state(model, state), strict=True)
    model.eval()
    runtime_metrics = {**metrics["metrics"], "production_approved": latency["production_approved"]}
    return LoadedStudent(model=model, metadata=metadata, metrics=runtime_metrics, latency=latency)


def validate_student_artifact(
    root: str | Path, *, expected_metadata_sha256: str | None = None,
) -> None:
    """Run the complete strict load/approval gate without starting a worker."""

    loaded = load_student_artifact(root, expected_metadata_sha256=expected_metadata_sha256)
    if loaded.metrics.get("production_approved") is not True:
        raise ValueError("runtime requires a production_approved student artifact")


__all__ = [
    "RUN_E_AGGREGATE_SHA256", "RUN_E_SHARD_COUNT", "RUN_E_SAMPLE_COUNT",
    "validate_run_e_corpus_identity", "validate_run_e_training_qualification",
    "DISTILLATION_LOSS_CONFIG",
    "DISTILLATION_TRAINING_DEFAULTS",
    "LoadedStudent",
    "default_distillation_training",
    "load_student_artifact",
    "save_student_artifact",
    "validate_student_artifact",
]
