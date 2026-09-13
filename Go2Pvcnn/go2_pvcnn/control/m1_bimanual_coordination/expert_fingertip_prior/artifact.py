"""Strict, SHA-pinned serialization for the deployable fingertip student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .contracts import PriorPhase, StudentArtifactMetadata
from .model import FingertipMixtureNet


_ARTIFACT_FILES = frozenset({"metadata.json", "student.pt", "metrics.json", "latency.json"})
_DISTILLATION_CONFIG = {"label_nll_weight": 0.5, "teacher_sample_weight": 0.5, "acceleration_weight": 1e-5, "jerk_weight": 1e-7}


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
    required = set(_metadata_document(_metadata_for_schema()).keys()) | {"metrics_sha256", "metrics_bytes", "latency_sha256", "latency_bytes", "distillation_config", "reproducibility_fingerprint"}
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


def _json_document(path: Path, *, label: str) -> dict[str, object]:
    _regular(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
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
    required_values = {"student_nll", "teacher_nll", "nll_delta_per_dim", "first_step_velocity_rmse", "first_step_zero_rmse", "first_step_improvement", "endpoint_rmse", "teacher_endpoint_rmse", "endpoint_zero_rmse", "endpoint_improvement", "production_approved", "deterministic_repeat_verified"}
    if set(values) != required_values or type(values.get("production_approved")) is not bool or not _finite_json(metrics):
        raise ValueError("artifact metrics/provenance is invalid")
    numeric = required_values - {"production_approved", "deterministic_repeat_verified"}
    if type(values["deterministic_repeat_verified"]) is not bool or any(type(values[key]) not in (int, float) or type(values[key]) is bool or not math.isfinite(float(values[key])) for key in numeric):
        raise ValueError("artifact metrics are invalid")
    if values["deterministic_repeat_verified"] is not True or any(float(values[key]) < 0.0 for key in ("first_step_velocity_rmse", "endpoint_rmse", "teacher_endpoint_rmse")) or any(float(values[key]) <= 0.0 for key in ("first_step_zero_rmse", "endpoint_zero_rmse")) or not math.isclose(values["nll_delta_per_dim"], (values["student_nll"] - values["teacher_nll"]) / 300.0, abs_tol=1e-8) or not math.isclose(values["first_step_improvement"], 1 - values["first_step_velocity_rmse"] / values["first_step_zero_rmse"], abs_tol=1e-8) or not math.isclose(values["endpoint_improvement"], 1 - values["endpoint_rmse"] / values["endpoint_zero_rmse"], abs_tol=1e-8):
        raise ValueError("artifact derived metrics are inconsistent")
    if set(provenance) != {"nonproduction_synthetic", "dataset_aggregate_sha256", "teacher_ensemble_manifest_sha256"} or type(provenance.get("nonproduction_synthetic")) is not bool:
        raise ValueError("artifact provenance is invalid")
    if set(latency) != {"warmups", "measurements", "p99_ms"}:
        raise ValueError("artifact latency schema is invalid")
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


def _metadata_integrity(document: dict[str, object], metrics_bytes: bytes, latency_bytes: bytes, reports: dict[str, object], metadata: StudentArtifactMetadata) -> None:
    if document.get("distillation_config") != _DISTILLATION_CONFIG:
        raise ValueError("artifact distillation configuration is invalid")
    for label, raw in (("metrics", metrics_bytes), ("latency", latency_bytes)):
        if document.get(f"{label}_sha256") != sha256(raw).hexdigest() or document.get(f"{label}_bytes") != len(raw):
            raise ValueError(f"artifact {label} SHA mismatch")
    values, provenance = reports["metrics"], reports["provenance"]
    expected = (not provenance["nonproduction_synthetic"] and values["nll_delta_per_dim"] <= 0.05 and values["endpoint_rmse"] <= 1.05 * values["teacher_endpoint_rmse"] and values["first_step_improvement"] >= 0.10 and values["endpoint_improvement"] >= 0.10 and json.loads(latency_bytes)["p99_ms"] < 2.0)
    if values["production_approved"] != expected:
        raise ValueError("artifact production approval does not match gates")
    fingerprint_body = {"metadata": _metadata_document(metadata), "distillation_config": _DISTILLATION_CONFIG, "metrics": reports}
    if document.get("reproducibility_fingerprint") != sha256(_canonical_json(fingerprint_body)).hexdigest():
        raise ValueError("artifact reproducibility fingerprint mismatch")


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
) -> StudentArtifactMetadata:
    """Atomically save exactly the four self-validating student artifact files."""

    if not isinstance(metadata, StudentArtifactMetadata):
        raise TypeError("metadata must be StudentArtifactMetadata")
    destination = Path(root).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"student artifact destination already exists: {destination}")
    reports = {"metrics": dict(metrics), "provenance": dict(provenance)}
    latency_document = dict(latency)
    _validate_reports(reports, latency_document)
    _validate_provenance(reports, metadata)
    state = _state_for_save(model)
    if model.hidden != metadata.hidden:
        raise ValueError("student model architecture does not match metadata hidden widths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        weights_path = stage / "student.pt"
        with weights_path.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        pinned = replace(metadata, weight_sha256=sha256_file(weights_path))
        metrics_bytes, latency_bytes = _canonical_json(reports), _canonical_json(latency_document)
        metadata_body = {**_metadata_document(pinned), "metrics_sha256": sha256(metrics_bytes).hexdigest(), "metrics_bytes": len(metrics_bytes), "latency_sha256": sha256(latency_bytes).hexdigest(), "latency_bytes": len(latency_bytes), "distillation_config": _DISTILLATION_CONFIG}
        metadata_body["reproducibility_fingerprint"] = sha256(_canonical_json({"metadata": _metadata_document(pinned), "distillation_config": _DISTILLATION_CONFIG, "metrics": reports})).hexdigest()
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


def load_student_artifact(root: str | Path) -> LoadedStudent:
    """Load only a regular, complete, SHA-verified artifact using weights-only Torch loading."""

    artifact = Path(root).resolve()
    if not artifact.is_dir() or artifact.is_symlink():
        raise ValueError("student artifact root must be a regular directory")
    if {path.name for path in artifact.iterdir()} != _ARTIFACT_FILES:
        raise ValueError("student artifact must contain exactly the required files")
    metadata_document = _json_document(artifact / "metadata.json", label="artifact metadata")
    metadata = _metadata_from_document(metadata_document)
    weights_path = artifact / "student.pt"
    _regular(weights_path, label="student weights")
    if sha256_file(weights_path) != metadata.weight_sha256:
        raise ValueError("student weight SHA mismatch")
    metrics_path, latency_path = artifact / "metrics.json", artifact / "latency.json"
    metrics = _json_document(metrics_path, label="artifact metrics")
    latency = _json_document(latency_path, label="artifact latency")
    _validate_reports(metrics, latency)
    _validate_provenance(metrics, metadata)
    _metadata_integrity(metadata_document, metrics_path.read_bytes(), latency_path.read_bytes(), metrics, metadata)
    model = FingertipMixtureNet(hidden=metadata.hidden)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError("student weights could not be safely loaded") from error
    model.load_state_dict(_validate_state(model, state), strict=True)
    model.eval()
    return LoadedStudent(model=model, metadata=metadata, metrics=metrics["metrics"], latency=latency)


def validate_student_artifact(root: str | Path) -> None:
    """Run the complete strict load/approval gate without starting a worker."""

    loaded = load_student_artifact(root)
    if loaded.metrics.get("production_approved") is not True:
        raise ValueError("runtime requires a production_approved student artifact")


__all__ = [
    "LoadedStudent",
    "load_student_artifact",
    "save_student_artifact",
    "validate_student_artifact",
]
