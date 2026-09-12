from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import json
from hashlib import sha256
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import (
    load_student_artifact,
    save_student_artifact,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    FINGER_ORDER,
    LEFT_REFLECTION,
    MODEL_INPUT_FIELD_ORDER,
    MIXTURE_OUTPUT_AXIS_ORDER,
    PHASE_ORDER,
    PRIOR_DT,
    StudentArtifactMetadata,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import FingertipMixtureNet


def _metadata(*, weight_sha256: str = "0" * 64) -> StudentArtifactMetadata:
    return StudentArtifactMetadata(
        format_version=1,
        input_dim=42,
        mixture_components=4,
        horizon=20,
        dt=PRIOR_DT,
        finger_order=FINGER_ORDER,
        phase_order=PHASE_ORDER,
        mirror_matrix=torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
        dataset_aggregate_sha256="a" * 64,
        teacher_ensemble_manifest_sha256="b" * 64,
        teacher_seed=1701,
        distillation_seed=42,
        code_commit="c" * 40,
        weight_sha256=weight_sha256,
        hidden=(16, 16),
        input_field_order=MODEL_INPUT_FIELD_ORDER,
        output_axis_order=MIXTURE_OUTPUT_AXIS_ORDER,
    )


def _write(root: Path) -> Path:
    torch.manual_seed(1)
    save_student_artifact(
        root,
        model=FingertipMixtureNet(hidden=(16, 16)),
        metadata=_metadata(),
        metrics={
            "student_nll": 1.0,
            "teacher_nll": 1.0,
            "nll_delta_per_dim": 0.0,
            "first_step_velocity_rmse": 0.8,
            "first_step_zero_rmse": 1.0,
            "first_step_improvement": 0.2,
            "endpoint_rmse": 0.8,
            "teacher_endpoint_rmse": 0.8,
            "endpoint_zero_rmse": 1.0,
            "endpoint_improvement": 0.2,
            "production_approved": False,
            "deterministic_repeat_verified": True,
        },
        latency={"warmups": 100, "measurements": 1000, "p99_ms": 1.0},
        provenance={
            "nonproduction_synthetic": True,
            "dataset_aggregate_sha256": "a" * 64,
            "teacher_ensemble_manifest_sha256": "b" * 64,
        },
    )
    return root


def test_artifact_round_trip_is_eval_only_and_hash_pinned(tmp_path: Path):
    loaded = load_student_artifact(_write(tmp_path / "artifact"))

    assert not loaded.model.training
    assert loaded.metadata.hidden == (16, 16)
    assert loaded.metrics["production_approved"] is False


def test_artifact_rejects_weight_tampering_before_deserialization(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    weights = root / "student.pt"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="weight SHA"):
        load_student_artifact(root)


def test_artifact_rejects_extra_file_and_metadata_self_hash_tampering(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    (root / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_student_artifact(root)

    (root / "unexpected.txt").unlink()
    metadata = root / "metadata.json"
    metadata.write_text(metadata.read_text(encoding="utf-8").replace('"hidden":[16,16]', '"hidden":[17,16]'), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata SHA"):
        load_student_artifact(root)


def test_artifact_rejects_metrics_provenance_that_does_not_match_frozen_pins(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    report = root / "metrics.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    document["provenance"]["dataset_aggregate_sha256"] = "d" * 64
    report.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        load_student_artifact(root)


def _rewrite_metadata(root: Path, mutate) -> None:
    path = root / "metadata.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    body = dict(document)
    body.pop("metadata_sha256", None)
    document["metadata_sha256"] = sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def test_artifact_requires_metadata_self_hash_before_trusting_weights(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    path = root / "metadata.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["metadata_sha256"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata SHA"):
        load_student_artifact(root)


def test_artifact_integrity_binds_reports_and_recomputes_nonproduction_approval(tmp_path: Path, monkeypatch):
    root = _write(tmp_path / "artifact")
    metrics_path = root / "metrics.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document["metrics"]["production_approved"] = True
    metrics_path.write_text(json.dumps(document), encoding="utf-8")

    calls = 0
    def forbidden_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("report tampering reached torch.load")
    monkeypatch.setattr("go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact.torch.load", forbidden_load)
    with pytest.raises(ValueError, match="metrics SHA"):
        load_student_artifact(root)
    assert calls == 0

    # A coordinated metadata rewrite cannot make synthetic provenance production-approved either.
    _rewrite_metadata(root, lambda metadata: metadata.update({
        "metrics_bytes": len(metrics_path.read_bytes()),
        "metrics_sha256": sha256(metrics_path.read_bytes()).hexdigest(),
    }))
    with pytest.raises(ValueError, match="production approval"):
        load_student_artifact(root)


def test_artifact_rejects_missing_report_binding_and_exposes_deterministic_fingerprint(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    document = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(document.get("reproducibility_fingerprint"), str)
    _rewrite_metadata(root, lambda metadata: metadata.pop("metrics_sha256", None))

    with pytest.raises(ValueError, match="metadata fields"):
        load_student_artifact(root)


def test_artifact_rejects_bool_or_incoherent_derived_metric_fields(tmp_path: Path):
    root = _write(tmp_path / "artifact")
    report = root / "metrics.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    document["metrics"]["endpoint_rmse"] = True
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_student_artifact(root)
