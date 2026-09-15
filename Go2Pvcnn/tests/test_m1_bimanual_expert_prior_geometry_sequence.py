from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.dexmanipnet import (
    audit_sequence,
    load_best_successful_rollout,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.geometry_overlay import (
    verify_geometry_overlay,
)
from test_m1_bimanual_expert_prior_dexmanipnet import write_minimal_sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sequence_and_overlay(tmp_path: Path):
    sequence = write_minimal_sequence(
        tmp_path / "sequence", source="oakinkv2", include_geometry=False
    )
    reference = "ObjURDF/align_ds/object/model.urdf"
    info_path = sequence / "seq_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["obj_rh_path"] = reference
    info["obj_lh_path"] = reference
    info_path.write_text(json.dumps(info), encoding="utf-8")

    overlay = tmp_path / "overlay"
    mesh = overlay / "objects/mesh.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    urdf = overlay / "objects/object.urdf"
    urdf.write_text(
        '<robot name="object"><link name="base"><visual><origin xyz="0 0 0" '
        'rpy="0 0 0"/><geometry><mesh filename="mesh.obj" scale="1 1 1"/>'
        '</geometry></visual><collision><origin xyz="0 0 0" rpy="0 0 0"/>'
        '<geometry><mesh filename="mesh.obj" scale="1 1 1"/></geometry>'
        '</collision></link></robot>',
        encoding="utf-8",
    )
    manifest = overlay / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "upstream": {
                    "repository": "kelvin34501/OakInk-v2",
                    "revision": "21705616140d726607027e70d58b7837f442ffd8",
                    "archive_sha256": "40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2",
                },
                "recipe": {"tool": "fixture", "version": "1", "arguments": {"seed": 1}},
                "entries": [{
                    "reference": reference,
                    "raw_member": "object_raw/align_ds/object.obj",
                    "raw_sha256": "a" * 64,
                    "urdf": "objects/object.urdf",
                    "urdf_sha256": _sha256(urdf),
                    "meshes": [{"path": "objects/mesh.obj", "sha256": _sha256(mesh)}],
                    "checks": {"fixture": True},
                }],
            }
        ),
        encoding="utf-8",
    )
    resolver = verify_geometry_overlay(overlay, manifest, _sha256(manifest))
    return sequence, resolver, urdf


def test_missing_oakink_geometry_uses_same_verified_overlay(sequence_and_overlay):
    sequence, resolver, expected_urdf = sequence_and_overlay

    audit = audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver)

    assert audit.accepted
    loaded = load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)
    assert loaded.object_geometry_path == expected_urdf
    assert loaded.source_sha256 == audit.input_sha256


def test_missing_oakink_geometry_without_overlay_still_rejects(sequence_and_overlay):
    sequence, _, _ = sequence_and_overlay

    assert audit_sequence(sequence, "oakinkv2", "rh").reason == "missing_object_geometry"
    with pytest.raises(ValueError, match="missing_object_geometry"):
        load_best_successful_rollout(sequence, "oakinkv2", "rh")


@pytest.mark.parametrize("mutation", ["unknown_reference", "byte_drift"])
def test_unverified_overlay_resolution_rejects_audit_and_load(sequence_and_overlay, mutation):
    sequence, resolver, urdf = sequence_and_overlay
    if mutation == "unknown_reference":
        info_path = sequence / "seq_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["obj_rh_path"] = "ObjURDF/align_ds/unknown/model.urdf"
        info_path.write_text(json.dumps(info), encoding="utf-8")
    else:
        urdf.write_text("byte drift", encoding="utf-8")

    audit = audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver)

    assert audit.reason == "invalid_object_geometry"
    with pytest.raises(ValueError, match="invalid_object_geometry"):
        load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)


@pytest.mark.parametrize("reference", ["/absolute.urdf", "../escape.urdf", "a\\b.urdf", "a//b.urdf", "./a.urdf"])
def test_invalid_oakink_reference_cannot_be_overridden(sequence_and_overlay, reference):
    sequence, resolver, _ = sequence_and_overlay
    info_path = sequence / "seq_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["obj_rh_path"] = reference
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver).reason == "invalid_object_geometry"
    with pytest.raises(ValueError, match="invalid_object_geometry"):
        load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)


def test_source_symlink_cannot_be_overridden(sequence_and_overlay):
    sequence, resolver, _ = sequence_and_overlay
    source_urdf = sequence.parents[1] / "ObjURDF/align_ds/object/model.urdf"
    source_urdf.parent.mkdir(parents=True)
    source_urdf.symlink_to(sequence / "seq_info.json")

    assert audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver).reason == "invalid_object_geometry"
    with pytest.raises(ValueError, match="invalid_object_geometry"):
        load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)


def test_existing_oakink_geometry_wins_over_overlay(sequence_and_overlay):
    sequence, resolver, overlay_urdf = sequence_and_overlay
    source_urdf = sequence.parents[1] / "ObjURDF/align_ds/object/model.urdf"
    source_urdf.parent.mkdir(parents=True)
    source_urdf.write_text('<robot name="source"/>', encoding="utf-8")

    loaded = load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)

    assert loaded.object_geometry_path == source_urdf
    assert loaded.object_geometry_path != overlay_urdf


def test_favor_is_identical_with_a_verified_overlay(sequence_and_overlay, tmp_path: Path):
    _, resolver, _ = sequence_and_overlay
    sequence = write_minimal_sequence(tmp_path / "favor", source="favor")

    default = load_best_successful_rollout(sequence, "favor", "rh")
    with_overlay = load_best_successful_rollout(sequence, "favor", "rh", geometry_resolver=resolver)

    assert with_overlay.source_sha256 == default.source_sha256
    assert with_overlay.object_geometry_path == default.object_geometry_path
    for field in ("q", "dq", "root_state", "object_state", "tip_force"):
        assert np.array_equal(getattr(with_overlay, field), getattr(default, field))


def test_invalid_resolver_type_raises_before_sequence_work(tmp_path: Path):
    missing = tmp_path / "missing-sequence"

    with pytest.raises(ValueError, match="geometry_resolver"):
        audit_sequence(missing, "oakinkv2", "rh", geometry_resolver=object())
    with pytest.raises(ValueError, match="geometry_resolver"):
        load_best_successful_rollout(missing, "oakinkv2", "rh", geometry_resolver=object())


def test_overlay_does_not_bypass_source_or_interaction_gates(sequence_and_overlay):
    sequence, resolver, _ = sequence_and_overlay
    info_path = sequence / "seq_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["interaction_mode"] = "lh_main"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver).reason == "interaction_mode_side_mismatch"
    assert audit_sequence(sequence, "unknown", "rh", geometry_resolver=resolver).reason == "unsupported_source"
