from __future__ import annotations
import hashlib, json, tarfile
from pathlib import Path
import pytest
from scripts.m1_oakink_geometry_bundle import collect_compatible_references, verify_geometry_bundle

def test_collects_unique_sorted_references(tmp_path: Path):
    for name, ref in [("b", "ObjURDF/align_ds/B/x.urdf"), ("a", "ObjURDF/align_ds/A/x.urdf")]:
        p = tmp_path / name; p.mkdir(); (p / "seq_info.json").write_text(json.dumps({"type":"rh", "interaction_mode":"rh_main", "obj_rh_path":ref}), encoding="utf-8")
    assert collect_compatible_references(tmp_path) == ("ObjURDF/align_ds/A/x.urdf", "ObjURDF/align_ds/B/x.urdf")

def test_verify_rejects_wrong_manifest_sha(tmp_path: Path):
    (tmp_path / "geometry_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        verify_geometry_bundle(tmp_path, "0" * 64)

def test_verify_rejects_wrong_count(tmp_path: Path):
    manifest = {"schema_version":1,"upstream": {"repository":"kelvin34501/OakInk-v2","revision":"21705616140d726607027e70d58b7837f442ffd8","archive_sha256":"40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2"},"entries":[]}
    p = tmp_path / "geometry_manifest.json"; p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        verify_geometry_bundle(tmp_path, hashlib.sha256(p.read_bytes()).hexdigest(), 90)
