"""CPU contracts for the offline RialTo catalog verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import ObjectInstance


ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_SOURCE = {
    "bottle": "bottle_fixed.usd",
    "cup": "coffeecup.usdz",
    "cube": "box.glb",
    "cylinder": "poly.glb",
}
ALL_CLASSES = ("bottle", "cup", "bowl", "book", "cube", "cylinder")


def _module():
    from scripts import verify_m1_rialto_object_catalog as module

    return module


def _fixture(tmp_path: Path, *, malformed_collision: bool = False):
    asset_root = tmp_path / "rialto"
    source_root = asset_root / "sources"
    converted_root = asset_root / "converted"
    source_root.mkdir(parents=True)
    converted_root.mkdir()
    source_records = {}
    classes = {}
    for object_class in ALL_CLASSES:
        source_name = {
            "bottle": "bottle_fixed.usd",
            "cup": "coffeecup.usdz",
            "bowl": "bowlnrack2.usd",
            "book": "book_fixed.usd",
            "cube": "box.glb",
            "cylinder": "poly.glb",
        }[object_class]
        source = source_root / source_name
        source.write_text(f"source-{object_class}\n", encoding="utf-8")
        resolved_name = {
            "bottle": "bottle_fixed.usd",
            "cup": "coffeecup.usd",
            "bowl": "bowlnrack2.usd",
            "book": "book_fixed.usd",
            "cube": "box.usd",
            "cylinder": "poly.usd",
        }[object_class]
        resolved = (source_root if object_class in {"bottle", "bowl", "book"} else converted_root) / resolved_name
        if object_class in {"bottle", "bowl", "book"}:
            source.write_text('#usda 1.0\n\ndef Xform "Object" {}\n', encoding="utf-8")
        else:
            resolved.write_text('#usda 1.0\n\ndef Xform "Object" {}\n', encoding="utf-8")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        source_records[source_name] = {
            "source_url": f"https://example.invalid/{source_name}",
            "source_sha256": source_sha,
            "resolved_path": f"sources/{source_name}",
        }
        classes[object_class] = {
            "usd_path": str(resolved.relative_to(asset_root)),
            "source_sha256": source_sha,
            "resolved_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "mass_kg": 0.5,
            "scale": 1.0,
            "collision_profile": "invalid" if malformed_collision else "box",
            "grasp_profile": "stable",
        }
    manifest = asset_root / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": "https://example.invalid/rialto.git",
                "revision": "a" * 40,
                "assets": source_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"schema_version": 1, "classes": classes}, indent=2),
        encoding="utf-8",
    )
    return catalog, asset_root


def test_missing_materialized_sources_emit_preparation_required_evidence(tmp_path: Path):
    module = _module()
    catalog = ROOT / "config" / "m1_object_catalog.json"
    asset_root = ROOT / "assets" / "m1_objects" / "rialto"
    evidence = tmp_path / "evidence"

    result = module.verify_catalog(
        catalog,
        asset_root,
        evidence_dir=evidence,
        run_gpu=False,
    )

    assert result["verification_status"] == "preparation_required"
    assert result["hard_gates_passed"] is False
    assert result["offline"]["status"] == "preparation_required"
    assert result["offline"]["missing_assets"]
    assert "m1_rialto_object_assets.py prepare" in result["preparation_command"]
    report = json.loads((evidence / "catalog_verification.json").read_text())
    aggregate = json.loads((evidence / "aggregate.manifest.json").read_text())
    assert report["verification_status"] == "preparation_required"
    assert len(aggregate["files"]) >= 2
    assert len(aggregate["aggregate_sha256"]) == 64


def test_complete_local_catalog_passes_offline_and_records_dependency_hashes(tmp_path: Path):
    module = _module()
    catalog, asset_root = _fixture(tmp_path)

    result = module.verify_catalog(
        catalog,
        asset_root,
        evidence_dir=tmp_path / "evidence",
        run_gpu=False,
    )

    assert result["verification_status"] == "passed"
    assert result["hard_gates_passed"] is True
    assert result["offline"]["status"] == "passed"
    assert set(result["offline"]["classes"]) == set(ALL_CLASSES)
    assert all(
        value["sha256_matches"] and value["dependencies_resolved"]
        for value in result["offline"]["classes"].values()
    )
    assert result["gpu_smoke"]["status"] == "not_requested"


def test_out_of_root_usd_dependency_fails_offline_verification(tmp_path: Path, monkeypatch):
    module = _module()
    catalog, asset_root = _fixture(tmp_path)
    outside = tmp_path / "outside.usd"
    outside.write_text('#usda 1.0\n', encoding="utf-8")
    bottle = asset_root / "sources" / "bottle_fixed.usd"
    bottle.write_text(
        '#usda 1.0\n'
        'def Xform "Root" {\n'
        f'  asset dependency = @{outside}@\n'
        '}\n',
        encoding="utf-8",
    )
    bottle_sha = hashlib.sha256(bottle.read_bytes()).hexdigest()
    manifest_path = asset_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["bottle_fixed.usd"]["source_sha256"] = bottle_sha
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    catalog_payload["classes"]["bottle"]["source_sha256"] = bottle_sha
    catalog_payload["classes"]["bottle"]["resolved_sha256"] = bottle_sha
    catalog.write_text(json.dumps(catalog_payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(module, "_write_evidence", lambda _path, _result: None)
    result = module.verify_catalog(
        catalog,
        asset_root,
        evidence_dir=None,
        run_gpu=False,
    )

    assert result["verification_status"] == "failed"
    assert result["offline"]["status"] == "failed"
    bottle_report = result["offline"]["classes"]["bottle"]
    assert bottle_report["dependencies_resolved"] is False
    assert bottle_report["outside_root_dependencies"] == [str(outside)]
    assert any("outside asset root" in error for error in result["offline"]["errors"])


def test_invalid_collision_metadata_and_duplicate_candidate_ids_fail(tmp_path: Path):
    module = _module()
    catalog, asset_root = _fixture(tmp_path, malformed_collision=True)

    result = module.verify_catalog(
        catalog,
        asset_root,
        evidence_dir=tmp_path / "evidence",
        candidate_instances=(
            ObjectInstance("bottle_000", "bottle", (0.0, 0.0, 0.0)),
            ObjectInstance("bottle_000", "bottle", (0.1, 0.0, 0.0)),
        ),
        run_gpu=False,
    )

    assert result["verification_status"] == "failed"
    assert result["hard_gates_passed"] is False
    assert any("collision" in error for error in result["offline"]["errors"])
    assert any("duplicate object_id" in error for error in result["offline"]["errors"])


def test_target_obstacle_partition_is_explicit_and_deterministic():
    module = _module()
    instances = (
        ObjectInstance("cylinder_000", "cylinder", (0.0, 0.0, 0.0)),
        ObjectInstance("bottle_000", "bottle", (0.0, 0.0, 0.0)),
        ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.0)),
        ObjectInstance("cup_000", "cup", (0.0, 0.0, 0.0)),
    )

    target, obstacles = module.partition_target_and_obstacles(
        instances, "bottle_000"
    )

    assert target is not None and target.object_id == "bottle_000"
    assert tuple(item.object_id for item in obstacles) == (
        "cube_000",
        "cup_000",
        "cylinder_000",
    )


def test_preparation_required_gpu_smoke_never_reports_contact_pass(tmp_path: Path):
    module = _module()
    catalog = ROOT / "config" / "m1_object_catalog.json"
    asset_root = ROOT / "assets" / "m1_objects" / "rialto"

    result = module.verify_catalog(
        catalog,
        asset_root,
        evidence_dir=tmp_path / "evidence",
        run_gpu=True,
    )

    assert result["verification_status"] == "preparation_required"
    assert result["gpu_smoke"]["status"] == "preparation_required"
    assert result["gpu_smoke"]["contact_initialization_passed"] is False


def test_gpu_smoke_uses_interactive_scene_mapping_access():
    source = (
        ROOT / "scripts" / "verify_m1_rialto_object_catalog.py"
    ).read_text(encoding="utf-8")

    assert "scene[instance.object_id]" in source
    assert "scene[name]" in source
    assert "except KeyError:" in source
    assert "getattr(scene" not in source
    assert "hasattr(scene" not in source
