"""Contract tests for the validated M1 RialTo object catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    AssetPreparationRequiredError,
    ObjectInstance,
    load_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "m1_object_catalog.json"
CLASS_NAMES = ("bottle", "cup", "bowl", "book", "cube", "cylinder")


def _write_catalog(
    tmp_path: Path,
    *,
    path_by_class: dict[str, str] | None = None,
    resolved_sha_by_class: dict[str, str] | None = None,
    source_sha_by_class: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    path_by_class = path_by_class or {
        name: f"{name}.usd" for name in CLASS_NAMES
    }
    resolved_sha_by_class = resolved_sha_by_class or {}
    source_sha_by_class = source_sha_by_class or {}
    classes: dict[str, dict[str, object]] = {}
    for name in CLASS_NAMES:
        relative = Path(path_by_class[name])
        if not relative.is_absolute() and ".." not in relative.parts:
            asset_path = asset_root / relative
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(f"{name}-asset".encode())
        digest = hashlib.sha256(
            (asset_root / relative).read_bytes()
        ).hexdigest() if (asset_root / relative).is_file() else "0" * 64
        classes[name] = {
            "usd_path": path_by_class[name],
            "source_sha256": source_sha_by_class.get(name, digest),
            "resolved_sha256": resolved_sha_by_class.get(name, digest),
            "mass_kg": 0.5,
            "scale": 1.0,
            "collision_profile": "convex_decomposition",
            "grasp_profile": "two_hand_stable",
        }
    classes["cylinder"]["geometry_note"] = (
        "poly.glb is retained as a polygonal cylinder candidate after inspection."
    )
    config = tmp_path / "catalog.json"
    config.write_text(
        json.dumps({"schema_version": 1, "classes": classes}, indent=2),
        encoding="utf-8",
    )
    return config, asset_root


def test_committed_catalog_declares_all_six_classes_and_cylinder_note() -> None:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert tuple(payload["classes"]) == CLASS_NAMES
    for object_class in CLASS_NAMES:
        record = payload["classes"][object_class]
        assert "sha256" not in record
        assert len(record["source_sha256"]) == 64
        if object_class in {"cup", "cube", "cylinder"}:
            assert record["resolved_sha256"] is None
        else:
            assert len(record["resolved_sha256"]) == 64
        assert record["usd_path"].endswith(".usd")
    assert "geometry_note" in payload["classes"]["cylinder"]
    assert "poly.glb" in payload["classes"]["cylinder"]["geometry_note"]


def test_load_catalog_resolves_all_six_classes_and_verifies_hashes(tmp_path: Path) -> None:
    config, asset_root = _write_catalog(tmp_path)

    catalog = load_catalog(config, asset_root)

    assert tuple(catalog.classes) == CLASS_NAMES
    for object_class in CLASS_NAMES:
        record = catalog.resolve(object_class)
        assert record.object_class == object_class
        assert record.usd_path.is_relative_to(asset_root)
        assert record.source_sha256 == record.resolved_sha256
        assert record.sha256 == record.resolved_sha256
        assert record.mass_kg == pytest.approx(0.5)
        assert record.scale == pytest.approx(1.0)


def test_load_catalog_validates_materialized_usd_against_resolved_sha_only(
    tmp_path: Path,
) -> None:
    config, asset_root = _write_catalog(
        tmp_path,
        path_by_class={
            "cup": "converted/coffeecup.usd",
            **{name: f"{name}.usd" for name in CLASS_NAMES if name != "cup"},
        },
        source_sha_by_class={"cup": "1" * 64},
    )

    catalog = load_catalog(config, asset_root)

    record = catalog.resolve("cup")
    assert record.source_sha256 == "1" * 64
    assert record.resolved_sha256 == hashlib.sha256(b"cup-asset").hexdigest()
    assert record.source_sha256 != record.resolved_sha256


def test_committed_catalog_missing_assets_require_explicit_preparation() -> None:
    asset_root = ROOT / "assets" / "m1_objects" / "rialto"

    with pytest.raises(AssetPreparationRequiredError) as error:
        load_catalog(DEFAULT_CONFIG, asset_root)

    message = str(error.value)
    assert "bottle" in message
    assert "sources/bottle_fixed.usd" in message
    assert "m1_rialto_object_assets.py prepare" in message
    assert str(asset_root) in message


def test_load_catalog_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path / "outside.usd"
    outside.write_bytes(b"outside")
    config, asset_root = _write_catalog(
        tmp_path,
        path_by_class={"bottle": "../outside.usd", **{
            name: f"{name}.usd" for name in CLASS_NAMES if name != "bottle"
        }},
    )

    with pytest.raises(ValueError, match="escapes|travers"):
        load_catalog(config, asset_root)


def test_load_catalog_rejects_sha_mismatch(tmp_path: Path) -> None:
    config, asset_root = _write_catalog(
        tmp_path,
        resolved_sha_by_class={"cup": "f" * 64},
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch.*cup"):
        load_catalog(config, asset_root)


def test_validate_instances_rejects_duplicate_ids_and_sorts_deterministically(
    tmp_path: Path,
) -> None:
    config, asset_root = _write_catalog(tmp_path)
    catalog = load_catalog(config, asset_root)
    instances = (
        ObjectInstance("cup_001", "cup", (0.0, 0.0, 0.0, 1.0)),
        ObjectInstance("bottle_002", "bottle", (0.0, 0.0, 0.0, 1.0)),
        ObjectInstance("bottle_001", "bottle", (0.0, 0.0, 0.0, 1.0)),
    )

    ordered = catalog.validate_instances(instances)

    assert tuple(instance.object_id for instance in ordered) == (
        "bottle_001",
        "bottle_002",
        "cup_001",
    )
    with pytest.raises(ValueError, match="duplicate object_id"):
        catalog.validate_instances(instances + (instances[0],))


def test_missing_catalog_uses_legacy_box_fallback(tmp_path: Path) -> None:
    catalog = load_catalog(None, tmp_path)

    assert catalog.uses_legacy_box is True
    assert catalog.classes == ()
    assert catalog.validate_instances(()) == ()
