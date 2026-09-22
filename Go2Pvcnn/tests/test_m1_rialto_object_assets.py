"""Contract tests for the pinned RialTo object source utility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "m1_objects" / "rialto"
MANIFEST = ASSET_ROOT / "source_manifest.json"


def _module():
    from scripts import m1_rialto_object_assets

    return m1_rialto_object_assets


def test_manifest_has_pinned_schema_and_explicit_sources():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["repo"] == "https://github.com/real-to-sim-to-real/RialToAssets.git"
    assert len(manifest["revision"]) == 40
    expected = {
        "bottle_fixed.usd",
        "coffeecup.usdz",
        "bowlnrack2.usd",
        "book_fixed.usd",
        "box.glb",
        "poly.glb",
    }
    assert set(manifest["assets"]) == expected
    for name, record in manifest["assets"].items():
        assert record["source_url"].endswith(f"/objects/{name}")
        assert len(record["source_sha256"]) == 64
        assert record["resolved_path"] == f"sources/{name}"


def test_sha256_file_records_content_digest(tmp_path):
    module = _module()
    source = tmp_path / "payload.bin"
    source.write_bytes(b"rialto")
    assert module.sha256_file(source) == hashlib.sha256(b"rialto").hexdigest()


def test_fetch_sources_requires_explicit_network_opt_in(tmp_path):
    module = _module()
    with pytest.raises(RuntimeError, match="network.*explicit"):
        module.fetch_sources(tmp_path)


def test_unsupported_conversion_extension_has_clear_error(tmp_path):
    module = _module()
    source = tmp_path / "object.obj"
    source.write_bytes(b"not supported")
    with pytest.raises(ValueError, match="unsupported.*extension"):
        module.convert_usdz_or_glb(source, tmp_path / "out")


def test_conversion_output_name_is_deterministic(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "Cup.USDZ"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        module,
        "_convert_with_available_tool",
        lambda _source, destination: destination.write_text("#usda 1.0\n", encoding="utf-8"),
    )
    first = module.convert_usdz_or_glb(source, tmp_path / "one")
    second = module.convert_usdz_or_glb(source, tmp_path / "two")
    assert first.name == second.name == "Cup.usd"
    assert first.parent.name == "one"
    assert second.parent.name == "two"


def test_inspect_usd_reports_prims_bounds_and_missing_dependencies(tmp_path):
    module = _module()
    path = tmp_path / "sample.usda"
    path.write_text(
        '#usda 1.0\n'
        'def Xform "Root" {\n'
        '  def Mesh "Mesh" {\n'
        '    float3[] extent = [(-1, -2, -3), (4, 5, 6)]\n'
        '  }\n'
        '  rel material:binding = </Missing>\n'
        '}\n',
        encoding="utf-8",
    )
    result = module.inspect_usd(path)
    assert result["prim_count"] == 2
    assert result["bounds"] == [[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]
    assert result["dependencies_resolved"] is True
