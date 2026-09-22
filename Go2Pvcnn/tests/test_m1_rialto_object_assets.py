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


def _write_glb_fixture(path: Path) -> None:
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(2.0, 4.0, 6.0)),
        geom_name="base",
        node_name="base",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
        transform=trimesh.transformations.translation_matrix((3.0, 0.0, 0.0)),
        geom_name="offset",
        node_name="offset",
    )
    payload = scene.export(file_type="glb")
    assert isinstance(payload, (bytes, bytearray))
    path.write_bytes(bytes(payload))


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


def test_fetch_sources_accepts_matching_local_hash_and_rejects_stale(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "sources" / "sample.usd"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"matching")
    manifest = {
        "repo": "https://example.invalid/rialto.git",
        "revision": "a" * 40,
        "assets": {
            "sample.usd": {
                "source_url": "https://example.invalid/sample.usd",
                "source_sha256": hashlib.sha256(b"matching").hexdigest(),
                "resolved_path": "sources/sample.usd",
            }
        },
    }
    monkeypatch.setattr(module, "load_manifest", lambda: manifest)
    assert module.fetch_sources(tmp_path) == {"sample.usd": source}
    source.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="network.*explicit.*sample.usd"):
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


def test_glb_conversion_writes_deterministic_combined_usda(tmp_path):
    module = _module()
    source = tmp_path / "box.glb"
    _write_glb_fixture(source)
    first = module.convert_usdz_or_glb(source, tmp_path / "one")
    second = module.convert_usdz_or_glb(source, tmp_path / "two")
    assert first.name == second.name == "box.usd"
    assert first.read_bytes() == second.read_bytes()
    assert module.sha256_file(first) == "d560385cce0956598b656f2139e8ea4cd15ed98b184133dbb97163cbccc33d55"
    text = first.read_text(encoding="utf-8")
    assert 'def Mesh "CombinedMesh"' in text
    assert "point3f[] points" in text
    assert "int[] faceVertexIndices" in text
    assert "int[] faceVertexCounts" in text
    inspection = module.inspect_usd(first)
    assert inspection["prim_count"] == 2
    assert inspection["bounds"] == [[-1.0, -2.0, -3.0], [3.5, 2.0, 3.0]]
    assert inspection["dependencies_resolved"] is True


def test_malformed_glb_fails_clearly_without_publishing_output(tmp_path):
    module = _module()
    source = tmp_path / "malformed.glb"
    source.write_bytes(b"glTF\x02\x00\x00\x00")
    output_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="cannot convert GLB.*failed to load"):
        module.convert_usdz_or_glb(source, output_dir)
    assert not (output_dir / "malformed.usd").exists()


def test_malformed_reconversion_preserves_existing_output_without_temp_leftovers(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "object.glb"
    source.write_bytes(b"glTF\x02\x00\x00\x00")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output = output_dir / "object.usd"
    original = b"#usda 1.0\n# previously validated output\n"
    output.write_bytes(original)

    def write_partial_then_fail(_source, destination):
        destination.write_bytes(b"#usda 1.0\n# malformed replacement\n")
        raise RuntimeError("malformed conversion")

    monkeypatch.setattr(module, "_convert_with_available_tool", write_partial_then_fail)
    with pytest.raises(RuntimeError, match="malformed conversion"):
        module.convert_usdz_or_glb(source, output_dir)

    assert output.read_bytes() == original
    assert list(output_dir.iterdir()) == [output]


def test_conversion_rejects_unresolved_usd_dependencies(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "Cup.USDZ"
    source.write_bytes(b"fixture")

    def write_unresolved(_source, destination):
        destination.write_text(
            '#usda 1.0\n'
            'def Xform "Root" {\n'
            '  asset dependency = @missing.usd@\n'
            '}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "_convert_with_available_tool", write_unresolved)
    with pytest.raises(RuntimeError, match="unresolved.*dependenc"):
        module.convert_usdz_or_glb(source, tmp_path / "out")


def test_inspect_binary_usdc_fails_without_pxr_instead_of_decoding_text(tmp_path, monkeypatch):
    module = _module()
    path = tmp_path / "binary.usd"
    path.write_bytes(b"PXR-USDC\x00\xff\x00\x01")
    monkeypatch.setattr(module, "_inspect_with_pxr", lambda _path: None)
    with pytest.raises(RuntimeError, match="binary.*PXR-USDC.*pxr"):
        module.inspect_usd(path)


def test_inspect_usd_detects_missing_file_dependency(tmp_path, monkeypatch):
    module = _module()
    path = tmp_path / "sample.usda"
    path.write_text(
        '#usda 1.0\n'
        'def Xform "Root" {\n'
        '  asset dependency = @missing.usd@\n'
        '}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_inspect_with_pxr", lambda _path: None)
    result = module.inspect_usd(path)
    assert result["dependencies_resolved"] is False
    assert result["missing_dependencies"] == ["missing.usd"]


def test_prepare_records_generated_usd_sha256_in_metadata(tmp_path, monkeypatch):
    module = _module()
    destination = tmp_path / "rialto"
    source = destination / "sources" / "box.glb"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"glTF")

    monkeypatch.setattr(
        module,
        "fetch_sources",
        lambda _destination, allow_network=False: {"box.glb": source},
    )
    monkeypatch.setattr(
        module,
        "_convert_with_available_tool",
        lambda _source, output: output.write_text("#usda 1.0\n", encoding="utf-8"),
    )
    result = module.prepare_assets(destination)
    prepared_manifest = destination / "prepared_manifest.json"
    payload = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    record = payload["assets"]["box.glb"]
    generated = destination / record["usd_path"]
    expected = hashlib.sha256(generated.read_bytes()).hexdigest()
    assert record["generated_usd_sha256"] == expected
    assert result["output_metadata"]["box.glb"]["generated_usd_sha256"] == expected


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
