from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


_NORMALIZER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "normalize_o6_assets.py"
)
_NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "m1_o6_normalizer", _NORMALIZER_PATH
)
assert _NORMALIZER_SPEC is not None and _NORMALIZER_SPEC.loader is not None
_NORMALIZER_MODULE = importlib.util.module_from_spec(_NORMALIZER_SPEC)
_NORMALIZER_SPEC.loader.exec_module(_NORMALIZER_MODULE)
normalize_o6_sources = _NORMALIZER_MODULE.normalize_o6_sources


_PACKAGE_NAMES = {
    "left": "1、O6，urdf/linkerhand_O6_left.urdf",
    "right": "1、O6，urdf/linkerhand_O6_right.urdf",
}


def _make_source_tree(root: Path) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for side, package_name in _PACKAGE_NAMES.items():
        package_root = root / package_name
        usd_root = package_root / f"linkerhand_O6_{side}.urdf"
        configuration = usd_root / "configuration"
        configuration.mkdir(parents=True)
        meshes = package_root / "meshes"
        meshes.mkdir()

        entry = configuration / f"O6_{side}.usd" if side == "left" else usd_root / "O6_right.usd"
        entry.write_bytes(
            b"#usda 1.0\n( subLayers = [@configuration/layer.usd@] )\n"
        )
        layer_paths = {}
        for layer in ("base", "physics", "robot", "sensor"):
            layer_path = configuration / f"linkerhand_O6_{side}.urdf_{layer}.usd"
            layer_path.write_bytes(f"#usda 1.0\n# {side}-{layer}\n".encode())
            layer_paths[layer] = layer_path
        mesh = meshes / "finger.STL"
        mesh.write_bytes(f"solid {side}\nendsolid {side}\n".encode())
        paths[side] = {"entry": entry, "mesh": mesh, **layer_paths}
    return paths


def test_normalizer_places_both_entries_above_configuration_and_hashes_every_file(tmp_path):
    source_root = tmp_path / "source"
    source_paths = _make_source_tree(source_root)
    destination = tmp_path / "destination"

    manifest = normalize_o6_sources(source_root, destination)

    assert (destination / "o6_left/O6_left.usd").is_file()
    assert (destination / "o6_right/O6_right.usd").is_file()
    assert manifest["left"]["entry"] == "o6_left/O6_left.usd"
    assert manifest["right"]["entry"] == "o6_right/O6_right.usd"
    assert set(manifest["left"]["layers"]) == {"base", "physics", "robot", "sensor"}
    assert set(manifest["right"]["layers"]) == {"base", "physics", "robot", "sensor"}
    copied_files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "source_manifest.json"
    )
    assert sorted(manifest["sha256"]) == copied_files
    assert all(len(digest) == 64 for digest in manifest["sha256"].values())
    assert manifest["sha256"]["o6_left/meshes/finger.STL"] == hashlib.sha256(
        source_paths["left"]["mesh"].read_bytes()
    ).hexdigest()
    assert (destination / "source_manifest.json").is_file()


def test_normalizer_is_lossless_and_deterministic(tmp_path):
    source_root = tmp_path / "source"
    source_paths = _make_source_tree(source_root)
    destination = tmp_path / "destination"

    first = normalize_o6_sources(source_root, destination)
    second = normalize_o6_sources(source_root, destination)

    assert first == second
    for side in ("left", "right"):
        assert (destination / f"o6_{side}/O6_{side}.usd").read_bytes() == source_paths[side][
            "entry"
        ].read_bytes()
        assert (destination / f"o6_{side}/meshes/finger.STL").read_bytes() == source_paths[
            side
        ]["mesh"].read_bytes()


def test_normalizer_rejects_missing_required_layer(tmp_path):
    source_root = tmp_path / "source"
    source_paths = _make_source_tree(source_root)
    source_paths["left"]["physics"].unlink()

    with pytest.raises(FileNotFoundError, match="physics"):
        normalize_o6_sources(source_root, tmp_path / "destination")


@pytest.mark.parametrize("forbidden", ["http://", "https://", "omniverse://", "/home/"])
def test_normalizer_rejects_nonportable_authored_paths(tmp_path, forbidden):
    source_root = tmp_path / "source"
    source_paths = _make_source_tree(source_root)
    source_paths["right"]["robot"].write_text(
        f'#usda 1.0\ndef Xform "bad" ( references = @{forbidden}asset.usd@ ) {{}}\n'
    )

    with pytest.raises(ValueError, match="non-portable"):
        normalize_o6_sources(source_root, tmp_path / "destination")


def test_normalizer_rejects_source_symlinks(tmp_path):
    source_root = tmp_path / "source"
    source_paths = _make_source_tree(source_root)
    mesh = source_paths["left"]["mesh"]
    mesh.unlink()
    mesh.symlink_to(source_paths["right"]["mesh"])

    with pytest.raises(ValueError, match="symlink"):
        normalize_o6_sources(source_root, tmp_path / "destination")
