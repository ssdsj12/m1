#!/usr/bin/env python3
"""Close the vendor O6 USD dependencies into a project-owned asset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_LAYERS = ("base", "physics", "robot", "sensor")
_FORBIDDEN_AUTHORED_PATHS = (b"http://", b"https://", b"omniverse://", b"/home/")
_SOURCE_PACKAGES = {
    "left": "1、O6，urdf/linkerhand_O6_left.urdf",
    "right": "1、O6，urdf/linkerhand_O6_right.urdf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    symlinks = [path for path in paths if path.is_symlink()]
    if root.is_symlink() or symlinks:
        offender = root if root.is_symlink() else symlinks[0]
        raise ValueError(f"source symlink is not allowed: {offender}")
    non_files = [path for path in paths if not path.is_dir() and not path.is_file()]
    if non_files:
        raise ValueError(f"unsupported source entry: {non_files[0]}")
    return [path for path in paths if path.is_file()]


def _reject_nonportable_usd_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        payload = path.read_bytes().lower()
        for marker in _FORBIDDEN_AUTHORED_PATHS:
            if marker in payload:
                raise ValueError(
                    f"non-portable authored path {marker.decode()} found in {path}"
                )


def _source_layout(source_root: Path, side: str) -> dict[str, Any]:
    package_root = source_root / _SOURCE_PACKAGES[side]
    usd_root = package_root / f"linkerhand_O6_{side}.urdf"
    configuration = usd_root / "configuration"
    meshes = package_root / "meshes"
    entry = configuration / "O6_left.usd" if side == "left" else usd_root / "O6_right.usd"
    layers = {
        name: configuration / f"linkerhand_O6_{side}.urdf_{name}.usd"
        for name in _LAYERS
    }
    for path in (entry, *layers.values()):
        if path.is_symlink():
            raise ValueError(f"source symlink is not allowed: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration_files = _tree_files(configuration)
    mesh_files = _tree_files(meshes)
    checked_files = sorted(
        set((entry, *configuration_files, *mesh_files)), key=lambda path: path.as_posix()
    )
    _reject_nonportable_usd_paths(checked_files)
    return {
        "entry": entry,
        "layers": layers,
        "configuration": configuration,
        "meshes": meshes,
    }


def _copy_side(layout: dict[str, Any], staging_root: Path, side: str) -> dict[str, Any]:
    relative_side = Path(f"o6_{side}")
    side_destination = staging_root / relative_side
    shutil.copytree(layout["configuration"], side_destination / "configuration")
    shutil.copytree(layout["meshes"], side_destination / "meshes")
    shutil.copy2(layout["entry"], side_destination / f"O6_{side}.usd")
    return {
        "entry": (relative_side / f"O6_{side}.usd").as_posix(),
        "layers": {
            name: (
                relative_side
                / "configuration"
                / f"linkerhand_O6_{side}.urdf_{name}.usd"
            ).as_posix()
            for name in _LAYERS
        },
        "meshes": sorted(
            path.relative_to(staging_root).as_posix()
            for path in (side_destination / "meshes").rglob("*")
            if path.is_file()
        ),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_o6_sources(source_root: Path, destination_root: Path) -> dict[str, object]:
    """Validate and losslessly copy both O6 hands into ``destination_root``.

    The vendor tree is treated as read-only. Both sides are completely staged and
    hashed before either project-owned side directory is replaced.
    """

    source_root = Path(source_root).resolve(strict=True)
    destination_root = Path(destination_root).resolve()
    layouts = {side: _source_layout(source_root, side) for side in ("left", "right")}

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination_root.name}.normalize.", dir=destination_root.parent)
    )
    try:
        manifest: dict[str, Any] = {"schema": 1}
        for side in ("left", "right"):
            manifest[side] = _copy_side(layouts[side], staging_root, side)

        copied_files = sorted(
            path for path in staging_root.rglob("*") if path.is_file()
        )
        _reject_nonportable_usd_paths(copied_files)
        manifest["sha256"] = {
            path.relative_to(staging_root).as_posix(): _sha256(path)
            for path in copied_files
        }

        destination_root.mkdir(parents=True, exist_ok=True)
        for side in ("left", "right"):
            staged_side = staging_root / f"o6_{side}"
            destination_side = destination_root / f"o6_{side}"
            if destination_side.exists():
                if destination_side.is_symlink() or not destination_side.is_dir():
                    raise ValueError(f"invalid destination side directory: {destination_side}")
                shutil.rmtree(destination_side)
            os.replace(staged_side, destination_side)
        _atomic_write_json(destination_root / "source_manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = normalize_o6_sources(args.source_root, args.destination_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
