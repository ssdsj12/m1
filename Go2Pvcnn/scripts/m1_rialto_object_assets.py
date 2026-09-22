"""Fetch, convert, and inspect the pinned RialTo object inputs.

Network access is intentionally opt-in and limited to the ``prepare`` command.
Importing this module, inspecting USD, and converting already-local files never
perform network I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "assets/m1_objects/rialto/source_manifest.json"
SUPPORTED_CONVERSION_EXTENSIONS = {".usdz", ".glb"}
USD_SOURCE_EXTENSIONS = {".usd", ".usda", ".usdc"}
USDC_MAGIC = b"PXR-USDC"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and minimally validate the pinned source manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"repo", "revision", "assets"}
    if not required.issubset(manifest):
        raise ValueError(f"RialTo manifest missing keys: {sorted(required - set(manifest))}")
    if not isinstance(manifest["assets"], dict) or not manifest["assets"]:
        raise ValueError("RialTo manifest assets must be a non-empty object")
    for name, record in manifest["assets"].items():
        if not isinstance(record, dict) or not {"source_url", "source_sha256", "resolved_path"}.issubset(record):
            raise ValueError(f"RialTo manifest entry is incomplete: {name}")
        if len(record["source_sha256"]) != 64 or not re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]):
            raise ValueError(f"RialTo manifest entry has invalid SHA-256: {name}")
    return manifest


def _safe_resolved_path(destination: Path, relative: str) -> Path:
    root = destination.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"manifest path escapes destination: {relative}")
    return path


def fetch_sources(destination: Path, *, allow_network: bool = False) -> dict[str, Path]:
    """Materialize pinned sources under *destination*.

    Existing files with matching digests are reused offline. Missing or stale
    files require ``allow_network=True``; this prevents runtime code from
    silently reaching the network.
    """

    destination = Path(destination)
    manifest = load_manifest()
    resolved: dict[str, Path] = {}
    pending: list[tuple[str, dict[str, Any], Path]] = []
    for name, record in sorted(manifest["assets"].items()):
        path = _safe_resolved_path(destination, record["resolved_path"])
        if path.is_file() and sha256_file(path) == record["source_sha256"]:
            resolved[name] = path
        else:
            pending.append((name, record, path))
    if pending and not allow_network:
        names = ", ".join(name for name, _record, _path in pending)
        raise RuntimeError(
            "network access requires explicit opt-in; rerun the asset-preparation "
            f"command with --allow-network (missing: {names})"
        )
    for name, record, path in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(record["source_url"]) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            digest = sha256_file(temporary)
            if digest != record["source_sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}: expected {record['source_sha256']}, got {digest}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        resolved[name] = path
    return resolved


def _convert_with_available_tool(source: Path, destination: Path) -> None:
    """Run an installed converter, raising a useful error when unavailable."""

    if source.suffix.lower() == ".usdz":
        converter = shutil.which("usdcat")
        if converter:
            try:
                subprocess.run(
                    [converter, str(source), "-o", str(destination)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or "").strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"USDZ converter failed for {source}{suffix}") from exc
            return
        raise RuntimeError("cannot convert USDZ: usdcat is not installed (install Pixar USD)")
    if source.suffix.lower() == ".glb":
        _convert_glb_with_trimesh(source, destination)
        return
    raise ValueError(f"unsupported conversion extension: {source.suffix or '<none>'}")


def _format_usda_float(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"GLB mesh contains non-finite coordinate: {number!r}")
    text = format(number, ".17g")
    if text == "-0":
        return "0.0"
    if "." not in text and "e" not in text.lower():
        return f"{text}.0"
    return text


def _format_usda_vector(values: object) -> str:
    return "(" + ", ".join(_format_usda_float(value) for value in values) + ")"


def _write_combined_mesh_usda(mesh: object, destination: Path) -> None:
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    bounds = getattr(mesh, "bounds", None)
    if vertices is None or faces is None or bounds is None:
        raise RuntimeError("cannot convert GLB: trimesh returned an incomplete mesh")
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("cannot convert GLB: mesh has no vertices or faces")
    if getattr(vertices, "ndim", None) != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"cannot convert GLB: expected Nx3 vertices, got {vertices.shape!r}")
    if getattr(faces, "ndim", None) != 2 or faces.shape[1] < 3:
        raise RuntimeError(f"cannot convert GLB: expected polygon faces, got {faces.shape!r}")
    if getattr(bounds, "shape", None) != (2, 3):
        raise RuntimeError(f"cannot convert GLB: expected 3D bounds, got {bounds.shape!r}")
    if not all(math.isfinite(float(value)) for value in vertices.reshape(-1)):
        raise RuntimeError("cannot convert GLB: mesh contains non-finite vertex coordinates")
    if not all(math.isfinite(float(value)) for value in bounds.reshape(-1)):
        raise RuntimeError("cannot convert GLB: mesh bounds are non-finite")

    point_values = ",\n            ".join(_format_usda_vector(row) for row in vertices)
    face_counts = ", ".join(str(int(len(face))) for face in faces)
    face_indices = ", ".join(str(int(index)) for face in faces for index in face)
    extent = f"[{_format_usda_vector(bounds[0])}, {_format_usda_vector(bounds[1])}]"
    text = (
        "#usda 1.0\n"
        "(\n"
        "    metersPerUnit = 1\n"
        "    upAxis = \"Z\"\n"
        ")\n"
        "\n"
        "def Xform \"Root\" {\n"
        "    def Mesh \"CombinedMesh\" {\n"
        f"        float3[] extent = {extent}\n"
        "        uniform token subdivisionScheme = \"none\"\n"
        f"        point3f[] points = [{point_values}]\n"
        f"        int[] faceVertexCounts = [{face_counts}]\n"
        f"        int[] faceVertexIndices = [{face_indices}]\n"
        "    }\n"
        "}\n"
    )
    destination.write_text(text, encoding="utf-8", newline="\n")


def _convert_glb_with_trimesh(source: Path, destination: Path) -> None:
    """Load a GLB with trimesh and publish one deterministic ASCII USDA mesh."""

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "cannot convert GLB: trimesh is not installed; install trimesh in the asset-preparation environment"
        ) from exc

    with source.open("rb") as stream:
        if stream.read(4) != b"glTF":
            raise RuntimeError(f"cannot convert GLB {source}: invalid glTF binary header")
    try:
        loaded = trimesh.load(source, file_type="glb", force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            parts = loaded.dump()
        elif isinstance(loaded, trimesh.Trimesh):
            parts = [loaded]
        else:
            parts = []
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(f"cannot convert GLB {source}: failed to load with trimesh: {detail}") from exc

    meshes = [part for part in parts if isinstance(part, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError(f"cannot convert GLB {source}: failed to load a mesh geometry")
    meshes.sort(
        key=lambda part: (
            str(part.metadata.get("name", "")),
            str(part.metadata.get("node", "")),
        )
    )
    try:
        combined = trimesh.util.concatenate(meshes)
        _write_combined_mesh_usda(combined, destination)
    except RuntimeError:
        raise
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(f"cannot convert GLB {source}: failed to combine mesh geometry: {detail}") from exc


def _dependency_paths_from_text(path: Path) -> list[str]:
    """Return missing local asset references from a textual USD layer."""

    text = path.read_text(encoding="utf-8", errors="replace")
    missing: list[str] = []
    for dependency in re.findall(r"@([^@]+)@", text):
        if dependency.startswith(("anon:", "http:", "https:")):
            continue
        dependency_path = Path(dependency)
        if not dependency_path.is_absolute():
            dependency_path = path.parent / dependency_path
        if not dependency_path.exists():
            missing.append(dependency)
    return missing


def _is_binary_usdc(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(USDC_MAGIC)) == USDC_MAGIC


def _validate_usd_dependencies(path: Path) -> dict[str, object]:
    inspection = inspect_usd(path)
    if not inspection["dependencies_resolved"]:
        missing = inspection.get("missing_dependencies") or []
        details = ", ".join(str(item) for item in missing) or "unknown dependency"
        raise RuntimeError(f"USD output has unresolved dependencies: {details}")
    return inspection


def convert_usdz_or_glb(source: Path, destination: Path) -> Path:
    """Convert one local USDZ/GLB to deterministic ``<stem>.usd`` output.

    Conversion and validation happen in a temporary sibling file. The
    previously published output is replaced atomically only after both steps
    succeed; failures leave an existing valid output untouched.
    """

    source = Path(source)
    extension = source.suffix.lower()
    if extension not in SUPPORTED_CONVERSION_EXTENSIONS:
        raise ValueError(f"unsupported conversion extension: {source.suffix or '<none>'}")
    if not source.is_file():
        raise FileNotFoundError(f"conversion source does not exist: {source}")
    destination = Path(destination)
    output = destination if destination.suffix.lower() in {".usd", ".usda", ".usdc"} else destination / f"{source.stem}.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _convert_with_available_tool(source, temporary)
        if not temporary.is_file():
            raise RuntimeError(f"converter did not create deterministic USD output: {output}")
        _validate_usd_dependencies(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _inspect_with_pxr(path: Path) -> dict[str, object] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except ImportError:
        return None
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"unable to open USD stage: {path}")
    prims = list(stage.Traverse())
    bounds: list[list[float]] | None = None
    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        minimum, maximum = box.GetMin(), box.GetMax()
        if all(value == value for value in (*minimum, *maximum)):
            bounds = [[float(value) for value in minimum], [float(value) for value in maximum]]
    except Exception:
        bounds = None
    unresolved = [str(asset) for asset in stage.GetUsedLayers() if not Path(str(asset.resolvedPath)).exists()]
    if not _is_binary_usdc(path):
        for dependency in _dependency_paths_from_text(path):
            if dependency not in unresolved:
                unresolved.append(dependency)
    return {
        "prim_count": len(prims),
        "bounds": bounds,
        "dependencies_resolved": not unresolved,
        "missing_dependencies": unresolved,
    }


def inspect_usd(path: Path) -> dict[str, object]:
    """Report primitive count, bounds, and dependency status for a USD file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"USD file does not exist: {path}")
    pxr_result = _inspect_with_pxr(path)
    if pxr_result is not None:
        return pxr_result
    if _is_binary_usdc(path):
        raise RuntimeError(
            f"cannot inspect binary PXR-USDC file without Pixar USD (pxr): {path}; "
            "refusing to decode binary data as text"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    prim_count = len(re.findall(r"^\s*(?:def|over)\s+\w+", text, flags=re.MULTILINE))
    extent_match = re.search(
        r"extent\s*=\s*\[\s*\(([^)]+)\)\s*,\s*\(([^)]+)\)\s*\]", text, flags=re.MULTILINE
    )
    bounds = None
    if extent_match:
        bounds = [[float(value.strip()) for value in extent_match.group(1).split(",")], [float(value.strip()) for value in extent_match.group(2).split(",")]]
    missing = _dependency_paths_from_text(path)
    return {
        "prim_count": prim_count,
        "bounds": bounds,
        "dependencies_resolved": not missing,
        "missing_dependencies": missing,
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_prepared_manifest(
    destination: Path,
    source_manifest: dict[str, Any],
    output_metadata: dict[str, dict[str, object]],
) -> Path:
    path = destination / "prepared_manifest.json"
    payload = {
        "schema_version": 1,
        "repo": source_manifest["repo"],
        "revision": source_manifest["revision"],
        "assets": output_metadata,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def prepare_assets(destination: Path, *, allow_network: bool = False) -> dict[str, object]:
    """Fetch, validate, convert, and record the prepared local object assets."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = load_manifest()
    paths = fetch_sources(destination, allow_network=allow_network)
    converted: dict[str, str] = {}
    output_metadata: dict[str, dict[str, object]] = {}
    for name, source in sorted(paths.items()):
        if source.suffix.lower() in SUPPORTED_CONVERSION_EXTENSIONS:
            usd_path = convert_usdz_or_glb(source, destination / "converted")
            converted[name] = str(usd_path)
            generated = usd_path.resolve() != source.resolve()
        elif source.suffix.lower() in USD_SOURCE_EXTENSIONS:
            usd_path = source
            _validate_usd_dependencies(usd_path)
            generated = False
        else:
            raise ValueError(f"unsupported prepared asset extension: {source.suffix or '<none>'}")
        inspection = _validate_usd_dependencies(usd_path)
        record: dict[str, object] = {
            "source_path": _relative_path(source, destination),
            "source_sha256": sha256_file(source),
            "usd_path": _relative_path(usd_path, destination),
            "usd_sha256": sha256_file(usd_path),
            "inspection": inspection,
        }
        if generated:
            record["generated_usd_sha256"] = record["usd_sha256"]
        output_metadata[name] = record
    prepared_manifest = _write_prepared_manifest(destination, source_manifest, output_metadata)
    return {
        "sources": paths,
        "converted": converted,
        "output_metadata": output_metadata,
        "prepared_manifest": str(prepared_manifest),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="fetch pinned sources and convert local inputs")
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--allow-network", action="store_true", help="explicitly permit source downloads")
    inspect = subparsers.add_parser("inspect", help="inspect one local USD")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_usd(args.path), sort_keys=True))
        return 0
    result = prepare_assets(args.destination, allow_network=args.allow_network)
    result["sources"] = {name: str(path) for name, path in result["sources"].items()}  # type: ignore[union-attr]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
