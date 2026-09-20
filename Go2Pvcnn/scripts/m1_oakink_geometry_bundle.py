"""Deterministic, fail-closed OakInk geometry bundle builder.

This command is deliberately independent of the existing overlay.  It never
rewrites source archives or ``run_e``; a failed member is left in diagnostics
and no production manifest is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.preprocess import load_object_collision_mesh
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.geometry_overlay import verify_geometry_overlay

UPSTREAM = {
    "repository": "kelvin34501/OakInk-v2",
    "revision": "21705616140d726607027e70d58b7837f442ffd8",
    "archive_sha256": "40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_compatible_references(sequence_root: Path, raw_archive: Path | None = None) -> tuple[str, ...]:
    refs: set[str] = set()
    for info_path in sorted(Path(sequence_root).rglob("seq_info.json")):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # ``bih`` sequences contain two main-hand sides and are part of the
        # dual-source corpus; passive single-hand recordings are not.
        if info.get("interaction_mode") not in {"rh_main", "lh_main", "bh_main"}:
            continue
        for key in ("obj_rh_path", "obj_lh_path"):
            value = info.get(key)
            if isinstance(value, str) and value and value.startswith("ObjURDF/"):
                refs.add(value)
    if raw_archive is not None:
        with tarfile.open(raw_archive, "r") as tar:
            members = set(tar.getnames())
        refs = {ref for ref in refs if _raw_member(ref) in members}
    return tuple(sorted(refs))


def _raw_member(reference: str) -> str:
    if not reference.startswith("ObjURDF/") or not reference.endswith(".urdf"):
        raise ValueError(f"unsafe geometry reference: {reference!r}")
    rel = reference.removeprefix("ObjURDF/")[:-5]
    return "object_raw/" + rel + ".ply"


def _write_urdf(path: Path, mesh_name: str) -> None:
    root = ET.Element("robot", name="oakink_object")
    link = ET.SubElement(root, "link", name="base")
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_name, scale="1 1 1")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_name, scale="1 1 1")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def _extract_member(archive: Path, member: str, destination: Path) -> None:
    with tarfile.open(archive, "r") as tar:
        info = tar.getmember(member)
        if not info.isfile() or PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
            raise ValueError("unsafe raw archive member")
        source = tar.extractfile(info)
        if source is None:
            raise ValueError("raw archive member is unreadable")
        destination.write_bytes(source.read())


def build_geometry_bundle(*, raw_archive: Path, sequence_root: Path, output_root: Path,
                          recipe: dict[str, object], expected_upstream: dict[str, str],
                          jobs: int = 1, generator_python: Path | None = None,
                          generator_script: Path | None = None) -> Path:
    if jobs != 1:
        raise ValueError("production geometry generation requires jobs=1")
    if expected_upstream != UPSTREAM:
        raise ValueError("upstream identity mismatch")
    raw_archive, sequence_root, output_root = map(Path, (raw_archive, sequence_root, output_root))
    if _sha(raw_archive) != UPSTREAM["archive_sha256"]:
        raise ValueError("raw archive SHA-256 mismatch")
    references = collect_compatible_references(sequence_root, raw_archive)
    if not references:
        raise ValueError("no source-compatible geometry references")
    if len(references) != 90:
        raise ValueError(f"expected exactly 90 compatible references, found {len(references)}")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_root.name + ".staging-", dir=parent))
    entries = []
    failures = []
    try:
        for reference in references:
            member = _raw_member(reference)
            safe = hashlib.sha256(reference.encode()).hexdigest()[:16]
            mesh = stage / "urdf" / "meshes" / f"{safe}.ply"
            urdf = stage / "urdf" / f"{safe}.urdf"
            mesh.parent.mkdir(parents=True, exist_ok=True)
            urdf.parent.mkdir(parents=True, exist_ok=True)
            try:
                _extract_member(raw_archive, member, mesh)
                raw_digest = _sha(mesh)
                # CoACD is run one object at a time with the pinned recipe.
                # The raw member is retained only as provenance; the URDF
                # points at the generated collision mesh.
                generated = mesh.with_suffix(".obj")
                if generator_python is not None and generator_script is not None:
                    subprocess.run([
                        str(generator_python), str(generator_script), "--quiet",
                        "--input", str(mesh), "--output", str(generated),
                        "--threshold", "0.07", "--max-convex-hull", "32",
                        "--preprocess-mode", "auto", "--prep-resolution", "50",
                        "--resolution", "2000", "--mcts-node", "20",
                        "--mcts-iteration", "2000", "--mcts-max-depth", "5",
                        "--seed", "1", "--apx-mode", "ch",
                    ], check=True, capture_output=True, text=True)
                    mesh.unlink()
                    mesh = generated
                _write_urdf(urdf, "meshes/" + mesh.name)
                collision = load_object_collision_mesh(urdf)
                if not collision.is_watertight or collision.volume <= 0:
                    raise ValueError("collision mesh is not positive watertight")
                entries.append({"reference": reference, "raw_member": member,
                    "raw_sha256": raw_digest, "urdf": str(urdf.relative_to(stage)),
                    "urdf_sha256": _sha(urdf),
                    "meshes": [{"path": str(mesh.relative_to(stage)), "sha256": _sha(mesh)}],
                    "checks": {"finite": True, "watertight": True,
                                "positive_volume": True, "zero_origin": True, "unit_scale": True}})
            except Exception as exc:  # diagnostics are part of the atomic contract
                failures.append({"reference": reference, "raw_member": member, "error": str(exc)})
                for path in (mesh, urdf):
                    path.unlink(missing_ok=True)
        (stage / "diagnostics.json").write_text(json.dumps({"failures": failures}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if failures or len(entries) != len(references):
            raise RuntimeError(f"geometry generation failed: {len(failures)} of {len(references)} references")
        manifest = {"schema_version": 1, "upstream": UPSTREAM, "recipe": recipe, "entries": entries}
        manifest_path = stage / "geometry_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        verify_geometry_bundle(stage, _sha(manifest_path), expected_count=len(references))
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(stage, output_root)
        return output_root / "geometry_manifest.json"
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def verify_geometry_bundle(output_root: Path, expected_manifest_sha256: str,
                          expected_count: int = 90) -> dict[str, object]:
    root = Path(output_root)
    manifest_path = root / "geometry_manifest.json"
    if _sha(manifest_path) != expected_manifest_sha256:
        raise ValueError("geometry manifest SHA-256 mismatch")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("upstream") != UPSTREAM or len(data.get("entries", [])) != expected_count:
        raise ValueError("geometry manifest count or upstream mismatch")
    references = set()
    for item in data["entries"]:
        ref = item["reference"]
        if ref in references:
            raise ValueError("duplicate geometry reference")
        references.add(ref)
        urdf = root / item["urdf"]
        if _sha(urdf) != item["urdf_sha256"]:
            raise ValueError("URDF SHA-256 drift")
        load_object_collision_mesh(urdf)
        for mesh in item["meshes"]:
            if _sha(root / mesh["path"]) != mesh["sha256"]:
                raise ValueError("mesh SHA-256 drift")
    # Re-run the immutable production resolver's XML/path/hash checks too.
    verify_geometry_overlay(root, manifest_path, expected_manifest_sha256)
    return {"manifest_sha256": expected_manifest_sha256, "count": len(references), "verified": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=90)
    parser.add_argument("--generator-python", type=Path)
    parser.add_argument("--generator-script", type=Path)
    args = parser.parse_args()
    recipe = {"tool": "oakink-geometry-bundle", "version": "1", "arguments": {"jobs": 1}}
    manifest = build_geometry_bundle(raw_archive=args.raw_archive, sequence_root=args.sequence_root,
                                     output_root=args.output_root, recipe=recipe,
                                     expected_upstream=UPSTREAM, jobs=1,
                                     generator_python=args.generator_python,
                                     generator_script=args.generator_script)
    verify_geometry_bundle(manifest.parent, _sha(manifest), args.expected_count)
    print(json.dumps({"manifest": str(manifest), "sha256": _sha(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
