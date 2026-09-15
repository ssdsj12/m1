"""Read-only, independently pinned OakInk geometry overlay verification.

This verifies declarations and byte identity, not mesh physical validity or
production corpus qualification. The preprocessing geometry gate remains required.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Mapping
import xml.etree.ElementTree as ET


_UPSTREAM = {
    "repository": "kelvin34501/OakInk-v2",
    "revision": "21705616140d726607027e70d58b7837f442ffd8",
    "archive_sha256": "40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2",
}


def _relative(value: object) -> str:
    if (type(value) is not str or not value or "\\" in value
            or "\x00" in value or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("expected normalized POSIX relative path")
    return value


def _hash(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256")
    return value


def _keys(value: object, expected: set[str]) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("unsupported manifest fields")
    return value


def _nonempty_object(value: object) -> None:
    if type(value) is not dict or not value:
        raise ValueError("expected nonempty evidence object")


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number: " + value)


def _absolute_root(root: Path) -> Path:
    path = Path(root)
    if ".." in path.parts:
        raise ValueError("root traversal")
    return path.absolute()


def _read_bounded(root: Path, relative: str, expected: str | None = None) -> bytes:
    """Open each component without following links, including root ancestors."""
    _relative(relative)
    descriptor = None
    try:
        descriptor = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = (*root.parts[1:], *PurePosixPath(relative).parts)
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("geometry path is not a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("geometry SHA-256 drift")
        return data
    except (OSError, RuntimeError) as error:
        raise ValueError("unsafe or missing geometry path") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class _Entry:
    reference: str
    urdf: str
    urdf_sha256: str
    meshes: tuple[tuple[str, str], ...]


def _vector(value: str, expected: tuple[float, float, float]) -> None:
    try:
        numbers = tuple(float(part) for part in value.split())
    except ValueError as error:
        raise ValueError("invalid geometry transform") from error
    if numbers != expected:
        raise ValueError("geometry requires unit scale and zero origin")


class _GeometryTreeBuilder(ET.TreeBuilder):
    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise ValueError("unsupported XML declarations")


def _verify_xml(data: bytes, entry: _Entry) -> None:
    try:
        robot = ET.fromstring(data, parser=ET.XMLParser(target=_GeometryTreeBuilder()))
    except ET.ParseError as error:
        raise ValueError("invalid URDF XML") from error
    if robot.tag != "robot":
        raise ValueError("expected URDF robot")
    declared = {path for path, _ in entry.meshes}
    used = set()
    collision_count = 0
    for element in robot.iter():
        if element.tag not in {"visual", "collision"}:
            continue
        origins = element.findall("origin")
        if len(origins) > 1:
            raise ValueError("duplicate geometry origin")
        for origin in origins:
            _vector(origin.get("xyz", "0 0 0"), (0., 0., 0.))
            _vector(origin.get("rpy", "0 0 0"), (0., 0., 0.))
        geometry = element.findall("geometry")
        if len(geometry) != 1 or len(geometry[0]) != 1 or geometry[0][0].tag != "mesh":
            raise ValueError("expected one declared mesh geometry")
        mesh = geometry[0][0]
        filename = _relative(mesh.get("filename"))
        path = str(PurePosixPath(entry.urdf).parent / filename)
        if path not in declared:
            raise ValueError("undeclared URDF mesh")
        _vector(mesh.get("scale", "1 1 1"), (1., 1., 1.))
        used.add(path)
        collision_count += element.tag == "collision"
    if not collision_count or used != declared:
        raise ValueError("missing collision mesh or unused declaration")
    # A mesh outside visual/collision is not a validated declaration.
    if len(list(robot.iter("mesh"))) != sum(len(e.findall("geometry/mesh")) for e in robot.iter() if e.tag in {"visual", "collision"}):
        raise ValueError("mesh outside visual/collision")


def _validate_entry(root: Path, entry: _Entry) -> None:
    data = _read_bounded(root, entry.urdf, entry.urdf_sha256)
    _verify_xml(data, entry)
    for path, digest in entry.meshes:
        _read_bounded(root, path, digest)


@dataclass(frozen=True)
class VerifiedGeometryResolver:
    """Private immutable records; requested geometry is reverified on every call."""
    _root: Path
    _manifest_sha256: str
    _entries: Mapping[str, _Entry]

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    def resolve(self, reference: str, *, source: str) -> Path:
        if type(source) is not str or source != "oakinkv2":
            raise ValueError("unsupported geometry source")
        reference = _relative(reference)
        entry = self._entries.get(reference)
        if entry is None:
            raise ValueError("unknown geometry reference")
        _validate_entry(self._root, entry)
        return self._root / entry.urdf


def verify_geometry_overlay(root: Path, manifest_path: Path, expected_sha256: str) -> VerifiedGeometryResolver:
    """Verify exact externally pinned manifest bytes and all generated assets."""
    expected_sha256 = _hash(expected_sha256)
    root = _absolute_root(root)
    manifest_path = Path(manifest_path)
    if manifest_path.is_absolute():
        try:
            manifest_relative = manifest_path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("manifest outside overlay root") from error
    else:
        manifest_relative = manifest_path.as_posix()
    data = _read_bounded(root, manifest_relative, expected_sha256)
    try:
        manifest = json.loads(data, object_pairs_hook=_json_pairs,
                              parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid geometry manifest JSON") from error
    manifest = _keys(manifest, {"schema_version", "upstream", "recipe", "entries"})
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported geometry schema")
    if _keys(manifest["upstream"], set(_UPSTREAM)) != _UPSTREAM:
        raise ValueError("wrong upstream geometry pin")
    _nonempty_object(manifest["recipe"])
    if type(manifest["entries"]) is not list or not manifest["entries"]:
        raise ValueError("expected geometry entries")
    entries = {}
    files = {manifest_relative}
    raw_members = set()
    for raw_entry in manifest["entries"]:
        item = _keys(raw_entry, {"reference", "raw_member", "raw_sha256", "urdf", "urdf_sha256", "meshes", "checks"})
        reference = _relative(item["reference"])
        raw_member = _relative(item["raw_member"])
        _hash(item["raw_sha256"])
        _nonempty_object(item["checks"])
        urdf = _relative(item["urdf"])
        if reference in entries or raw_member in raw_members or urdf in files:
            raise ValueError("duplicate geometry reference or path")
        raw_members.add(raw_member)
        files.add(urdf)
        if type(item["meshes"]) is not list or not item["meshes"]:
            raise ValueError("expected declared meshes")
        meshes = []
        for raw_mesh in item["meshes"]:
            mesh = _keys(raw_mesh, {"path", "sha256"})
            path = _relative(mesh["path"])
            if path in files:
                raise ValueError("duplicate geometry file path")
            files.add(path)
            meshes.append((path, _hash(mesh["sha256"])))
        entry = _Entry(reference, urdf, _hash(item["urdf_sha256"]), tuple(meshes))
        _validate_entry(root, entry)
        entries[reference] = entry
    return VerifiedGeometryResolver(root, expected_sha256, MappingProxyType(entries))
