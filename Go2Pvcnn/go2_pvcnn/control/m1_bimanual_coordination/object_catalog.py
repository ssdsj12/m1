"""Offline RialTo object catalog and instance validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shlex
from types import MappingProxyType
from typing import Mapping, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_OBJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PREPARATION_SCRIPT = "Go2Pvcnn/scripts/m1_rialto_object_assets.py"


def _require_non_empty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _require_sha256(
    name: str, value: object, *, allow_none: bool = False
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _relative_asset_path(asset_path: Path, asset_root: Path) -> str:
    try:
        return asset_path.resolve().relative_to(asset_root.resolve()).as_posix()
    except ValueError:
        return str(asset_path)


def _preparation_command(asset_root: Path) -> str:
    return (
        "PYTHONPATH=Go2Pvcnn python "
        f"{_PREPARATION_SCRIPT} prepare --destination "
        f"{shlex.quote(str(asset_root))} --allow-network"
    )


class AssetPreparationRequiredError(FileNotFoundError):
    """Raised when the explicit asset-preparation step is still required."""

    def __init__(
        self,
        object_class: str,
        asset_path: Path,
        asset_root: Path,
        *,
        reason: str = "resolved USD asset is missing",
    ) -> None:
        relative_path = _relative_asset_path(asset_path, asset_root)
        command = _preparation_command(asset_root)
        super().__init__(
            f"object catalog preparation required for {object_class!r}: {reason} "
            f"{relative_path!r} at {asset_path}; run `{command}`"
        )
        self.object_class = object_class
        self.asset_path = asset_path
        self.asset_root = asset_root


PreparationRequiredError = AssetPreparationRequiredError


@dataclass(frozen=True)
class ObjectClassRecord:
    object_class: str
    usd_path: Path
    source_sha256: str
    resolved_sha256: str | None
    mass_kg: float
    scale: float
    collision_profile: str
    grasp_profile: str

    def __post_init__(self) -> None:
        _require_non_empty_text("object_class", self.object_class)
        if not isinstance(self.usd_path, Path):
            raise TypeError("usd_path must be a pathlib.Path")
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256("resolved_sha256", self.resolved_sha256, allow_none=True)
        object.__setattr__(
            self, "mass_kg", _require_positive_real("mass_kg", self.mass_kg)
        )
        object.__setattr__(
            self, "scale", _require_positive_real("scale", self.scale)
        )
        _require_non_empty_text("collision_profile", self.collision_profile)
        _require_non_empty_text("grasp_profile", self.grasp_profile)

    @property
    def sha256(self) -> str | None:
        """Backward-compatible alias for the resolved USD digest."""

        return self.resolved_sha256


@dataclass(frozen=True)
class ObjectInstance:
    object_id: str
    object_class: str
    pose: tuple[float, ...]
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.object_id, str)
            or _OBJECT_ID_RE.fullmatch(self.object_id) is None
        ):
            raise ValueError(
                "object_id must contain only letters, digits, '.', '_' or '-' "
                "and must start with a letter or digit"
            )
        _require_non_empty_text("object_class", self.object_class)
        if not isinstance(self.pose, tuple) or not self.pose:
            raise TypeError("pose must be a non-empty tuple of finite numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.pose
        ):
            raise ValueError("pose must contain only finite numbers")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")


@dataclass(frozen=True)
class ObjectCatalog:
    """Immutable class records plus optional geometry inspection notes."""

    _records: Mapping[str, ObjectClassRecord]
    _geometry_notes: Mapping[str, str]
    uses_legacy_box: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_records", MappingProxyType(dict(self._records))
        )
        object.__setattr__(
            self, "_geometry_notes", MappingProxyType(dict(self._geometry_notes))
        )
        if not isinstance(self.uses_legacy_box, bool):
            raise TypeError("uses_legacy_box must be bool")

    @property
    def classes(self) -> tuple[str, ...]:
        """Return class names in catalog-file order."""

        return tuple(self._records)

    @property
    def geometry_notes(self) -> Mapping[str, str]:
        return self._geometry_notes

    def resolve(self, object_class: str) -> ObjectClassRecord:
        """Resolve one class or raise ``KeyError`` for an unknown class."""

        try:
            return self._records[object_class]
        except KeyError as error:
            raise KeyError(f"unknown object class: {object_class!r}") from error

    def validate_instances(
        self, instances: Sequence[ObjectInstance]
    ) -> tuple[ObjectInstance, ...]:
        """Validate classes/IDs and return instances in stable ID order."""

        if isinstance(instances, (str, bytes)):
            raise TypeError("instances must be a sequence of ObjectInstance values")
        checked: list[ObjectInstance] = []
        seen: set[str] = set()
        for instance in instances:
            if not isinstance(instance, ObjectInstance):
                raise TypeError("instances must contain ObjectInstance values")
            if instance.object_id in seen:
                raise ValueError(f"duplicate object_id: {instance.object_id!r}")
            seen.add(instance.object_id)
            self.resolve(instance.object_class)
            checked.append(instance)
        return tuple(sorted(checked, key=lambda item: item.object_id))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_relative_path(value: object, object_class: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{object_class} usd_path must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{object_class} usd_path must use POSIX separators")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{object_class} usd_path must be relative")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{object_class} usd_path contains traversal components")
    return parts


def _resolve_asset_path(
    asset_root: Path, relative_path: object, object_class: str
) -> Path:
    parts = _catalog_relative_path(relative_path, object_class)
    root = asset_root.resolve()
    path = root.joinpath(*parts)
    if not root.is_dir():
        raise AssetPreparationRequiredError(
            object_class,
            path,
            root,
            reason="asset root is missing",
        )
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{object_class} usd_path escapes asset root: {relative_path!r}"
        ) from error
    if path.is_symlink():
        raise ValueError(f"{object_class} usd_path must not be a symlink")
    if resolved.suffix.lower() not in _SUPPORTED_USD_SUFFIXES:
        raise ValueError(
            f"{object_class} usd_path must be a USD file, got {resolved.name!r}"
        )
    if not resolved.is_file():
        raise AssetPreparationRequiredError(
            object_class,
            resolved,
            root,
        )
    return resolved


def _load_class_record(
    object_class: str, payload: object, asset_root: Path
) -> tuple[ObjectClassRecord, str | None]:
    if not isinstance(payload, dict):
        raise ValueError(f"catalog entry must be an object: {object_class}")
    required = {
        "usd_path",
        "source_sha256",
        "resolved_sha256",
        "mass_kg",
        "scale",
        "collision_profile",
        "grasp_profile",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            f"catalog entry missing keys for {object_class}: {', '.join(missing)}"
        )
    path = _resolve_asset_path(asset_root, payload["usd_path"], object_class)
    source_sha256 = _require_sha256(
        f"{object_class} source_sha256", payload["source_sha256"]
    )
    resolved_sha256 = _require_sha256(
        f"{object_class} resolved_sha256",
        payload["resolved_sha256"],
        allow_none=True,
    )
    if resolved_sha256 is None:
        raise AssetPreparationRequiredError(
            object_class,
            path,
            Path(asset_root),
            reason="resolved USD SHA-256 is not recorded",
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != resolved_sha256:
        raise ValueError(
            f"resolved SHA-256 mismatch for {object_class}: expected {resolved_sha256}, "
            f"got {actual_sha256}"
        )
    note = payload.get("geometry_note")
    if note is not None:
        note = _require_non_empty_text(
            f"{object_class} geometry_note", note
        )
    record = ObjectClassRecord(
        object_class=object_class,
        usd_path=path,
        source_sha256=source_sha256,
        resolved_sha256=resolved_sha256,
        mass_kg=payload["mass_kg"],
        scale=payload["scale"],
        collision_profile=payload["collision_profile"],
        grasp_profile=payload["grasp_profile"],
    )
    return record, note


def load_catalog(path: Path | None, asset_root: Path) -> ObjectCatalog:
    """Load and validate a local catalog, or select legacy ``/Box`` mode.

    ``None`` is an explicit request to preserve the existing single-box scene.
    A concrete path is always validated strictly; missing or malformed files
    never silently fall back to legacy behavior.
    """

    if path is None:
        return ObjectCatalog({}, {}, uses_legacy_box=True)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"object catalog does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read object catalog: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("object catalog root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("object catalog schema_version must be 1")
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("object catalog classes must be a non-empty object")

    records: dict[str, ObjectClassRecord] = {}
    geometry_notes: dict[str, str] = {}
    for object_class, class_payload in classes.items():
        _require_non_empty_text("object_class", object_class)
        if object_class in records:
            raise ValueError(f"duplicate object class: {object_class}")
        record, note = _load_class_record(
            object_class, class_payload, Path(asset_root)
        )
        records[object_class] = record
        if note is not None:
            geometry_notes[object_class] = note
    return ObjectCatalog(records, geometry_notes)


__all__ = [
    "AssetPreparationRequiredError",
    "PreparationRequiredError",
    "ObjectCatalog",
    "ObjectClassRecord",
    "ObjectInstance",
    "load_catalog",
]
