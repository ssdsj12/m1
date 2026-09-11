"""Offline-only helpers for pinned DexManipNet archive handling.

This module deliberately contains no Hugging Face, HDF5, training, or runtime
dependencies.  The command-line fetcher imports its network client lazily.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


DATASET_REPOSITORY = "LiKailin/DexManipNet"


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a regular file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unsafe_posix_path(path: PurePosixPath) -> bool:
    return path.is_absolute() or ".." in path.parts


def validate_tar_members(members: Iterable[tarfile.TarInfo]) -> tuple[tarfile.TarInfo, ...]:
    """Reject archive entries that could escape or create unsafe filesystem nodes."""

    checked: list[tarfile.TarInfo] = []
    for member in members:
        name = PurePosixPath(member.name)
        if not member.name or _unsafe_posix_path(name) or member.isdev() or member.isfifo():
            raise ValueError(f"unsafe tar member: {member.name}")
        if not (member.isdir() or member.isreg() or member.issym() or member.islnk()):
            raise ValueError(f"unsafe tar member: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if not member.linkname or _unsafe_posix_path(target):
                raise ValueError(f"unsafe tar member link: {member.name}")
        checked.append(member)
    return tuple(checked)


def _validate_required_root(staging: Path, required_root: str | None) -> None:
    if required_root is None:
        return
    path = PurePosixPath(required_root)
    if not required_root or _unsafe_posix_path(path):
        raise ValueError("required root must be a safe relative path")
    candidate = staging.joinpath(*path.parts)
    if not candidate.is_dir():
        raise ValueError(f"required root is missing after extraction: {required_root}")


def atomic_extract_tar(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    required_root: str | None = None,
) -> None:
    """Safely extract an archive into a staged sibling and atomically install it.

    A failure leaves a newly requested destination absent.  Existing destinations
    are never overwritten, so a failed retry cannot damage a previously verified
    extraction.
    """

    archive_path = Path(archive)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.extract-", dir=destination_path.parent)
    )
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            members = validate_tar_members(tar.getmembers())
            # ``data`` adds tarfile's platform-aware safety checks after the
            # explicit validation above, including link handling during extract.
            tar.extractall(staging, members=members, filter="data")
        _validate_required_root(staging, required_root)
        os.replace(staging, destination_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_download_manifest(
    download_root: str | os.PathLike[str],
    archive_paths: Iterable[str | os.PathLike[str]],
    revision: str,
    source_commit: str,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable record for pinned downloads."""

    root = Path(download_root).resolve()
    archives: list[dict[str, Any]] = []
    for supplied_path in archive_paths:
        path = Path(supplied_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"archive does not exist: {path}")
        try:
            relative_path = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"archive must live below download root: {path}") from error
        archives.append(
            {
                "name": path.name,
                "path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": (
                    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
                    f"{revision}/{path.name}"
                ),
            }
        )
    if not archives:
        raise ValueError("at least one archive is required")
    return {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": revision,
        "maniptrans_commit": source_commit,
        "archives": archives,
    }


__all__ = [
    "DATASET_REPOSITORY",
    "atomic_extract_tar",
    "build_download_manifest",
    "sha256_file",
    "validate_tar_members",
]
