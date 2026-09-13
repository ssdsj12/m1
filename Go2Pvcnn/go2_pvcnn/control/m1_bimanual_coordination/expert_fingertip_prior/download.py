"""Offline-only helpers for pinned DexManipNet archive handling.

This module deliberately contains no Hugging Face, HDF5, training, or runtime
dependencies.  The command-line fetcher imports its network client lazily.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


DATASET_REPOSITORY = "LiKailin/DexManipNet"


def git_readonly(directory: Path, *args: str) -> str:
    """Run a Git query without optional locks or intentional index refresh."""

    completed = subprocess.run(
        ("git", "--no-optional-locks", "-C", str(directory), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_checkout(directory: str | os.PathLike[str]) -> None:
    """Reject tracked, untracked, ignored, or submodule drift in a source checkout."""

    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"missing pinned source checkout: {path}")
    status = git_readonly(
        path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError(f"pinned source checkout is not clean: {status}")


def verify_clean_checkout(
    directory: str | os.PathLike[str],
    *,
    expected_commit: str,
    expected_tree: str | None = None,
) -> dict[str, str]:
    """Return actual commit/tree only after a clean, exact pinned checkout check."""

    path = Path(directory)
    require_clean_checkout(path)
    commit = git_readonly(path, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"source checkout is not pinned to {expected_commit}: {commit}")
    tree = git_readonly(path, "rev-parse", "HEAD^{tree}")
    if expected_tree is not None and tree != expected_tree:
        raise ValueError(f"source checkout tree does not match manifest: {tree}")
    return {"commit": commit, "tree": tree}


def _canonical_json(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def verify_pinned_external_inputs(
    root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    archive_names: tuple[str, ...],
    expected_revision: str,
    expected_commit: str,
    source_repository: str,
) -> dict[str, Any]:
    """Recompute exact archive bytes and source Git facts against the stored manifest."""

    import json

    external_root = Path(root)
    manifest = Path(manifest_path)
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"missing pinned download manifest: {manifest}")
    try:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("download manifest is invalid") from error
    if type(stored) is not dict:
        raise ValueError("download manifest must be a JSON object")
    tree = stored.get("maniptrans_tree")
    if (
        type(tree) is not str
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in tree)
    ):
        raise ValueError("download manifest has no valid source tree")

    download_root = external_root / "downloads"
    archive_paths = tuple(download_root / name for name in archive_names)
    missing = [str(path) for path in archive_paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"missing pinned archive(s): {', '.join(missing)}")
    source = verify_clean_checkout(
        external_root / "source_maniptrans",
        expected_commit=expected_commit,
        expected_tree=tree,
    )
    actual = build_download_manifest(
        download_root,
        archive_paths,
        revision=expected_revision,
        source_commit=expected_commit,
    )
    actual["maniptrans_repository"] = source_repository
    actual["maniptrans_tree"] = source["tree"]
    if stored != actual:
        raise ValueError("download manifest does not exactly match pinned archive/source bytes")

    archives = [
        {"name": item["name"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in actual["archives"]
    ]
    facts = {"archives": archives, "source": source}
    return {
        "archive_manifest_sha256": sha256_file(manifest),
        "source_manifest_sha256": hashlib.sha256(_canonical_json(source)).hexdigest(),
        "facts": facts,
    }


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


def _extraction_payload_root(
    staging: Path, destination: Path, required_root: str | None
) -> Path:
    """Return the validated archive payload, allowing one exact-name wrapper.

    Published DexManipNet tarballs wrap their contents in a directory named for
    the archive.  Promote only that exact, sole wrapper so the installed tree
    keeps the frozen ``destination/sequences`` contract without accepting an
    arbitrary archive layout.
    """

    try:
        _validate_required_root(staging, required_root)
    except ValueError:
        wrapper = staging / destination.name
        children = tuple(sorted(staging.iterdir(), key=lambda path: path.name))
        if children != (wrapper,):
            raise
        _validate_required_root(wrapper, required_root)
        return wrapper
    return staging


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
        payload = _extraction_payload_root(staging, destination_path, required_root)
        os.replace(payload, destination_path)
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
    "git_readonly",
    "require_clean_checkout",
    "sha256_file",
    "validate_tar_members",
    "verify_clean_checkout",
    "verify_pinned_external_inputs",
]
