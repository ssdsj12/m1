#!/usr/bin/env python3
"""Fetch the pinned DexManipNet archives and ManipTrans source checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    DEXMANIPNET_REVISION,
    MANIPTRANS_COMMIT,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.download import (
    DATASET_REPOSITORY,
    atomic_extract_tar,
    build_download_manifest,
)


ARCHIVE_NAMES = ("dexmanipnet_favor.tar.gz", "dexmanipnet_oakinkv2.tar.gz")
ARCHIVE_SIZES_BYTES = {
    "dexmanipnet_favor.tar.gz": 5_242_217_074,
    "dexmanipnet_oakinkv2.tar.gz": 3_068_372_422,
}
MANIPTRANS_REPOSITORY = "https://github.com/ManipTrans/ManipTrans.git"


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "external" / "dexmanipnet"


def _archive_paths(download_root: Path) -> tuple[Path, Path]:
    return tuple(download_root / name for name in ARCHIVE_NAMES)  # type: ignore[return-value]


def _require_archives(download_root: Path) -> tuple[Path, Path]:
    paths = _archive_paths(download_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned archive(s): {', '.join(missing)}")
    return paths


def _ensure_download_space(download_root: Path) -> None:
    missing_bytes = sum(
        ARCHIVE_SIZES_BYTES[path.name] for path in _archive_paths(download_root) if not path.is_file()
    )
    if missing_bytes and shutil.disk_usage(download_root).free < missing_bytes:
        raise RuntimeError(
            f"insufficient free space for pinned archives: need {missing_bytes} bytes"
        )


def _fetch_archives(download_root: Path) -> tuple[Path, Path]:
    """Use the hub cache/resume machinery, importing it only for a real fetch."""

    from huggingface_hub import snapshot_download

    download_root.mkdir(parents=True, exist_ok=True)
    _ensure_download_space(download_root)
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DEXMANIPNET_REVISION,
        allow_patterns=("README.md", "dexmanipnet_favor.tar.gz", "dexmanipnet_oakinkv2.tar.gz"),
        local_dir=download_root,
    )
    return _require_archives(download_root)


def _git(directory: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "--no-optional-locks", "-C", str(directory), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_maniptrans(destination: Path) -> None:
    """Reject mutable source trees without refreshing or locking their index."""

    status = _git(
        destination,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError(f"ManipTrans checkout is not clean: {status}")


def _clone_maniptrans(destination: Path) -> tuple[str, str]:
    """Install a detached, pinned checkout without replacing an existing one."""

    if destination.exists():
        _require_clean_maniptrans(destination)
        commit = _git(destination, "rev-parse", "HEAD")
        if commit != MANIPTRANS_COMMIT:
            raise ValueError(
                f"existing ManipTrans checkout is not pinned to {MANIPTRANS_COMMIT}: {commit}"
            )
        return commit, _git(destination, "rev-parse", "HEAD^{tree}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.clone-", dir=destination.parent))
    shutil.rmtree(staging)
    try:
        subprocess.run(
            ("git", "clone", "--filter=blob:none", MANIPTRANS_REPOSITORY, str(staging)), check=True
        )
        _git(staging, "checkout", "--detach", MANIPTRANS_COMMIT)
        _require_clean_maniptrans(staging)
        commit = _git(staging, "rev-parse", "HEAD")
        if commit != MANIPTRANS_COMMIT:
            raise RuntimeError(f"ManipTrans checkout did not resolve pinned commit: {commit}")
        tree = _git(staging, "rev-parse", "HEAD^{tree}")
        os.replace(staging, destination)
        return commit, tree
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_maniptrans(destination: Path) -> tuple[str, str]:
    if not destination.is_dir():
        raise FileNotFoundError(f"missing pinned ManipTrans checkout: {destination}")
    _require_clean_maniptrans(destination)
    commit = _git(destination, "rev-parse", "HEAD")
    if commit != MANIPTRANS_COMMIT:
        raise ValueError(f"ManipTrans checkout is not pinned to {MANIPTRANS_COMMIT}: {commit}")
    return commit, _git(destination, "rev-parse", "HEAD^{tree}")


def _validate_existing_extraction(destination: Path) -> None:
    if not (destination / "sequences").is_dir():
        raise ValueError(f"existing extraction has no sequences/ root: {destination}")


def _write_manifest(root: Path, document: dict[str, object]) -> Path:
    manifest_path = root / "manifests" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent, delete=False
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, manifest_path)
    return manifest_path


def _verify_manifest(root: Path, document: dict[str, object]) -> Path:
    """Compare stored evidence with current local facts without changing either."""

    manifest_path = root / "manifests" / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing download manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored != document:
        raise ValueError("download manifest does not exactly match pinned local archives/source")
    for archive_name in ARCHIVE_NAMES:
        _validate_existing_extraction(root / "extracted" / archive_name.removesuffix(".tar.gz"))
    return manifest_path


def run(root: Path, *, download_only: bool, verify_only: bool) -> Path:
    if download_only and verify_only:
        raise ValueError("--download-only and --verify-only cannot be combined")

    download_root = root / "downloads"
    source_root = root / "source_maniptrans"
    if verify_only:
        archive_paths = _require_archives(download_root)
        _, tree = _verify_maniptrans(source_root)
    else:
        archive_paths = _fetch_archives(download_root)
        _, tree = _clone_maniptrans(source_root)

    manifest = build_download_manifest(
        download_root,
        archive_paths,
        revision=DEXMANIPNET_REVISION,
        source_commit=MANIPTRANS_COMMIT,
    )
    manifest["maniptrans_repository"] = MANIPTRANS_REPOSITORY
    manifest["maniptrans_tree"] = tree

    if verify_only:
        return _verify_manifest(root, manifest)

    if not download_only:
        extracted_root = root / "extracted"
        for archive in archive_paths:
            destination = extracted_root / archive.name.removesuffix(".tar.gz")
            if destination.exists():
                _validate_existing_extraction(destination)
            else:
                atomic_extract_tar(archive, destination, required_root="sequences")

    return _write_manifest(root, manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root(), help="external data root")
    parser.add_argument(
        "--download-only", action="store_true", help="download archives and source without extracting"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="verify existing archives and pinned source without network"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        manifest_path = run(
            args.root, download_only=args.download_only, verify_only=args.verify_only
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"fetch failed: {error}") from error
    if args.verify_only:
        print(f"verified pinned downloads and source: {manifest_path}")
    else:
        print(f"wrote pinned download manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
