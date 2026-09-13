import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.download import (
    atomic_extract_tar,
    build_download_manifest,
    sha256_file,
    validate_tar_members,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_fetch_dexmanipnet.py"


def _write_tar(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return archive


def _load_fetch_script():
    spec = importlib.util.spec_from_file_location("fetch_dexmanipnet_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(directory), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize("name", ["/abs/file", "../escape", "safe/../../escape"])
def test_tar_validation_rejects_escape(name):
    member = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="unsafe tar member"):
        validate_tar_members([member])


@pytest.mark.parametrize("kind", [tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_tar_validation_rejects_devices_and_special_files(kind):
    member = tarfile.TarInfo("unsafe")
    member.type = kind
    with pytest.raises(ValueError, match="unsafe tar member"):
        validate_tar_members([member])


def test_tar_validation_rejects_unsafe_symbolic_and_hard_links():
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        member = tarfile.TarInfo("safe/link")
        member.type = kind
        member.linkname = "../escape"
        with pytest.raises(ValueError, match="unsafe tar member link"):
            validate_tar_members([member])


def test_download_cli_contains_both_archives_and_fixed_revision():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dexmanipnet_favor.tar.gz" in source
    assert "dexmanipnet_oakinkv2.tar.gz" in source
    assert "revision=DEXMANIPNET_REVISION" in source


def test_failed_extraction_never_installs_destination(tmp_path):
    archive = _write_tar(tmp_path, {"../escape": b"bad"})
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="unsafe tar member"):
        atomic_extract_tar(archive, destination)
    assert not destination.exists()


def test_atomic_extract_validates_before_installing_and_installs_only_complete_tree(tmp_path):
    incomplete = _write_tar(tmp_path, {"other/item": b"bad"})
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="required root"):
        atomic_extract_tar(incomplete, destination, required_root="sequences")
    assert not destination.exists()

    archive = _write_tar(tmp_path, {"sequences/item.bin": b"complete"})
    atomic_extract_tar(archive, destination, required_root="sequences")
    assert (destination / "sequences" / "item.bin").read_bytes() == b"complete"


def test_atomic_extract_promotes_matching_archive_wrapper_for_required_root(tmp_path):
    archive = _write_tar(tmp_path, {"installed/sequences/item.bin": b"complete"})
    destination = tmp_path / "installed"

    atomic_extract_tar(archive, destination, required_root="sequences")

    assert (destination / "sequences" / "item.bin").read_bytes() == b"complete"
    assert not (destination / "installed").exists()
    assert not list(tmp_path.glob(".installed.extract-*"))


def test_sha256_and_manifest_record_pinned_archives(tmp_path):
    root = tmp_path / "dexmanipnet"
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    favor = downloads / "dexmanipnet_favor.tar.gz"
    oakink = downloads / "dexmanipnet_oakinkv2.tar.gz"
    favor.write_bytes(b"favor")
    oakink.write_bytes(b"oakink")

    assert sha256_file(favor) == hashlib.sha256(b"favor").hexdigest()
    manifest = build_download_manifest(
        root,
        (favor, oakink),
        revision="a" * 40,
        source_commit="b" * 40,
    )

    assert manifest["dataset_revision"] == "a" * 40
    assert manifest["maniptrans_commit"] == "b" * 40
    assert [entry["name"] for entry in manifest["archives"]] == [favor.name, oakink.name]
    assert manifest["archives"][0]["size_bytes"] == len(b"favor")
    assert manifest["archives"][1]["sha256"] == hashlib.sha256(b"oakink").hexdigest()


def test_cli_help_exits_without_fetching():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--download-only" in completed.stdout
    assert "--verify-only" in completed.stdout


def test_verify_only_compares_existing_manifest_without_writing_or_fetching(tmp_path, monkeypatch):
    fetch = _load_fetch_script()
    root = tmp_path / "dexmanipnet"
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    archive_paths = tuple(downloads / name for name in fetch.ARCHIVE_NAMES)
    for path, payload in zip(archive_paths, (b"favor", b"oakink"), strict=True):
        path.write_bytes(payload)
        (root / "extracted" / path.name.removesuffix(".tar.gz") / "sequences").mkdir(
            parents=True
        )

    source = root / "source_maniptrans"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test User")
    (source / "README").write_text("pinned source\n", encoding="utf-8")
    _git(source, "add", "README")
    _git(source, "commit", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(fetch, "DEXMANIPNET_REVISION", "a" * 40)
    monkeypatch.setattr(fetch, "MANIPTRANS_COMMIT", commit)

    manifest = build_download_manifest(downloads, archive_paths, "a" * 40, commit)
    manifest["maniptrans_repository"] = fetch.MANIPTRANS_REPOSITORY
    manifest["maniptrans_tree"] = tree
    manifest_path = root / "manifests" / "download_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before_content = manifest_path.read_bytes()
    before_mtime_ns = manifest_path.stat().st_mtime_ns
    before_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    assert fetch.run(root, download_only=False, verify_only=True) == manifest_path

    assert manifest_path.read_bytes() == before_content
    assert manifest_path.stat().st_mtime_ns == before_mtime_ns
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before_paths


@pytest.mark.parametrize(
    ("mutation_path", "contents"),
    [("README", "modified source\n"), ("untracked.txt", "untracked source\n")],
)
def test_existing_and_verify_only_reject_dirty_maniptrans_without_writing(
    tmp_path, monkeypatch, mutation_path, contents
):
    fetch = _load_fetch_script()
    root = tmp_path / "dexmanipnet"
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    archive_paths = tuple(downloads / name for name in fetch.ARCHIVE_NAMES)
    for path, payload in zip(archive_paths, (b"favor", b"oakink"), strict=True):
        path.write_bytes(payload)
        (root / "extracted" / path.name.removesuffix(".tar.gz") / "sequences").mkdir(
            parents=True
        )

    source = root / "source_maniptrans"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test User")
    (source / "README").write_text("pinned source\n", encoding="utf-8")
    _git(source, "add", "README")
    _git(source, "commit", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(fetch, "DEXMANIPNET_REVISION", "a" * 40)
    monkeypatch.setattr(fetch, "MANIPTRANS_COMMIT", commit)

    manifest = build_download_manifest(downloads, archive_paths, "a" * 40, commit)
    manifest["maniptrans_repository"] = fetch.MANIPTRANS_REPOSITORY
    manifest["maniptrans_tree"] = tree
    manifest_path = root / "manifests" / "download_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (source / mutation_path).write_text(contents, encoding="utf-8")
    before_manifest = manifest_path.read_bytes()
    before_manifest_mtime = manifest_path.stat().st_mtime_ns
    index_path = source / ".git" / "index"
    before_index_mtime = index_path.stat().st_mtime_ns
    before_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    with pytest.raises(ValueError, match="not clean"):
        fetch._clone_maniptrans(source)
    with pytest.raises(ValueError, match="not clean"):
        fetch.run(root, download_only=False, verify_only=True)

    assert manifest_path.read_bytes() == before_manifest
    assert manifest_path.stat().st_mtime_ns == before_manifest_mtime
    assert index_path.stat().st_mtime_ns == before_index_mtime
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before_paths


def test_verify_only_reports_verification_not_manifest_write(monkeypatch, capsys):
    fetch = _load_fetch_script()
    manifest_path = Path("/tmp/download_manifest.json")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--verify-only"])
    monkeypatch.setattr(fetch, "run", lambda *args, **kwargs: manifest_path)

    assert fetch.main() == 0
    output = capsys.readouterr().out
    assert "verified pinned downloads and source" in output
    assert "wrote pinned" not in output
