from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    DEXMANIPNET_REVISION,
    ExpertWindow,
    MANIPTRANS_COMMIT,
    PRIOR_HORIZON,
    PriorPhase,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import SOURCE_HANDS
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import (
    deterministic_group_split,
    verify_aggregate_manifest,
    write_shards,
)


def _window(group: str, index: int) -> ExpertWindow:
    return ExpertWindow(
        fingertip_position_palm=torch.full((5, 3), index / 100.0, dtype=torch.float32),
        fingertip_velocity_palm=torch.full((5, 3), index / 10.0, dtype=torch.float32),
        contact_mask=torch.tensor([(index >> bit) & 1 for bit in range(5)], dtype=torch.bool),
        phase=PriorPhase(index % len(PriorPhase)),
        future_fingertip_velocity_palm=torch.full(
            (PRIOR_HORIZON, 5, 3), index, dtype=torch.float32
        ),
        source_group=group,
        source_sha256=f"{index + 1:064x}",
    )


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_group_split_is_unique_deterministic_and_never_leaks_sequence_windows():
    groups = [f"seq-{index:02d}" for index in range(20)]
    duplicated = [group for group in groups for _ in range(3)]

    split = deterministic_group_split(duplicated, seed=42)

    assert split == deterministic_group_split(list(reversed(duplicated)), seed=42)
    assert (len(split.train), len(split.validation), len(split.test)) == (16, 2, 2)
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))
    assert set(split.train + split.validation + split.test) == set(groups)


def test_two_complete_writes_have_identical_shard_and_aggregate_hashes(tmp_path: Path):
    groups = [f"favor/sequence_{index:02d}/rh" for index in range(10)]
    windows = tuple(_window(group, index) for index, group in enumerate(groups))
    split = deterministic_group_split(groups, seed=7)
    audits = [
        {
            "accepted": True,
            "frames": 21,
            "hand": "inspire_rh",
            "input_sha256": f"{index + 1:064x}",
            "reason": "accepted",
            "sequence": f"sequence_{index:02d}",
            "side": "rh",
            "source": "favor",
        }
        for index in range(10)
    ]

    manifest_a = write_shards(
        tmp_path / "a", windows, split, shard_size=2, audits=audits,
        archive_manifest_sha256="a" * 64, source_manifest_sha256="b" * 64,
    )
    manifest_b = write_shards(
        tmp_path / "b", tuple(reversed(windows)), split, shard_size=2,
        audits=list(reversed(audits)), archive_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )

    hashes_a = _tree_hash(tmp_path / "a")
    hashes_b = _tree_hash(tmp_path / "b")
    assert hashes_a == hashes_b
    assert manifest_a.aggregate_sha256 == manifest_b.aggregate_sha256
    assert manifest_a.shards == tuple(sorted(manifest_a.shards, key=lambda item: item.path))


def test_shards_have_strict_arrays_stats_audit_jsonl_and_no_group_leakage(tmp_path: Path):
    output = tmp_path / "output"
    groups = [f"oakinkv2/sequence_{index:02d}/lh" for index in range(10)]
    windows = tuple(_window(group, index) for index, group in enumerate(groups))
    split = deterministic_group_split(groups, seed=3)

    manifest = write_shards(
        output,
        windows,
        split,
        shard_size=3,
        audits=[{"sequence": group, "accepted": True} for group in groups],
        archive_manifest_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
    )

    seen: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for record in manifest.shards:
        shard = output / record.path
        assert sha256(shard.read_bytes()).hexdigest() == record.sha256
        with np.load(shard, allow_pickle=False) as arrays:
            assert arrays.files == sorted(arrays.files)
            count = arrays["phase"].shape[0]
            assert arrays["fingertip_position_palm"].shape == (count, 5, 3)
            assert arrays["fingertip_position_palm"].dtype == np.float32
            assert arrays["fingertip_velocity_palm"].shape == (count, 5, 3)
            assert arrays["contact_mask"].shape == (count, 5)
            assert arrays["contact_mask"].dtype == np.bool_
            assert arrays["future_fingertip_velocity_palm"].shape == (
                count,
                PRIOR_HORIZON,
                5,
                3,
            )
            assert arrays["source_group"].dtype.kind == "U"
            assert np.isfinite(arrays["fingertip_position_palm"]).all()
            seen[record.split].update(arrays["source_group"].tolist())
        assert sum(record.phase_counts.values()) == record.samples
        assert sum(record.contact_pattern_counts.values()) == record.samples
    assert not (seen["train"] & seen["validation"])
    assert not (seen["train"] & seen["test"])
    assert not (seen["validation"] & seen["test"])

    rows = [json.loads(line) for line in (output / "audit.jsonl").read_text().splitlines()]
    assert rows == sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    document = json.loads((output / "aggregate_manifest.json").read_text())
    assert document["aggregate_sha256"] == manifest.aggregate_sha256
    assert document["shards"] == [asdict(record) for record in manifest.shards]
    audit_bytes = (output / "audit.jsonl").read_bytes()
    assert document["audit_jsonl_bytes"] == len(audit_bytes)
    assert document["audit_jsonl_sha256"] == sha256(audit_bytes).hexdigest()
    assert verify_aggregate_manifest(output)["aggregate_sha256"] == manifest.aggregate_sha256

    for name in ("audit", "shard", "aggregate"):
        copied = tmp_path / f"mutated-{name}"
        shutil.copytree(output, copied)
        if name == "audit":
            (copied / "audit.jsonl").write_bytes(audit_bytes + b"{}\n")
        elif name == "shard":
            (copied / manifest.shards[0].path).write_bytes(b"tampered shard")
        else:
            aggregate_path = copied / "aggregate_manifest.json"
            aggregate_path.write_bytes(
                aggregate_path.read_bytes().replace(b'"format_version":1', b'"format_version":2')
            )
        with pytest.raises(ValueError, match=name if name != "aggregate" else "aggregate"):
            verify_aggregate_manifest(copied)


def test_storage_rejects_missing_group_assignment_nonfinite_or_existing_output(tmp_path: Path):
    groups = [f"sequence-{index}" for index in range(10)]
    split = deterministic_group_split(groups, seed=1)
    outside = _window("outside", 1)
    bad = _window(groups[0], 2)
    bad.fingertip_position_palm[0, 0] = torch.nan

    for windows in ((outside,), (bad,)):
        try:
            write_shards(tmp_path / "target", windows, split)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid shard input was accepted")
        assert not (tmp_path / "target").exists()

    valid = tuple(_window(group, index) for index, group in enumerate(groups))
    write_shards(tmp_path / "complete", valid, split)
    try:
        write_shards(tmp_path / "complete", valid, split)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output was overwritten")


def test_conversion_cli_help_does_not_inspect_or_download_data(tmp_path: Path):
    script = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_convert_dexmanipnet.py"
    missing_root = tmp_path / "must-stay-missing"

    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(missing_root), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )

    assert completed.returncode == 0
    assert "offline" in completed.stdout.lower()
    assert not missing_root.exists()


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(directory), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_convert_script():
    script = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_convert_dexmanipnet.py"
    spec = importlib.util.spec_from_file_location("convert_dexmanipnet_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_full_conversion_fixture(root: Path) -> tuple[str, str]:
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    archive_payloads = {
        "dexmanipnet_favor.tar.gz": b"favor-archive-fixture",
        "dexmanipnet_oakinkv2.tar.gz": b"oakink-archive-fixture",
    }
    archive_records = []
    for name, payload in archive_payloads.items():
        path = downloads / name
        path.write_bytes(payload)
        archive_records.append(
            {"name": name, "path": name, "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload),
             "source_url": f"https://huggingface.co/datasets/LiKailin/DexManipNet/resolve/{'a' * 40}/{name}"}
        )

    source_root = root / "source_maniptrans"
    source_root.mkdir()
    _git(source_root, "init")
    _git(source_root, "config", "user.email", "test@example.invalid")
    _git(source_root, "config", "user.name", "Test User")

    spec = SOURCE_HANDS["inspire_rh"]
    hand_path = source_root / spec.urdf_relpath
    hand_path.parent.mkdir(parents=True)
    links = ['<link name="R_hand_base_link"/>']
    joints = []
    parent = "R_hand_base_link"
    for index, name in enumerate(spec.joint_order):
        child = f"moving_{index}"
        links.append(f'<link name="{child}"/>')
        kind = "prismatic" if index == 0 else "revolute"
        joints.append(
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="1 0 0"/></joint>'
        )
        parent = child
    for index, tip in enumerate(spec.fingertip_links):
        links.append(f'<link name="{tip}"/>')
        joints.append(
            f'<joint name="tip_{index}" type="fixed"><parent link="{parent}"/>'
            f'<child link="{tip}"/><origin xyz="0 {0.01 * index} 0"/></joint>'
        )
    hand_path.write_text(
        '<robot name="fixture">' + "".join(links + joints) + "</robot>", encoding="utf-8"
    )
    _git(source_root, "add", ".")
    _git(source_root, "commit", "-m", "fixture")
    commit = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")

    manifest = {
        "dataset_repository": "LiKailin/DexManipNet",
        "dataset_revision": "a" * 40,
        "maniptrans_commit": commit,
        "maniptrans_repository": "https://github.com/ManipTrans/ManipTrans.git",
        "maniptrans_tree": tree,
        "archives": archive_records,
    }
    manifest_path = root / "manifests" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    favor = root / "extracted" / "dexmanipnet_favor"
    oakink = root / "extracted" / "dexmanipnet_oakinkv2"
    sequence = favor / "sequences" / "sequence_000"
    sequence.mkdir(parents=True)
    (oakink / "sequences").mkdir(parents=True)
    geometry = favor / "ObjURDF" / "object.urdf"
    geometry.parent.mkdir()
    geometry.write_text(
        '<robot name="box"><link name="object"><collision><geometry>'
        '<box size="0.2 0.2 0.2"/></geometry></collision></link></robot>', encoding="utf-8"
    )
    frames = 22
    (sequence / "seq_info.json").write_text(
        json.dumps(
            {
                "seq_len": frames,
                "dexhand": "inspire",
                "interaction_mode": "rh_main",
                "obj_rh_path": "ObjURDF/object.urdf",
            }
        ),
        encoding="utf-8",
    )
    q = np.zeros((frames, 12), dtype=np.float64)
    q[:, 0] = np.linspace(0.11, 0.09, frames)
    state = np.zeros((frames, 13), dtype=np.float64)
    state[:, 6] = 1.0
    with h5py.File(sequence / "rollouts.hdf5", "w") as h5:
        rollout = h5.create_group("rollouts/successful/rollout_0")
        rollout.create_dataset("reward", data=np.ones(frames))
        rollout.create_dataset("q_rh", data=q)
        rollout.create_dataset("dq_rh", data=np.zeros_like(q))
        rollout.create_dataset("state_rh", data=state)
        rollout.create_dataset("state_manip_obj_rh", data=state)
    return commit, tree


def test_two_complete_offline_conversions_have_identical_all_file_hashes(tmp_path: Path, monkeypatch):
    root = tmp_path / "external"
    commit, _ = _write_full_conversion_fixture(root)
    convert = _load_convert_script()
    monkeypatch.setattr(convert, "DEXMANIPNET_REVISION", "a" * 40)
    monkeypatch.setattr(convert, "MANIPTRANS_COMMIT", commit)
    outputs = (tmp_path / "conversion-a", tmp_path / "conversion-b")

    for output in outputs:
        convert.run(root, output, seed=17, shard_size=7)

    assert _tree_hash(outputs[0]) == _tree_hash(outputs[1])
    manifest_a = json.loads((outputs[0] / "aggregate_manifest.json").read_text())
    manifest_b = json.loads((outputs[1] / "aggregate_manifest.json").read_text())
    assert manifest_a["aggregate_sha256"] == manifest_b["aggregate_sha256"]
    assert manifest_a["shards"] == manifest_b["shards"]
    assert manifest_a["shards"][0]["hand_counts"] == {"inspire_rh": 7}
    assert manifest_a["verified_inputs"]["archives"] == manifest_b["verified_inputs"]["archives"]
    assert manifest_a["verified_inputs"]["source"] == {"commit": commit, "tree": manifest_a["verified_inputs"]["source"]["tree"]}


def test_conversion_rejects_malformed_archive_provenance_before_touching_output(tmp_path: Path, monkeypatch):
    root = tmp_path / "external"
    commit, _ = _write_full_conversion_fixture(root)
    manifest_path = root / "manifests" / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archives"][0]["sha256"] = "not-a-sha"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "must-stay-missing"
    convert = _load_convert_script()
    monkeypatch.setattr(convert, "DEXMANIPNET_REVISION", "a" * 40)
    monkeypatch.setattr(convert, "MANIPTRANS_COMMIT", commit)
    try:
        convert.run(root, output, seed=42, shard_size=8)
    except ValueError as error:
        assert "manifest" in str(error).lower() or "archive" in str(error).lower()
    else:
        raise AssertionError("malformed archive provenance was accepted")
    assert not output.exists()


@pytest.mark.parametrize(
    "failure", ["mutated_archive", "missing_archive", "missing_source", "dirty_source"]
)
def test_conversion_verifies_actual_archives_and_clean_source_before_output(
    tmp_path: Path, monkeypatch, failure: str
):
    root = tmp_path / "external"
    commit, _ = _write_full_conversion_fixture(root)
    convert = _load_convert_script()
    monkeypatch.setattr(convert, "DEXMANIPNET_REVISION", "a" * 40)
    monkeypatch.setattr(convert, "MANIPTRANS_COMMIT", commit)
    if failure == "mutated_archive":
        (root / "downloads" / "dexmanipnet_favor.tar.gz").write_bytes(b"tampered archive")
    elif failure == "missing_archive":
        (root / "downloads" / "dexmanipnet_oakinkv2.tar.gz").unlink()
    elif failure == "missing_source":
        (root / "source_maniptrans").rename(root / "source_maniptrans.removed")
    else:
        source_file = root / "source_maniptrans" / SOURCE_HANDS["inspire_rh"].urdf_relpath
        source_file.write_text(source_file.read_text() + "\n<!-- dirty -->\n", encoding="utf-8")
    output = tmp_path / "must-stay-missing"

    with pytest.raises((FileNotFoundError, ValueError)):
        convert.run(root, output, seed=42, shard_size=8)

    assert not output.exists()
