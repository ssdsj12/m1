from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json
import importlib.util
from hashlib import sha256

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_train_fingertip_expert.py"
PLAN = Path(__file__).parents[2] / "docs" / "superpowers" / "plans" / "2026-09-11-t500-dexmanipnet-fingertip-prior.md"


def _trainer_module():
    spec = importlib.util.spec_from_file_location("task6_trainer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_script_does_not_import_task_or_object_features():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("description", "primitive", "object_id", "palm_target", "box_pose"):
        assert forbidden not in source


def test_training_script_has_hash_verification_and_nonproduction_smoke_gate():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_aggregate_manifest" in source
    assert "synthetic_smoke" in source
    assert "production_deployable" in source
    assert "0.01" in source


def test_formal_task11_command_pins_gpu0_cublas_device_and_batch_size():
    source = PLAN.read_text(encoding="utf-8")
    expected = (
        "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$PWD "
        "/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_fingertip_expert.py "
        "--dataset-manifest data/external/dexmanipnet/converted/run_a/aggregate_manifest.json "
        "--output-dir data/external/dexmanipnet/artifacts/expert --member-seeds 42,43,44,45,46 "
        "--epochs 200 --device cuda:0 --batch-size 128"
    )

    assert expected in source


def test_cuda_determinism_preflight_sets_supported_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    _trainer_module()._configure_cuda_determinism("cuda:0")

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("value", [":4096:8", ":16:8"])
def test_cuda_determinism_preflight_preserves_supported_setting(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)

    _trainer_module()._configure_cuda_determinism("cuda:0")

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == value


def test_cuda_determinism_preflight_rejects_invalid_setting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        _trainer_module()._configure_cuda_determinism("cuda:0")


def test_cpu_determinism_preflight_does_not_change_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    _trainer_module()._configure_cuda_determinism("cpu")

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_invalid_cuda_setting_rejects_before_output_state_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "cuda-rejected"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        _trainer_module().main([*_smoke_args(output), "--device", "cuda:0"])

    assert not output.exists()
    assert not (tmp_path / ".cuda-rejected.resume-v1").exists()


def _smoke_args(output: Path) -> list[str]:
    return [
        "--synthetic-smoke",
        "--output-dir",
        str(output),
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--hidden",
        "8",
        "--member-seeds",
        "101,202",
    ]


@pytest.mark.parametrize("failure_point", [(0, 1), (0, 2), (1, 1)])
def test_interrupted_training_resumes_to_byte_identical_outputs(
    tmp_path: Path, failure_point: tuple[int, int]
):
    module = _trainer_module()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    module.main(_smoke_args(uninterrupted))

    def fail_after_first_epoch(member_index: int, completed_epoch: int) -> None:
        if (member_index, completed_epoch) == failure_point:
            raise RuntimeError("injected epoch failure")

    with pytest.raises(RuntimeError, match="injected epoch failure"):
        module.main(_smoke_args(resumed), epoch_commit_hook=fail_after_first_epoch)

    resume_workspace = tmp_path / ".resumed.resume-v1"
    assert not resumed.exists()
    assert (resume_workspace / "progress.json").is_file()

    module.main(_smoke_args(resumed))

    assert not resume_workspace.exists()
    expected_manifest = (uninterrupted / "ensemble_manifest.json").read_bytes()
    actual_manifest = (resumed / "ensemble_manifest.json").read_bytes()
    assert actual_manifest == expected_manifest
    manifest = json.loads(actual_manifest)
    for record in manifest["members"]:
        relative = record["checkpoint"]
        expected = (uninterrupted / relative).read_bytes()
        actual = (resumed / relative).read_bytes()
        assert actual == expected
        assert sha256(actual).hexdigest() == record["checkpoint_sha256"]


def test_resume_identity_mismatch_rejects_without_publishing_output(tmp_path: Path):
    module = _trainer_module()
    output = tmp_path / "identity-mismatch"

    def interrupt(member_index: int, completed_epoch: int) -> None:
        raise RuntimeError(f"stop {member_index}/{completed_epoch}")

    with pytest.raises(RuntimeError, match="stop"):
        module.main(_smoke_args(output), epoch_commit_hook=interrupt)
    progress = tmp_path / ".identity-mismatch.resume-v1" / "progress.json"
    before = progress.read_bytes()

    changed = [*_smoke_args(output), "--learning-rate", "0.002"]
    with pytest.raises(ValueError, match="resume identity"):
        module.main(changed)

    assert not output.exists()
    assert progress.read_bytes() == before


def test_explicit_resume_checkpoint_rejects_training_configuration_mismatch(tmp_path: Path):
    module = _trainer_module()
    initial = tmp_path / "explicit-initial"
    destination = tmp_path / "explicit-rejected"
    module.main(_smoke_args(initial))

    changed = [
        *_smoke_args(destination),
        "--batch-size",
        "4",
        "--resume-checkpoint",
        str(initial),
    ]
    with pytest.raises(ValueError, match="training identity"):
        module.main(changed)

    assert not destination.exists()
    assert not (tmp_path / ".explicit-rejected.resume-v1").exists()


def test_resume_checkpoint_corruption_rejects_without_publishing_output(tmp_path: Path):
    module = _trainer_module()
    output = tmp_path / "checkpoint-corruption"

    def interrupt(member_index: int, completed_epoch: int) -> None:
        raise RuntimeError(f"stop {member_index}/{completed_epoch}")

    with pytest.raises(RuntimeError, match="stop"):
        module.main(_smoke_args(output), epoch_commit_hook=interrupt)
    workspace = tmp_path / ".checkpoint-corruption.resume-v1"
    progress = json.loads((workspace / "progress.json").read_text(encoding="utf-8"))
    checkpoint = workspace / progress["members"][0]["last"]["path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        module.main(_smoke_args(output))

    assert not output.exists()
    assert workspace.exists()


def test_unidentified_resume_directory_is_rejected_and_preserved(tmp_path: Path):
    module = _trainer_module()
    output = tmp_path / "occupied"
    workspace = tmp_path / ".occupied.resume-v1"
    workspace.mkdir()
    marker = workspace / "user-file.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="resume workspace"):
        module.main(_smoke_args(output))

    assert not output.exists()
    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_atomic_progress_write_is_not_blocked_by_stale_interrupted_temporary(tmp_path: Path):
    module = _trainer_module()
    destination = tmp_path / "progress.json"
    stale = tmp_path / ".progress.json.tmp"
    stale.write_bytes(b"partial")

    module._atomic_bytes(destination, b"complete")

    assert destination.read_bytes() == b"complete"
    assert stale.read_bytes() == b"partial"


def test_resume_checkpoint_record_cannot_escape_workspace(tmp_path: Path):
    module = _trainer_module()
    workspace = tmp_path / "workspace"
    (workspace / "checkpoints").mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="path"):
        module._record_file(
            workspace,
            {
                "path": "checkpoints/../../outside.pt",
                "sha256": sha256(outside.read_bytes()).hexdigest(),
            },
            label="last",
        )


def test_resume_directory_retains_selected_member_when_epochs_are_complete(tmp_path: Path):
    initial = tmp_path / "initial"
    resumed = tmp_path / "resumed"
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])}
    command = [sys.executable, str(SCRIPT), "--synthetic-smoke", "--epochs", "1"]

    subprocess.run([*command, "--output-dir", str(initial)], check=True, env=environment)
    aggregate = json.loads((initial / "synthetic-shards" / "aggregate_manifest.json").read_text())
    assert aggregate["verified_inputs"] == {"nonproduction_synthetic": True}
    subprocess.run(
        [*command, "--output-dir", str(resumed), "--resume-checkpoint", str(initial)],
        check=True,
        env=environment,
    )

    assert (resumed / "checkpoints" / "member-00-best.pt").is_file()
    assert "nonproduction_synthetic" in SCRIPT.read_text(encoding="utf-8")


def test_group_overlap_is_rejected_before_shards_are_loaded():
    document = {
        "split_groups": {
            "train": ["sequence-a"],
            "validation": ["sequence-a"],
            "test": ["sequence-b"],
        }
    }

    with pytest.raises(ValueError, match="overlap"):
        _trainer_module()._validate_group_assignments(document)


@pytest.mark.parametrize("field, value", [("seed", 9999), ("checkpoint_sha256", "0" * 64)])
def test_resume_rejects_manifest_member_tampering_before_checkpoint_load(
    tmp_path: Path, field: str, value: object
):
    initial = tmp_path / "initial"
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])}
    subprocess.run(
        [sys.executable, str(SCRIPT), "--synthetic-smoke", "--epochs", "1", "--output-dir", str(initial)],
        check=True,
        env=environment,
    )
    manifest_path = initial / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][1][field] = value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256"):
        _trainer_module()._load_resume(
            initial,
            member_index=0,
            seed=1701,
            hidden=(512, 512, 512),
            aggregate_sha=manifest["dataset_aggregate_sha256"],
        )
