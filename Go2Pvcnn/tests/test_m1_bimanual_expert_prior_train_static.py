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
LEGACY_MANIFEST_FIXTURE = Path(__file__).parent / "fixtures" / "t500_legacy_expert_ensemble_manifest_v1.json"


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


def test_production_member_roster_rejects_before_dataset_or_output_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    module = _trainer_module()
    output = tmp_path / "wrong-production-roster"
    monkeypatch.setattr(
        module,
        "_resolve_manifest",
        lambda *_args, **_kwargs: pytest.fail("wrong roster reached dataset loading"),
    )

    with pytest.raises(ValueError, match="production member.*42.*46|roster"):
        module.main([
            "--dataset-manifest", "unused.json",
            "--output-dir", str(output),
            "--member-seeds", "1,2,3,4,5",
        ])

    assert not output.exists()
    assert not module._resume_workspace_path(output).exists()
    assert not list(tmp_path.glob(".wrong-production-roster.train-*"))


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


def _workspace_identity(module, *, epochs: int = 3):
    return module._training_identity(
        aggregate_sha="a" * 64,
        member_seeds=(101, 202),
        hidden=(8,),
        epochs=epochs,
        batch_size=8,
        learning_rate=1e-3,
        acceleration_weight=1e-5,
        jerk_weight=1e-7,
        device=module.torch.device("cpu"),
        synthetic_smoke=True,
    )


def _resume_state(module, workspace, *, epoch: int, best_nll: float):
    return {
        "format_version": 1,
        "member_index": 0,
        "seed": 101,
        "hidden": (8,),
        "epoch": epoch,
        "best_validation_nll": best_nll,
        "dataset_aggregate_sha256": "a" * 64,
        "training_identity_sha256": workspace.identity_sha256,
        "model_state": {},
        "optimizer_state": {},
    }


def test_checkpoint_slots_never_overwrite_progress_referenced_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _trainer_module()
    path = tmp_path / ".output.resume-v1"
    workspace = module._ResumeWorkspace(path, _workspace_identity(module))
    workspace.commit_epoch(
        member_index=0,
        seed=101,
        state=_resume_state(module, workspace, epoch=1, best_nll=3.0),
        best_changed=True,
    )
    workspace.commit_epoch(
        member_index=0,
        seed=101,
        state=_resume_state(module, workspace, epoch=2, best_nll=3.0),
        best_changed=False,
    )
    committed = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    old_last = path / committed["members"][0]["last"]["path"]
    old_best = path / committed["members"][0]["best"]["path"]
    old_last_bytes, old_best_bytes = old_last.read_bytes(), old_best.read_bytes()

    def fail_progress(*_args, **_kwargs):
        raise OSError("injected progress publication failure")

    monkeypatch.setattr(module, "_write_progress", fail_progress)
    with pytest.raises(OSError, match="injected progress"):
        workspace.commit_epoch(
            member_index=0,
            seed=101,
            state=_resume_state(module, workspace, epoch=3, best_nll=2.0),
            best_changed=True,
        )
    monkeypatch.undo()

    assert old_last.read_bytes() == old_last_bytes
    assert old_best.read_bytes() == old_best_bytes
    workspace.commit_epoch(
        member_index=0,
        seed=101,
        state=_resume_state(module, workspace, epoch=3, best_nll=2.0),
        best_changed=True,
    )
    restored = module._ResumeWorkspace(path, _workspace_identity(module))
    resumed = restored.load_member(member_index=0, seed=101, hidden=(8,))
    assert resumed is not None
    assert resumed.state["epoch"] == 3


def test_resume_workspace_rejects_symlink_parent_without_external_write(tmp_path: Path):
    module = _trainer_module()
    external = tmp_path / "external"
    external.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module._ResumeWorkspace(linked_parent / ".output.resume-v1", _workspace_identity(module))

    assert list(external.iterdir()) == []


def test_resume_workspace_rejects_symlink_checkpoint_directory_without_external_write(tmp_path: Path):
    module = _trainer_module()
    path = tmp_path / ".output.resume-v1"
    workspace = module._ResumeWorkspace(path, _workspace_identity(module))
    checkpoints = path / "checkpoints"
    if checkpoints.exists():
        checkpoints.rmdir()
    external = tmp_path / "external-checkpoints"
    external.mkdir()
    checkpoints.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        workspace.commit_epoch(
            member_index=0,
            seed=101,
            state=_resume_state(module, workspace, epoch=1, best_nll=2.0),
            best_changed=True,
        )

    assert list(external.iterdir()) == []


def test_main_rejects_symlink_output_parent_before_external_write(tmp_path: Path):
    module = _trainer_module()
    external = tmp_path / "external-output"
    external.mkdir()
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.main(_smoke_args(linked_parent / "expert"))

    assert list(external.iterdir()) == []


def test_dataset_manifest_rejects_symlink_parent(tmp_path: Path):
    module = _trainer_module()
    external = tmp_path / "external-dataset"
    external.mkdir()
    (external / "aggregate_manifest.json").write_text("{}", encoding="utf-8")
    linked_parent = tmp_path / "linked-dataset"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module._resolve_manifest(str(linked_parent / "aggregate_manifest.json"))


def test_explicit_resume_rejects_symlink_parent(tmp_path: Path):
    module = _trainer_module()
    external = tmp_path / "external-resume"
    external.mkdir()
    linked_parent = tmp_path / "linked-resume"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module._load_resume(
            linked_parent,
            member_index=0,
            seed=101,
            hidden=(8,),
            aggregate_sha="a" * 64,
            expected_member_seeds=(101, 202),
        )


def test_cpu_seed_and_preflight_never_probe_or_seed_cuda(monkeypatch: pytest.MonkeyPatch):
    module = _trainer_module()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU path touched CUDA")

    monkeypatch.setattr(module.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(module.torch.cuda, "is_initialized", forbidden)
    monkeypatch.setattr(module.torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(module.torch, "manual_seed", forbidden)

    module._configure_cuda_determinism("cpu")
    module._set_seed(42)


def test_cuda_preflight_rejects_when_cuda_was_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _trainer_module()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(module.torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="already initialized"):
        module._configure_cuda_determinism("cuda:0")

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_training_identity_binds_semantic_sources_cuda_build_device_and_cublas(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _trainer_module()
    fingerprint = {"type": "cuda", "uuid": "GPU-test", "compute_capability": [12, 0]}
    monkeypatch.setattr(module, "_device_fingerprint", lambda _device: fingerprint, raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    first = module._training_identity(
        aggregate_sha="a" * 64,
        member_seeds=(101, 202), hidden=(8,), epochs=2, batch_size=8,
        learning_rate=1e-3, acceleration_weight=1e-5, jerk_weight=1e-7,
        device=module.torch.device("cuda:0"), synthetic_smoke=False,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    second = module._training_identity(
        aggregate_sha="a" * 64,
        member_seeds=(101, 202), hidden=(8,), epochs=2, batch_size=8,
        learning_rate=1e-3, acceleration_weight=1e-5, jerk_weight=1e-7,
        device=module.torch.device("cuda:0"), synthetic_smoke=False,
    )

    assert first["device"] == fingerprint
    assert first["software"]["torch_cuda_build"] == module.torch.version.cuda
    assert first["software"]["numpy_version"] == module.np.__version__
    assert set(first["training_semantics"]["source_sha256"]) == {"trainer", "model", "contracts"}
    assert first["training_semantics"]["aggregate_sha256"] == module._training_semantics_sha256(
        first["training_semantics"]["source_sha256"]
    )
    assert first["cublas_workspace_config"] == ":4096:8"
    assert second["cublas_workspace_config"] == ":16:8"
    assert module._identity_sha256(first) != module._identity_sha256(second)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert module._training_identity(
        aggregate_sha="a" * 64,
        member_seeds=(101, 202), hidden=(8,), epochs=2, batch_size=8,
        learning_rate=1e-3, acceleration_weight=1e-5, jerk_weight=1e-7,
        device=module.torch.device("cuda:0"), synthetic_smoke=False,
    ) == first


@pytest.mark.parametrize(
    "option,value",
    [
        ("--learning-rate", "nan"),
        ("--learning-rate", "inf"),
        ("--learning-rate", "-inf"),
        ("--learning-rate", "0"),
        ("--learning-rate", "-1"),
        ("--acceleration-weight", "nan"),
        ("--acceleration-weight", "inf"),
        ("--acceleration-weight", "-inf"),
        ("--acceleration-weight", "-1"),
        ("--jerk-weight", "nan"),
        ("--jerk-weight", "inf"),
        ("--jerk-weight", "-inf"),
        ("--jerk-weight", "-1"),
    ],
)
def test_nonfinite_hyperparameters_reject_before_any_output_state(
    tmp_path: Path, option: str, value: str
):
    module = _trainer_module()
    output = tmp_path / f"rejected-{option.removeprefix('--')}"

    argument = f"{option}={value}" if value.startswith("-") else None
    with pytest.raises(ValueError, match="training values"):
        module.main([*_smoke_args(output), *([argument] if argument else [option, value])])

    assert not output.exists()
    assert not module._resume_workspace_path(output).exists()
    assert not list(tmp_path.glob(f".{output.name}.train-*"))


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


@pytest.mark.parametrize("mutation", ["subset", "extra", "duplicate", "reorder"])
@pytest.mark.parametrize("component", ["top_level_seeds", "member_records"])
@pytest.mark.parametrize("legacy", [False, True], ids=["identified", "legacy"])
def test_resume_requires_exact_manifest_member_roster_before_import(
    tmp_path: Path,
    mutation: str,
    component: str,
    legacy: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _trainer_module()
    initial = tmp_path / "member-roster-source"
    module.main(_smoke_args(initial))
    manifest_path = initial / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if legacy:
        manifest.pop("training_identity")
        manifest.pop("training_identity_sha256")
    target = manifest["member_seeds"] if component == "top_level_seeds" else manifest["members"]
    if mutation == "subset":
        target.pop()
    elif mutation == "extra":
        if component == "top_level_seeds":
            target.append(303)
        else:
            extra = dict(manifest["members"][-1])
            extra.update(
                member_index=2,
                seed=303,
                checkpoint="checkpoints/member-02-best.pt",
                checkpoint_sha256="0" * 64,
            )
            target.append(extra)
    elif mutation == "duplicate":
        if component == "top_level_seeds":
            target[1] = target[0]
        else:
            target[1]["checkpoint"] = target[0]["checkpoint"]
    elif mutation == "reorder":
        target.reverse()
    else:  # pragma: no cover - pytest owns the closed mutation set
        raise AssertionError(mutation)
    body = dict(manifest)
    body.pop("ensemble_manifest_sha256")
    manifest["ensemble_manifest_sha256"] = module._ensemble_manifest_sha256(body)
    module._atomic_bytes(manifest_path, module._canonical_json(manifest))
    destination = tmp_path / "member-roster-rejected"

    def forbidden_import(*_args, **_kwargs):
        raise AssertionError("member import occurred before whole-roster validation")

    monkeypatch.setattr(module._ResumeWorkspace, "import_member", forbidden_import)

    with pytest.raises(ValueError, match="member roster"):
        module.main(
            [*_smoke_args(destination), "--resume-checkpoint", str(initial)]
        )

    assert not destination.exists()
    assert not module._resume_workspace_path(destination).exists()


def test_pre_resume_workspace_legacy_completed_ensemble_is_safely_migrated(tmp_path: Path):
    module = _trainer_module()
    dataset_stage = tmp_path / "legacy-dataset"
    manifest_path = module._synthetic_manifest(dataset_stage)
    aggregate_sha = json.loads(manifest_path.read_text(encoding="utf-8"))["aggregate_sha256"]
    legacy = tmp_path / "legacy-ensemble"
    checkpoint_shas: list[str] = []
    for member_index, seed in enumerate((101, 202)):
        module._set_seed(seed)
        model = module.FingertipMixtureNet(hidden=(8,))
        optimizer = module.torch.optim.AdamW(model.parameters(), lr=1e-3)
        best_state = {
            "format_version": 1,
            "member_index": member_index,
            "seed": seed,
            "hidden": (8,),
            "epoch": 2,
            "best_validation_nll": 1.0,
            "dataset_aggregate_sha256": aggregate_sha,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        checkpoint = legacy / "checkpoints" / f"member-{member_index:02d}-best.pt"
        module._atomic_torch_save(checkpoint, best_state)
        checkpoint_shas.append(sha256(checkpoint.read_bytes()).hexdigest())
    fixture = LEGACY_MANIFEST_FIXTURE.read_text(encoding="utf-8")
    body = json.loads(
        fixture.replace("$DATASET_AGGREGATE_SHA256", aggregate_sha)
        .replace("$MEMBER_00_SHA256", checkpoint_shas[0])
        .replace("$MEMBER_01_SHA256", checkpoint_shas[1])
    )
    module._atomic_bytes(
        legacy / "ensemble_manifest.json",
        module._canonical_json(
            {**body, "ensemble_manifest_sha256": module._ensemble_manifest_sha256(body)}
        ),
    )
    output = tmp_path / "migrated"

    module.main([*_smoke_args(output), "--resume-checkpoint", str(legacy)])

    migrated = json.loads((output / "ensemble_manifest.json").read_text(encoding="utf-8"))
    assert isinstance(migrated["training_identity"], dict)
    assert len(migrated["training_identity_sha256"]) == 64
    for record in migrated["members"]:
        state = module.torch.load(output / record["checkpoint"], weights_only=True)
        assert state["training_identity_sha256"] == migrated["training_identity_sha256"]
        assert state["epoch"] == 2
    assert not module._resume_workspace_path(output).exists()


def test_legacy_ensemble_cannot_continue_from_an_earlier_best_epoch(tmp_path: Path):
    module = _trainer_module()
    dataset_stage = tmp_path / "legacy-dataset"
    manifest_path = module._synthetic_manifest(dataset_stage)
    aggregate_sha = json.loads(manifest_path.read_text(encoding="utf-8"))["aggregate_sha256"]
    legacy = tmp_path / "legacy-ensemble"
    records = []
    for member_index, seed in enumerate((101, 202)):
        module._set_seed(seed)
        model = module.FingertipMixtureNet(hidden=(8,))
        optimizer = module.torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "format_version": 1,
            "member_index": member_index,
            "seed": seed,
            "hidden": (8,),
            "epoch": 1,
            "best_validation_nll": 1.0,
            "dataset_aggregate_sha256": aggregate_sha,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        checkpoint = legacy / "checkpoints" / f"member-{member_index:02d}-best.pt"
        module._atomic_torch_save(checkpoint, state)
        records.append(
            {
                "member_index": member_index,
                "seed": seed,
                "best_validation_nll": 1.0,
                "checkpoint": f"checkpoints/member-{member_index:02d}-best.pt",
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    body = {
        "format_version": 1,
        "dataset_aggregate_sha256": aggregate_sha,
        "member_seeds": [101, 202],
        "hidden": [8],
        "members": records,
        "metrics": {},
        "synthetic_smoke": True,
        "production_deployable": False,
    }
    module._atomic_bytes(
        legacy / "ensemble_manifest.json",
        module._canonical_json(
            {**body, "ensemble_manifest_sha256": module._ensemble_manifest_sha256(body)}
        ),
    )

    with pytest.raises(ValueError, match="requested final epoch"):
        module.main(
            [*_smoke_args(tmp_path / "must-not-continue"), "--resume-checkpoint", str(legacy)]
        )


def test_legacy_manifest_fixture_is_valid_json_and_contains_only_relative_paths():
    document = json.loads(LEGACY_MANIFEST_FIXTURE.read_text(encoding="utf-8"))

    assert set(document) == {
        "format_version",
        "dataset_aggregate_sha256",
        "member_seeds",
        "hidden",
        "members",
        "metrics",
        "synthetic_smoke",
        "production_deployable",
    }
    assert document["format_version"] == 1
    for record in document["members"]:
        checkpoint = Path(record["checkpoint"])
        assert not checkpoint.is_absolute()
        assert ".." not in checkpoint.parts
    serialized = json.dumps(document, sort_keys=True)
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized


def test_resume_checkpoint_with_valid_hash_but_invalid_payload_is_safely_rejected(
    tmp_path: Path,
):
    module = _trainer_module()
    initial = tmp_path / "invalid-payload"
    module.main([*_smoke_args(initial), "--epochs", "1"])
    manifest_path = initial / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = initial / manifest["members"][0]["checkpoint"]
    checkpoint.write_bytes(b"not a torch checkpoint")
    manifest["members"][0]["checkpoint_sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    body = dict(manifest)
    body.pop("ensemble_manifest_sha256")
    manifest["ensemble_manifest_sha256"] = module._ensemble_manifest_sha256(body)
    module._atomic_bytes(manifest_path, module._canonical_json(manifest))

    with pytest.raises(ValueError, match="safely loaded"):
        module._load_resume(
            initial,
            member_index=0,
            seed=101,
            hidden=(8,),
            aggregate_sha=manifest["dataset_aggregate_sha256"],
            expected_member_seeds=(101, 202),
            training_identity_sha=manifest["training_identity_sha256"],
        )


def test_workspace_checkpoint_with_valid_hash_but_invalid_payload_is_safely_rejected(
    tmp_path: Path,
):
    module = _trainer_module()
    path = tmp_path / ".invalid-workspace.resume-v1"
    identity = _workspace_identity(module)
    workspace = module._ResumeWorkspace(path, identity)
    workspace.commit_epoch(
        member_index=0,
        seed=101,
        state=_resume_state(module, workspace, epoch=1, best_nll=1.0),
        best_changed=True,
    )
    progress_path = path / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    checkpoint = path / progress["members"][0]["last"]["path"]
    checkpoint.write_bytes(b"not a torch checkpoint")
    progress["members"][0]["last"]["sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    body = dict(progress)
    body.pop("progress_sha256")
    progress["progress_sha256"] = sha256(module._canonical_json(body)).hexdigest()
    module._atomic_bytes(progress_path, module._canonical_json(progress))

    restored = module._ResumeWorkspace(path, identity)
    with pytest.raises(ValueError, match="safely loaded"):
        restored.load_member(member_index=0, seed=101, hidden=(8,))


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
            expected_member_seeds=(1701, 2718, 3141),
        )
