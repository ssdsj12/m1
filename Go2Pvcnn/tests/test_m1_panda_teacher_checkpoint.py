from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from rsl_rl.modules import ActorCritic

from go2_pvcnn.tasks.m1_panda_teacher import (
    M1PandaDisturbanceCfg,
    stage_disturbance_cfg,
)
from go2_pvcnn.tasks.m1_residual_action import M1ResidualActionComposerCfg
from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
    atomic_write_manifest,
    build_run_manifest,
    file_sha256,
    load_frozen_teacher_actor,
    load_manifest_for_checkpoint,
    module_sha256,
    validate_teacher_checkpoint,
)


def test_manifest_is_atomic_and_checkpoint_hash_is_stable(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "run_manifest.json"

    atomic_write_manifest(manifest, {"schema_version": 1, "stage": "A0"})

    assert json.loads(manifest.read_text())["stage"] == "A0"
    assert file_sha256(checkpoint) == hashlib.sha256(b"checkpoint").hexdigest()
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


def test_load_manifest_uses_checkpoint_parent(tmp_path):
    checkpoint = tmp_path / "model_3.pt"
    checkpoint.write_bytes(b"x")
    atomic_write_manifest(
        tmp_path / "run_manifest.json",
        {"schema_version": 1, "stage": "A1", "observation_dim": 60},
    )

    manifest = load_manifest_for_checkpoint(checkpoint)

    assert manifest == {
        "observation_dim": 60,
        "schema_version": 1,
        "stage": "A1",
    }


@pytest.mark.parametrize("payload", [None, [], "A0", 1])
def test_manifest_requires_a_dictionary(tmp_path, payload):
    with pytest.raises(TypeError, match="payload must be a dictionary"):
        atomic_write_manifest(tmp_path / "run_manifest.json", payload)


def test_atomic_manifest_removes_temporary_file_after_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "run_manifest.json"

    def fail_replace(source, destination):
        assert Path(destination) == target
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        atomic_write_manifest(target, {"stage": "A0"})

    assert not target.exists()
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


def test_missing_checkpoint_or_manifest_is_rejected(tmp_path):
    missing_checkpoint = tmp_path / "model_0.pt"
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        file_sha256(missing_checkpoint)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_manifest_for_checkpoint(missing_checkpoint)

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="run_manifest.json"):
        load_manifest_for_checkpoint(checkpoint)


def test_manifest_must_decode_to_a_dictionary(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"x")
    (tmp_path / "run_manifest.json").write_text("[]")

    with pytest.raises(ValueError, match="JSON object"):
        load_manifest_for_checkpoint(checkpoint)


def _actor(*, actor_obs=60, critic_obs=60, actions=16, hidden=(256, 128)):
    return ActorCritic(
        actor_obs,
        critic_obs,
        actions,
        actor_hidden_dims=list(hidden),
        critic_hidden_dims=list(hidden),
        activation="elu",
        init_noise_std=0.01,
    )


def _write_checkpoint(
    tmp_path,
    *,
    stage="A0",
    observation_dim=60,
    action_dim=16,
    hidden=(256, 128),
    state_dict=None,
    include_model=True,
    include_optimizer=True,
    base_checkpoint_sha256=None,
):
    checkpoint_path = tmp_path / "model_0.pt"
    checkpoint = {"iter": 0, "infos": None}
    if include_model:
        checkpoint["model_state_dict"] = (
            _actor().state_dict() if state_dict is None else state_dict
        )
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "actor_hidden_dims": list(hidden),
    }
    if base_checkpoint_sha256 is not None:
        manifest["base_checkpoint_sha256"] = base_checkpoint_sha256
    atomic_write_manifest(tmp_path / "run_manifest.json", manifest)
    return checkpoint_path


def test_validate_teacher_checkpoint_accepts_exact_a0_contract(tmp_path):
    path = _write_checkpoint(tmp_path)

    checkpoint, manifest = validate_teacher_checkpoint(
        path,
        expected_stage="A0",
        expected_observation_dim=60,
        expected_action_dim=16,
        expected_actor_hidden_dims=(256, 128),
        require_optimizer=True,
    )

    assert checkpoint["iter"] == 0
    assert manifest["stage"] == "A0"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("stage", "A1", "stage"),
        ("observation_dim", 572, "observation_dim"),
        ("action_dim", 12, "action_dim"),
        ("actor_hidden_dims", [512, 128], "actor_hidden_dims"),
    ],
)
def test_validate_teacher_checkpoint_rejects_manifest_contract_drift(
    tmp_path, field, value, message
):
    path = _write_checkpoint(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    atomic_write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        validate_teacher_checkpoint(
            path,
            expected_stage="A0",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
        )


def test_validate_teacher_checkpoint_rejects_missing_model_state(tmp_path):
    path = _write_checkpoint(tmp_path, include_model=False)
    with pytest.raises(ValueError, match="model_state_dict"):
        validate_teacher_checkpoint(
            path,
            expected_stage="A0",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
        )


def test_resume_validation_requires_optimizer_state(tmp_path):
    path = _write_checkpoint(tmp_path, include_optimizer=False)
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        validate_teacher_checkpoint(
            path,
            expected_stage="A0",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
            require_optimizer=True,
        )


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    [
        ("actor.0.weight", torch.zeros(256, 572), "actor.0.weight"),
        ("actor.4.weight", torch.zeros(12, 128), "actor.4.weight"),
        ("critic.0.weight", torch.zeros(256, 59), "critic.0.weight"),
        ("critic.4.weight", torch.zeros(2, 128), "critic.4.weight"),
        ("std", torch.zeros(12), "std"),
    ],
)
def test_validate_teacher_checkpoint_rejects_actual_tensor_shape_drift(
    tmp_path, key, replacement, message
):
    state = _actor().state_dict()
    state[key] = replacement
    path = _write_checkpoint(tmp_path, state_dict=state)

    with pytest.raises(ValueError, match=message):
        validate_teacher_checkpoint(
            path,
            expected_stage="A0",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
        )


def test_validate_teacher_checkpoint_rejects_missing_required_tensor(tmp_path):
    state = _actor().state_dict()
    state.pop("actor.2.bias")
    path = _write_checkpoint(tmp_path, state_dict=state)

    with pytest.raises(ValueError, match="actor.2.bias"):
        validate_teacher_checkpoint(
            path,
            expected_stage="A0",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
        )


def test_a1_resume_contract_requires_matching_base_hash(tmp_path):
    path = _write_checkpoint(
        tmp_path, stage="A1", base_checkpoint_sha256="recorded-base-hash"
    )

    with pytest.raises(ValueError, match="base_checkpoint_sha256"):
        validate_teacher_checkpoint(
            path,
            expected_stage="A1",
            expected_observation_dim=60,
            expected_action_dim=16,
            expected_actor_hidden_dims=(256, 128),
            expected_base_sha256="different-base-hash",
        )


def test_module_sha256_is_stable_and_changes_with_parameter_data():
    actor = _actor()
    initial = module_sha256(actor)

    actor.act_inference(torch.zeros(2, 60))

    assert module_sha256(actor) == initial
    with torch.no_grad():
        next(actor.parameters()).add_(1.0)
    assert module_sha256(actor) != initial


def _policy_cfg():
    return {
        "class_name": "ActorCritic",
        "init_noise_std": 0.01,
        "noise_std_type": "log",
        "state_dependent_std": False,
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
        "activation": "elu",
    }


def test_load_frozen_teacher_actor_is_strict_eval_and_immutable(tmp_path):
    path = _write_checkpoint(tmp_path)

    actor = load_frozen_teacher_actor(path, device="cpu", policy_cfg=_policy_cfg())
    initial = module_sha256(actor)
    output = actor.act_inference(torch.zeros(3, 60))

    assert isinstance(actor, ActorCritic)
    assert actor.training is False
    assert all(parameter.requires_grad is False for parameter in actor.parameters())
    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()
    assert module_sha256(actor) == initial


def test_load_frozen_teacher_actor_rejects_unexpected_state_key(tmp_path):
    state = _actor().state_dict()
    state["unexpected.weight"] = torch.zeros(1)
    path = _write_checkpoint(tmp_path, state_dict=state)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_frozen_teacher_actor(path, device="cpu", policy_cfg=_policy_cfg())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_hidden_dims", [512, 128]),
        ("critic_hidden_dims", [256, 64]),
        ("activation", "relu"),
    ],
)
def test_load_frozen_teacher_actor_rejects_policy_contract_drift(
    tmp_path, field, value
):
    path = _write_checkpoint(tmp_path)
    policy = _policy_cfg()
    policy[field] = value

    with pytest.raises(ValueError, match="policy_cfg"):
        load_frozen_teacher_actor(path, device="cpu", policy_cfg=policy)


def test_build_a0_run_manifest_contains_complete_runtime_contract(tmp_path):
    resume = _write_checkpoint(tmp_path)
    composer = M1ResidualActionComposerCfg()
    disturbance = stage_disturbance_cfg("A0")

    manifest = build_run_manifest(
        stage="A0",
        task_id="Isaac-M1-Panda-Teacher-A0-v0",
        seed=42,
        composer_cfg=composer,
        disturbance_cfg=disturbance,
        resume_checkpoint=resume,
    )

    assert manifest["schema_version"] == 1
    assert manifest["stage"] == "A0"
    assert manifest["task_id"] == "Isaac-M1-Panda-Teacher-A0-v0"
    assert manifest["observation_dim"] == 60
    assert manifest["action_dim"] == 16
    assert manifest["actor_hidden_dims"] == [256, 128]
    assert manifest["seed"] == 42
    assert manifest["composer"]["leg_residual_limit_rad"] == pytest.approx(0.05)
    assert manifest["disturbance"]["force_limit_n"] == [10.0, 10.0, 10.0]
    assert manifest["checkpoint_pattern"] == "model_<iteration>.pt"
    assert manifest["base_checkpoint"] is None
    assert manifest["base_checkpoint_sha256"] is None
    assert manifest["frozen_actor_initial_sha256"] is None
    assert manifest["resume_checkpoint"] == str(resume.resolve())
    assert manifest["status"] == "running"


def test_build_a1_run_manifest_records_base_path_hash_and_frozen_hash(tmp_path):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-checkpoint")
    frozen = _actor()

    manifest = build_run_manifest(
        stage="A1",
        task_id="Isaac-M1-Panda-Teacher-A1-v0",
        seed=7,
        composer_cfg=M1ResidualActionComposerCfg(),
        disturbance_cfg=stage_disturbance_cfg("A1"),
        base_checkpoint=base,
        frozen_actor=frozen,
    )

    assert manifest["base_checkpoint"] == str(base.resolve())
    assert manifest["base_checkpoint_sha256"] == hashlib.sha256(
        b"base-checkpoint"
    ).hexdigest()
    assert manifest["frozen_actor_initial_sha256"] == module_sha256(frozen)
    assert manifest["resume_checkpoint"] is None


def test_build_run_manifest_rejects_stage_and_config_mismatch(tmp_path):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")

    with pytest.raises(ValueError, match="A1 requires"):
        build_run_manifest(
            stage="A1",
            task_id="Isaac-M1-Panda-Teacher-A1-v0",
            seed=0,
            composer_cfg=M1ResidualActionComposerCfg(),
            disturbance_cfg=stage_disturbance_cfg("A1"),
        )
    with pytest.raises(ValueError, match="A0 does not accept"):
        build_run_manifest(
            stage="A0",
            task_id="Isaac-M1-Panda-Teacher-A0-v0",
            seed=0,
            composer_cfg=M1ResidualActionComposerCfg(),
            disturbance_cfg=stage_disturbance_cfg("A0"),
            base_checkpoint=base,
            frozen_actor=_actor(),
        )
    with pytest.raises(ValueError, match="disturbance_cfg"):
        build_run_manifest(
            stage="A0",
            task_id="Isaac-M1-Panda-Teacher-A0-v0",
            seed=0,
            composer_cfg=M1ResidualActionComposerCfg(),
            disturbance_cfg=stage_disturbance_cfg("A1"),
        )
