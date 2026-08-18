import dataclasses
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.student_model import (
    M1PandaStudent,
    StudentNetworkCfg,
)
from go2_pvcnn.tasks.m1_panda_student_checkpoint import (
    StudentCheckpointManifest,
    load_student_checkpoint,
    save_student_checkpoint,
)


def _manifest(**overrides) -> StudentCheckpointManifest:
    values = dict(
        schema_version=1,
        asset_sha="643fd061-zero-clearance",
        teacher_commit="teacher-commit",
        dataset_sha="dataset-sha",
        observation_dim=100,
        history_length=10,
        action_dim=23,
        action_scales={
            "leg_position_rad": 0.25,
            "wheel_velocity_radps": 8.0,
            "arm_position_rad": 0.2,
            "leg_slew_per_step": 0.02,
            "wheel_slew_per_step": 0.5,
            "arm_slew_per_step": 0.01,
        },
        control_dt=0.005,
        dagger_stage="teacher-pretrain",
        teacher_probability=1.0,
        model_config=dataclasses.asdict(StudentNetworkCfg()),
        loss_weights={
            "action": 1.0,
            "wrench": 0.25,
            "safety": 0.25,
            "slew": 0.05,
            "saturation": 0.05,
            "hard_sample_multiplier": 2.0,
        },
    )
    values.update(overrides)
    return StudentCheckpointManifest(**values)


def _model_optimizer():
    model = M1PandaStudent(StudentNetworkCfg())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(2, 10, 100)).action.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_checkpoint_round_trip_is_strict(tmp_path):
    torch.manual_seed(13)
    source, source_optimizer = _model_optimizer()
    manifest = _manifest()
    path = tmp_path / "student-s1.pt"
    save_student_checkpoint(path, source, source_optimizer, manifest, global_step=123)

    restored, restored_optimizer = _model_optimizer()
    loaded = load_student_checkpoint(
        path, restored, restored_optimizer, expected=manifest
    )
    assert loaded.global_step == 123
    assert loaded.manifest == manifest
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert restored_optimizer.state_dict()["state"].keys() == source_optimizer.state_dict()["state"].keys()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("asset_sha", "old-10mm-sha"),
        ("teacher_commit", "wrong"),
        ("dataset_sha", "wrong-dataset"),
        ("observation_dim", 99),
        ("history_length", 9),
        ("action_dim", 16),
        ("control_dt", 0.01),
        ("dagger_stage", "wrong-stage"),
        ("teacher_probability", 0.5),
    ],
)
def test_checkpoint_rejects_incompatible_manifest(tmp_path, field, value):
    source, optimizer = _model_optimizer()
    accepted = _manifest()
    path = tmp_path / "student-s1.pt"
    save_student_checkpoint(path, source, optimizer, accepted, global_step=1)
    rejected = dataclasses.replace(accepted, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_student_checkpoint(path, M1PandaStudent(StudentNetworkCfg()), expected=rejected)


def test_checkpoint_rejects_action_scale_loss_and_model_config_mismatch(tmp_path):
    source, optimizer = _model_optimizer()
    accepted = _manifest()
    path = tmp_path / "student-s1.pt"
    save_student_checkpoint(path, source, optimizer, accepted, global_step=1)
    replacements = {
        "action_scales": {**accepted.action_scales, "arm_position_rad": 0.3},
        "loss_weights": {**accepted.loss_weights, "action": 2.0},
        "model_config": {**accepted.model_config, "gru_hidden_dim": 64},
    }
    for field, value in replacements.items():
        with pytest.raises(ValueError, match=field):
            load_student_checkpoint(
                path,
                M1PandaStudent(StudentNetworkCfg()),
                expected=dataclasses.replace(accepted, **{field: value}),
            )


def test_checkpoint_rejects_nonfinite_weights_before_publishing(tmp_path):
    model, optimizer = _model_optimizer()
    next(model.parameters()).data[0, 0] = torch.nan
    path = tmp_path / "student-s1.pt"
    with pytest.raises(ValueError, match="finite"):
        save_student_checkpoint(path, model, optimizer, _manifest(), global_step=2)
    assert not path.exists()
    assert not Path(f"{path}.manifest.json").exists()


def test_resume_rejects_missing_or_nonfinite_optimizer_state(tmp_path):
    model, optimizer = _model_optimizer()
    path = tmp_path / "student-s1.pt"
    manifest = _manifest()
    save_student_checkpoint(path, model, optimizer, manifest, global_step=2)
    payload = torch.load(path, weights_only=False)
    payload.pop("optimizer_state_dict")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer_state_dict"):
        load_student_checkpoint(
            path, M1PandaStudent(StudentNetworkCfg()), _model_optimizer()[1], expected=manifest
        )

    save_student_checkpoint(path, model, optimizer, manifest, global_step=2)
    payload = torch.load(path, weights_only=False)
    state = next(iter(payload["optimizer_state_dict"]["state"].values()))
    state["exp_avg"].flatten()[0] = torch.inf
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_student_checkpoint(
            path, M1PandaStudent(StudentNetworkCfg()), _model_optimizer()[1], expected=manifest
        )


def test_checkpoint_rejects_model_shape_mismatch_without_partial_load(tmp_path):
    model, optimizer = _model_optimizer()
    path = tmp_path / "student-s1.pt"
    manifest = _manifest()
    save_student_checkpoint(path, model, optimizer, manifest, global_step=2)
    payload = torch.load(path, weights_only=False)
    payload["model_state_dict"]["actor.4.weight"] = torch.zeros(22, 128)
    torch.save(payload, path)
    target = M1PandaStudent(StudentNetworkCfg())
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="actor.4.weight"):
        load_student_checkpoint(path, target, expected=manifest)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_and_manifest_are_atomic_and_canonical(tmp_path):
    model, optimizer = _model_optimizer()
    path = tmp_path / "student-s1.pt"
    manifest = _manifest()
    save_student_checkpoint(path, model, optimizer, manifest, global_step=7)
    manifest_path = Path(f"{path}.manifest.json")
    assert path.is_file() and manifest_path.is_file()
    expected_json = json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True) + "\n"
    assert manifest_path.read_text() == expected_json
    assert list(tmp_path.glob(".*.tmp")) == []


def test_inference_load_allows_optimizer_to_be_omitted(tmp_path):
    source, optimizer = _model_optimizer()
    path = tmp_path / "student-s1.pt"
    manifest = _manifest()
    save_student_checkpoint(path, source, optimizer, manifest, global_step=9)
    target = M1PandaStudent(StudentNetworkCfg())
    loaded = load_student_checkpoint(path, target, expected=manifest)
    assert loaded.global_step == 9
