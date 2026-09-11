from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.dexmanipnet import (
    LoadedHandSequence,
    SequenceAudit,
    audit_sequence,
    load_best_successful_rollout,
)


def write_minimal_sequence(
    tmp_path: Path,
    *,
    source: str = "favor",
    side: str = "rh",
    dexhand: str = "inspire",
    rewards: list[list[float]] | None = None,
    seq_len: int = 2,
    q_frames: int | None = None,
    dq_frames: int | None = None,
    joint_dim: int | None = None,
    include_geometry: bool = True,
    include_tip_force: bool = True,
) -> Path:
    sequence = tmp_path / source / "sequences" / "sequence_000"
    sequence.mkdir(parents=True)
    object_geometry = sequence.parents[1] / "ObjURDF" / "object.urdf"
    if include_geometry:
        object_geometry.parent.mkdir()
        object_geometry.write_text('<robot name="object"><link name="object"/></robot>', encoding="utf-8")

    metadata = {
        "seq_len": seq_len,
        "dexhand": dexhand,
        "interaction_mode": "right" if side == "rh" else "left",
        "obj_rh_path": "ObjURDF/object.urdf",
        "obj_lh_path": "ObjURDF/object.urdf",
        # Deliberately sensitive fields: the geometry loader must never copy these.
        "primitive": "task_primitive",
        "description": "private instruction text",
        "oid_rh": "private-object-id",
        "oid_lh": "private-object-id",
        "object_name": "private-object-name",
    }
    (sequence / "seq_info.json").write_text(json.dumps(metadata), encoding="utf-8")

    if joint_dim is None:
        joint_dim = 12 if dexhand == "inspire" else 22
    if rewards is None:
        rewards = [[1.0] * seq_len]
    with h5py.File(sequence / "rollouts.hdf5", "w") as h5:
        successful = h5.create_group("rollouts/successful")
        for index, reward in enumerate(rewards):
            rollout = successful.create_group(f"rollout_{index}")
            rollout.create_dataset("reward", data=np.asarray(reward, dtype=np.float64))
            rollout.create_dataset(
                f"q_{side}", data=np.zeros((q_frames or seq_len, joint_dim), dtype=np.float64)
            )
            rollout.create_dataset(
                f"dq_{side}", data=np.zeros((dq_frames or seq_len, joint_dim), dtype=np.float64)
            )
            rollout.create_dataset(f"state_{side}", data=np.zeros((seq_len, 13), dtype=np.float64))
            rollout.create_dataset(
                f"state_manip_obj_{side}", data=np.zeros((seq_len, 13), dtype=np.float64)
            )
            if include_tip_force:
                rollout.create_dataset(
                    f"tip_force_{side}", data=np.zeros((seq_len, 15), dtype=np.float64)
                )
    return sequence


def _rewrite_dataset(sequence: Path, dataset: str, value: np.ndarray) -> None:
    with h5py.File(sequence / "rollouts.hdf5", "a") as h5:
        del h5[f"rollouts/successful/rollout_0/{dataset}"]
        h5[f"rollouts/successful/rollout_0"].create_dataset(dataset, data=value)


def test_loader_selects_highest_total_reward_without_exposing_task_id(tmp_path):
    sequence = write_minimal_sequence(tmp_path, rewards=[[1.0, 1.0], [0.0, 3.0]])

    loaded = load_best_successful_rollout(sequence, source="favor", side="rh")

    assert isinstance(loaded, LoadedHandSequence)
    assert loaded.rollout_name == "rollout_1"
    assert loaded.total_reward == pytest.approx(3.0)
    assert loaded.q.shape[0] == loaded.dq.shape[0] == 2
    for forbidden in ("task_id", "object_id", "object_name", "primitive", "description", "text"):
        assert not hasattr(loaded, forbidden)
    field_names = {field.name for field in fields(LoadedHandSequence)}
    assert field_names.isdisjoint(
        {"task_id", "object_id", "object_name", "primitive", "description", "text"}
    )


def test_length_mismatch_is_one_atomic_rejection(tmp_path):
    sequence = write_minimal_sequence(tmp_path, q_frames=3, dq_frames=2)

    audit = audit_sequence(sequence, source="favor", side="rh")

    assert isinstance(audit, SequenceAudit)
    assert not audit.accepted
    assert audit.reason == "length_mismatch"
    assert len(audit.input_sha256) == 64
    with pytest.raises(ValueError, match="length_mismatch"):
        load_best_successful_rollout(sequence, source="favor", side="rh")


def test_equal_reward_tie_selects_lexicographically_largest_rollout_name(tmp_path):
    sequence = write_minimal_sequence(tmp_path, rewards=[[1.0, 1.0], [0.5, 1.5]])

    loaded = load_best_successful_rollout(sequence, source="favor", side="rh")

    assert loaded.rollout_name == "rollout_1"


def test_oakinkv2_left_shadow_sequence_uses_exact_registered_dimension(tmp_path):
    sequence = write_minimal_sequence(
        tmp_path, source="oakinkv2", side="lh", dexhand="shadow", joint_dim=22
    )

    loaded = load_best_successful_rollout(sequence, source="oakinkv2", side="lh")
    audit = audit_sequence(sequence, source="oakinkv2", side="lh")

    assert loaded.source_hand_key == "shadow_lh"
    assert loaded.q.shape == (2, 22)
    assert loaded.root_state.shape == loaded.object_state.shape == (2, 13)
    assert loaded.tip_force is not None and loaded.tip_force.shape == (2, 15)
    assert loaded.object_geometry_path.name == "object.urdf"
    assert audit.accepted and audit.reason == "accepted"
    assert audit.frames == 2
    assert len(audit.input_sha256) == 64


def test_tip_force_is_optional_because_mesh_geometry_can_drive_contact_inference(tmp_path):
    sequence = write_minimal_sequence(tmp_path, include_tip_force=False)

    loaded = load_best_successful_rollout(sequence, source="favor", side="rh")

    assert loaded.tip_force is None


@pytest.mark.parametrize(
    ("source", "side", "reason"),
    [
        ("unknown", "rh", "unsupported_source"),
        ("favor", "lh", "unsupported_source_side"),
        ("oakinkv2", "bih", "unsupported_side"),
    ],
)
def test_source_side_mapping_is_strict_and_atomic(tmp_path, source, side, reason):
    sequence = write_minimal_sequence(tmp_path)

    audit = audit_sequence(sequence, source=source, side=side)

    assert not audit.accepted
    assert audit.reason == reason


def test_unknown_source_hand_and_wrong_joint_width_are_rejected(tmp_path):
    unknown = write_minimal_sequence(tmp_path / "unknown", dexhand="allegro", joint_dim=16)
    wrong_width = write_minimal_sequence(tmp_path / "width", joint_dim=11)

    assert audit_sequence(unknown, source="favor", side="rh").reason == "unsupported_hand"
    assert audit_sequence(wrong_width, source="favor", side="rh").reason == "joint_dimension_mismatch"


@pytest.mark.parametrize(
    ("dataset", "value", "reason"),
    [
        ("q_rh", np.full((2, 12), np.nan), "non_finite"),
        ("state_rh", np.zeros((2, 12)), "root_state_shape"),
        ("state_manip_obj_rh", np.zeros((2, 7)), "object_state_shape"),
        ("tip_force_rh", np.zeros((2, 14)), "tip_force_shape"),
    ],
)
def test_geometry_arrays_have_strict_shapes_and_finite_values(tmp_path, dataset, value, reason):
    sequence = write_minimal_sequence(tmp_path)
    _rewrite_dataset(sequence, dataset, value)

    audit = audit_sequence(sequence, source="favor", side="rh")

    assert not audit.accepted
    assert audit.reason == reason


def test_missing_or_escaping_object_geometry_is_rejected(tmp_path):
    missing = write_minimal_sequence(tmp_path / "missing", include_geometry=False)
    escaping = write_minimal_sequence(tmp_path / "escaping")
    info_path = escaping / "seq_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["obj_rh_path"] = "../outside.urdf"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert audit_sequence(missing, source="favor", side="rh").reason == "missing_object_geometry"
    assert audit_sequence(escaping, source="favor", side="rh").reason == "invalid_object_geometry"


def test_invalid_json_and_empty_successful_group_are_rejected(tmp_path):
    invalid_json = write_minimal_sequence(tmp_path / "json")
    (invalid_json / "seq_info.json").write_text("[]", encoding="utf-8")
    empty = write_minimal_sequence(tmp_path / "empty")
    with h5py.File(empty / "rollouts.hdf5", "w") as h5:
        h5.create_group("rollouts/successful")

    assert audit_sequence(invalid_json, source="favor", side="rh").reason == "invalid_seq_info"
    assert audit_sequence(empty, source="favor", side="rh").reason == "no_successful_rollout"


def test_all_successful_rollouts_are_audited_before_best_is_returned(tmp_path):
    sequence = write_minimal_sequence(tmp_path, rewards=[[4.0, 4.0], [0.0, 1.0]])
    _rewrite_dataset(sequence, "reward", np.array([np.inf, 1.0]))

    audit = audit_sequence(sequence, source="favor", side="rh")

    assert not audit.accepted
    assert audit.reason == "non_finite"
