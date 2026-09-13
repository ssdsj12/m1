from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    PRIOR_HORIZON,
    PriorPhase,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.preprocess import (
    canonicalize_left,
    convert_loaded_sequence,
    decanonicalize_left,
    infer_contact_hysteresis,
    infer_prior_phase,
    load_object_collision_mesh,
    object_geometry_sha256,
    object_relative_surface_kinematics,
    resample_fingertips,
    resample_fingertips_with_velocity,
    windows_from_sequence,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.dexmanipnet import (
    LoadedHandSequence,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import SOURCE_HANDS
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.urdf_fk import (
    UrdfKinematicTree,
)


def test_left_mirror_is_exact_involution_and_rejects_bad_geometry():
    points = np.arange(45, dtype=np.float64).reshape(3, 5, 3) / 100.0

    canonical = canonicalize_left(points)

    np.testing.assert_array_equal(canonical[..., 0], points[..., 0])
    np.testing.assert_array_equal(canonical[..., 1], -points[..., 1])
    np.testing.assert_array_equal(canonical[..., 2], points[..., 2])
    np.testing.assert_array_equal(decanonicalize_left(canonical), points)
    with pytest.raises(ValueError, match="trailing shape"):
        canonicalize_left(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        canonicalize_left(np.full((1, 5, 3), np.nan))


def test_cubic_resampling_uses_in_interval_100hz_timestamps_and_analytic_derivative():
    source_hz = 60
    target_hz = 100
    time = np.arange(7, dtype=np.float64) / source_hz
    points = np.empty((time.size, 5, 3), dtype=np.float64)
    for finger in range(5):
        points[:, finger, 0] = time**3 + finger
        points[:, finger, 1] = 2.0 * time**2
        points[:, finger, 2] = -3.0 * time

    positions, velocities, timestamps = resample_fingertips_with_velocity(
        points, source_hz=source_hz, target_hz=target_hz
    )

    np.testing.assert_array_equal(timestamps, np.arange(11, dtype=np.float64) / target_hz)
    assert timestamps[-1] == pytest.approx(time[-1])
    np.testing.assert_allclose(positions[:, 0, 0], timestamps**3, atol=1e-12)
    np.testing.assert_allclose(velocities[:, 0, 0], 3.0 * timestamps**2, atol=1e-12)
    np.testing.assert_allclose(velocities[:, :, 2], -3.0, atol=1e-12)
    np.testing.assert_array_equal(
        resample_fingertips(points, source_hz=source_hz, target_hz=target_hz), positions
    )


@pytest.mark.parametrize(
    "points,source_hz,target_hz",
    [
        (np.zeros((1, 5, 3)), 60, 100),
        (np.zeros((2, 4, 3)), 60, 100),
        (np.full((2, 5, 3), np.inf), 60, 100),
        (np.zeros((2, 5, 3)), 0, 100),
        (np.zeros((2, 5, 3)), 60, True),
    ],
)
def test_resampling_rejects_unusable_inputs(points, source_hz, target_hz):
    with pytest.raises((TypeError, ValueError)):
        resample_fingertips(points, source_hz=source_hz, target_hz=target_hz)


def test_contact_hysteresis_and_phase_labels_use_geometry_only():
    distance = np.array([[0.02] * 5, [0.001] * 5, [0.003] * 5, [0.02] * 5])
    normal_speed = np.array([[-0.02] * 5, [-0.01] * 5, [0.0] * 5, [0.02] * 5])

    contact = infer_contact_hysteresis(distance, normal_speed, enter_m=0.002, exit_m=0.005)
    phase = infer_prior_phase(contact, fingertip_speed=np.array([0.03, 0.01, 0.0, 0.02]))

    assert contact.dtype == np.bool_
    assert contact[:, 0].tolist() == [False, True, True, False]
    assert phase.tolist() == [
        PriorPhase.APPROACH,
        PriorPhase.PRELOAD,
        PriorPhase.HOLD,
        PriorPhase.RELEASE,
    ]


def test_penetrating_tip_is_contact_even_when_current_sample_is_separating():
    contact = infer_contact_hysteresis(
        np.array([[-0.001] * 5]), np.array([[0.01] * 5]), enter_m=0.002, exit_m=0.005
    )

    assert contact.all()


def test_phase_classifier_exposes_all_seven_task_independent_labels():
    contact = np.array(
        [
            [False] * 5,
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
            [False] * 5,
            [False] * 5,
        ],
        dtype=bool,
    )
    speed = np.array([0.03, 0.01, 0.008, 0.03, 0.0, 0.02, 0.0])

    phase = infer_prior_phase(contact, speed)

    assert phase.tolist() == [
        PriorPhase.APPROACH,
        PriorPhase.PRELOAD,
        PriorPhase.GRASP,
        PriorPhase.MANIPULATE,
        PriorPhase.HOLD,
        PriorPhase.RELEASE,
        PriorPhase.UNKNOWN,
    ]


def test_windows_are_20_future_nodes_and_have_frozen_geometry_only_contract():
    frames = PRIOR_HORIZON + 3
    positions = np.arange(frames * 15, dtype=np.float64).reshape(frames, 5, 3) / 1000
    velocities = positions + 1.0
    contacts = np.zeros((frames, 5), dtype=bool)
    phases = np.full(frames, PriorPhase.APPROACH, dtype=np.uint8)
    source_sha = "a" * 64

    windows = windows_from_sequence(
        positions,
        velocities,
        contacts,
        phases,
        source_group="favor/sequence_001/rh",
        source_sha256=source_sha,
    )

    assert len(windows) == 3
    first = windows[0]
    np.testing.assert_allclose(first.fingertip_position_palm.numpy(), positions[0].astype(np.float32))
    np.testing.assert_allclose(
        first.future_fingertip_velocity_palm.numpy(), velocities[1 : PRIOR_HORIZON + 1]
    )
    assert first.network_input().shape == (42,)
    assert first.source_group == "favor/sequence_001/rh"
    assert first.source_sha256 == source_sha


def test_missing_or_unusable_object_collision_geometry_is_atomic_rejection(tmp_path: Path):
    missing = tmp_path / "missing.urdf"
    empty = tmp_path / "empty.urdf"
    empty.write_text('<robot name="empty"><link name="object"/></robot>', encoding="utf-8")
    unsafe = tmp_path / "unsafe.urdf"
    unsafe.write_text(
        '<robot name="bad"><link name="object"><collision><geometry>'
        '<mesh filename="../escape.obj"/></geometry></collision></link></robot>',
        encoding="utf-8",
    )

    for path in (missing, empty, unsafe):
        with pytest.raises(ValueError, match="object geometry"):
            load_object_collision_mesh(path)


def test_object_box_geometry_is_loaded_without_dataset_semantics(tmp_path: Path):
    urdf = tmp_path / "box.urdf"
    urdf.write_text(
        '<robot name="box"><link name="object"><collision>'
        '<origin xyz="0.1 0 0" rpy="0 0 0"/>'
        '<geometry><box size="0.2 0.4 0.6"/></geometry>'
        '</collision></link></robot>',
        encoding="utf-8",
    )

    mesh = load_object_collision_mesh(urdf)

    assert mesh.is_watertight
    np.testing.assert_allclose(mesh.extents, [0.2, 0.4, 0.6], atol=1e-12)
    assert len(object_geometry_sha256(urdf)) == 64
    np.testing.assert_allclose(mesh.centroid, [0.1, 0.0, 0.0], atol=1e-12)


def test_object_urdf_with_leading_ascii_whitespace_before_xml_declaration_loads(tmp_path: Path):
    urdf = tmp_path / "whitespace-box.urdf"
    urdf.write_text(
        '\n  <?xml version="1.0"?>\n'
        '<robot name="box"><link name="object"><collision><geometry>'
        '<box size="0.2 0.4 0.6"/></geometry></collision></link></robot>',
        encoding="utf-8",
    )

    mesh = load_object_collision_mesh(urdf)

    assert mesh.is_watertight
    np.testing.assert_allclose(mesh.extents, [0.2, 0.4, 0.6], atol=1e-12)
    assert len(object_geometry_sha256(urdf)) == 64


def _write_mesh_urdf(tmp_path: Path, mesh: object, name: str) -> Path:
    mesh_path = tmp_path / f"{name}.obj"
    mesh.export(mesh_path)
    urdf = tmp_path / f"{name}.urdf"
    urdf.write_text(
        f'<robot name="mesh"><link name="object"><collision><geometry>'
        f'<mesh filename="{mesh_path.name}"/></geometry></collision></link></robot>',
        encoding="utf-8",
    )
    return urdf


def test_inward_mesh_is_deterministically_normalized_to_outward_normals(tmp_path: Path):
    import trimesh

    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.faces = box.faces[:, ::-1]
    assert box.volume < 0.0 and box.is_winding_consistent

    loaded = load_object_collision_mesh(_write_mesh_urdf(tmp_path, box, "inward"))

    assert loaded.is_watertight and loaded.is_winding_consistent
    assert loaded.volume > 0.0
    assert np.all(np.einsum("ij,ij->i", loaded.triangles_center, loaded.face_normals) > 0.0)


def test_inconsistent_mesh_winding_is_atomic_rejection(tmp_path: Path):
    import trimesh

    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    box.faces[0] = box.faces[0, ::-1]
    assert not box.is_winding_consistent

    with pytest.raises(ValueError, match="winding"):
        load_object_collision_mesh(_write_mesh_urdf(tmp_path, box, "inconsistent"))


def test_moving_object_relative_normal_speed_is_target_spline_analytic_derivative():
    import trimesh

    source_hz = 60
    source_time = np.arange(7, dtype=np.float64) / source_hz
    fingertips = np.zeros((source_time.size, 5, 3), dtype=np.float64)
    fingertips[..., 0] = 0.2
    hand_state = np.zeros((source_time.size, 13), dtype=np.float64)
    object_state = np.zeros_like(hand_state)
    hand_state[:, 6] = object_state[:, 6] = 1.0
    # A stationary fingertip relative to a cubically moving object gives
    # x_object = 0.2 + t^3 and exact normal speed +3*t^2 on the +x box face.
    object_state[:, 0] = -(source_time**3)

    distance, normal_speed, target_time = object_relative_surface_kinematics(
        fingertips,
        hand_state,
        object_state,
        trimesh.creation.box(extents=(0.2, 0.2, 0.2)),
        source_hz=source_hz,
        target_hz=100,
    )

    np.testing.assert_allclose(distance[:, 0], 0.1 + target_time**3, atol=1e-10)
    np.testing.assert_allclose(normal_speed[:, 0], 3.0 * target_time**2, atol=1e-10)
    assert normal_speed[0, 0] == pytest.approx(0.0, abs=1e-12)


def _write_inspire_fixture(path: Path) -> Path:
    spec = SOURCE_HANDS["inspire_rh"]
    links = ['<link name="R_hand_base_link"/>']
    joints = []
    parent = "R_hand_base_link"
    for index, name in enumerate(spec.joint_order):
        child = f"moving_{index}"
        links.append(f'<link name="{child}"/>')
        kind = "prismatic" if index == 0 else "revolute"
        axis = "1 0 0" if index == 0 else "0 0 1"
        joints.append(
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="{axis}"/></joint>'
        )
        parent = child
    for index, tip in enumerate(spec.fingertip_links):
        links.append(f'<link name="{tip}"/>')
        joints.append(
            f'<joint name="tip_{index}" type="fixed"><parent link="{parent}"/>'
            f'<child link="{tip}"/><origin xyz="0 {0.01 * (index - 2)} 0"/></joint>'
        )
    path.write_text(
        '<robot name="fixture">' + "".join(links + joints) + "</robot>", encoding="utf-8"
    )
    return path


def test_loaded_sequence_conversion_uses_fk_mesh_geometry_and_not_tip_force(tmp_path: Path):
    hand_urdf = _write_inspire_fixture(tmp_path / "hand.urdf")
    object_urdf = tmp_path / "object.urdf"
    object_urdf.write_text(
        '<robot name="box"><link name="object"><collision><geometry>'
        '<box size="0.2 0.2 0.2"/></geometry></collision></link></robot>', encoding="utf-8"
    )
    frames = 22
    q = np.zeros((frames, 12), dtype=np.float64)
    q[:, 0] = np.linspace(0.11, 0.09, frames)
    state = np.zeros((frames, 13), dtype=np.float64)
    state[:, 6] = 1.0  # Isaac root-state quaternion order is xyzw.
    sequence = LoadedHandSequence(
        source="favor",
        sequence="sequence_fixture",
        side="rh",
        source_hand_key="inspire_rh",
        rollout_name="rollout_0",
        total_reward=1.0,
        q=q.copy(),
        dq=np.zeros_like(q),
        root_state=state.copy(),
        object_state=state.copy(),
        # Deliberately nonsensical force: Task 5 contact is geometry-only.
        tip_force=np.full((frames, 15), 1.0e9),
        object_geometry_path=object_urdf,
        source_sha256="e" * 64,
    )

    windows = convert_loaded_sequence(sequence, UrdfKinematicTree.from_file(hand_urdf))

    assert windows
    assert all(window.source_group == "favor/sequence_fixture/rh" for window in windows)
    assert not windows[0].contact_mask.any()
    assert any(window.contact_mask.any() for window in windows)
