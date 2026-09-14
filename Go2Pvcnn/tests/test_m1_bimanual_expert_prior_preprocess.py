from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior import (
    preprocess as preprocess_module,
)
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


def _stationary_surface_query_fixture():
    import trimesh

    fingertips = np.zeros((2, 5, 3), dtype=np.float64)
    fingertips[..., 0] = 0.2
    state = np.zeros((2, 13), dtype=np.float64)
    state[:, 6] = 1.0
    return fingertips, state, trimesh.creation.box(extents=(0.2, 0.2, 0.2))


def _assert_random_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_surface_distance_never_accesses_numpy_global_rng(monkeypatch):
    fingertips, state, mesh = _stationary_surface_query_fixture()

    def forbidden_global_rng(*_args, **_kwargs):
        raise AssertionError("surface distance must not access NumPy's global RNG")

    for name in ("get_state", "set_state", "seed", "random"):
        monkeypatch.setattr(np.random, name, forbidden_global_rng)

    distance = object_relative_surface_kinematics(fingertips, state, state, mesh)[0]

    assert np.isfinite(distance).all()


def test_surface_distance_failure_does_not_access_numpy_global_rng(monkeypatch):
    fingertips, state, mesh = _stationary_surface_query_fixture()

    def failing_closest_point(_mesh, _points):
        raise RuntimeError("simulated deterministic trimesh failure")

    def forbidden_global_rng(*_args, **_kwargs):
        raise AssertionError("failure handling must not access NumPy's global RNG")

    monkeypatch.setattr("trimesh.proximity.closest_point", failing_closest_point)
    for name in ("get_state", "set_state", "seed", "random"):
        monkeypatch.setattr(np.random, name, forbidden_global_rng)

    with pytest.raises(ValueError, match="object geometry distance query failed"):
        object_relative_surface_kinematics(fingertips, state, state, mesh)


def test_surface_distance_does_not_corrupt_concurrent_external_numpy_rng(monkeypatch):
    fingertips, state, mesh = _stationary_surface_query_fixture()
    original_closest_point = preprocess_module.trimesh.proximity.closest_point
    entered = Event()
    release = Event()

    def coordinated_closest_point(query_mesh, points):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_closest_point(query_mesh, points)

    monkeypatch.setattr("trimesh.proximity.closest_point", coordinated_closest_point)
    np.random.seed(161803)
    expected = np.random.RandomState(161803).random_sample(8)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            object_relative_surface_kinematics, fingertips, state, state, mesh
        )
        assert entered.wait(timeout=2.0)
        external_during_query = np.random.random(4)
        release.set()
        assert np.isfinite(future.result()[0]).all()
    external_after_query = np.random.random(4)

    np.testing.assert_array_equal(
        np.concatenate((external_during_query, external_after_query)), expected
    )


def test_real_trimesh_surface_query_preserves_caller_numpy_rng():
    fingertips, state, mesh = _stationary_surface_query_fixture()
    np.random.seed(314159)
    before = np.random.get_state()

    first = object_relative_surface_kinematics(fingertips, state, state, mesh)[0]
    _assert_random_state_equal(np.random.get_state(), before)
    np.random.seed(271828)
    second_before = np.random.get_state()
    second = object_relative_surface_kinematics(fingertips, state, state, mesh)[0]

    np.testing.assert_array_equal(second, first)
    _assert_random_state_equal(np.random.get_state(), second_before)


def test_nonfallback_signed_distance_is_exactly_trimesh_compatible(monkeypatch):
    import trimesh

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    points = np.array(
        [[0.15, 0.01, 0.02], [-0.04, 0.03, 0.02], [0.02, -0.16, 0.01]],
        dtype=np.float64,
    )

    expected = trimesh.proximity.signed_distance(mesh, points)

    def empty_rays_only(_intersector, ray_points):
        assert ray_points.shape == (0, 3)
        return np.zeros(0, dtype=bool)

    monkeypatch.setattr(
        preprocess_module,
        "_contains_points_with_fixed_fallback",
        empty_rays_only,
    )
    actual = preprocess_module._deterministic_signed_distance(mesh, points)

    np.testing.assert_array_equal(actual, expected)


def test_signed_distance_mixed_surface_triangle_and_raycast_indices_match_trimesh(
    monkeypatch,
):
    import trimesh

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    # Index zero has zero distance, indices 1, 2, 6, 7 project onto a closest
    # triangle, and indices 3, 4, 5 require the raycast branch.  Keeping the
    # zero-distance row first proves that normal indices stay in the original
    # query domain rather than the compressed nonzero domain.
    points = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.15, 0.01, 0.02],
            [-0.04, 0.03, 0.02],
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.0],
            [0.2, 0.05, 0.2],
            [0.0, 0.0, 0.0],
            [0.02, -0.16, 0.01],
        ],
        dtype=np.float64,
    )
    expected = trimesh.proximity.signed_distance(mesh, points)
    original_contains = preprocess_module._contains_points_with_fixed_fallback
    raycast_queries = []

    def recording_contains(intersector, query):
        raycast_queries.append(np.asarray(query).copy())
        return original_contains(intersector, query)

    monkeypatch.setattr(
        preprocess_module, "_contains_points_with_fixed_fallback", recording_contains
    )
    actual = preprocess_module._deterministic_signed_distance(mesh, points)

    np.testing.assert_array_equal(actual, expected)
    assert len(raycast_queries) == 1
    np.testing.assert_array_equal(raycast_queries[0], points[[3, 4, 5]])
    np.testing.assert_array_equal(points[0], [0.1, 0.0, 0.0])


def test_fixed_fallback_direction_exactly_matches_legacy_random_state_zero():
    import trimesh

    expected = trimesh.util.unitize(
        np.random.RandomState(0).random_sample(3) - 0.5
    )

    np.testing.assert_array_equal(
        preprocess_module._CONTAINS_FALLBACK_DIRECTION, expected
    )


def test_broken_ray_fallback_result_matches_legacy_seed_zero():
    class DirectionSensitiveIntersector:
        mesh = SimpleNamespace(
            bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        )

        def __init__(self):
            self.fallback_directions = []

        def intersects_location(self, ray_origins, ray_directions):
            ray_count = len(ray_origins) // 2
            if ray_count == 2:
                np.testing.assert_array_equal(
                    ray_directions[:2],
                    np.tile(preprocess_module._CONTAINS_DEFAULT_DIRECTION, (2, 1)),
                )
                # Point zero is broken (one forward, two backward hits), while
                # point one agrees as inside (one hit in each direction).
                ray_index = np.array([0, 2, 2, 1, 3], dtype=np.int64)
            else:
                assert ray_count == 1
                self.fallback_directions.append(ray_directions[0].copy())
                ray_index = np.array([0, 1], dtype=np.int64)
            return (
                np.zeros((len(ray_index), 3), dtype=np.float64),
                ray_index,
                np.zeros(len(ray_index), dtype=np.int64),
            )

    points = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    legacy = DirectionSensitiveIntersector()
    caller_state = np.random.get_state()
    try:
        np.random.seed(0)
        expected = preprocess_module.trimesh.ray.ray_util.contains_points(
            legacy, points
        )
    finally:
        np.random.set_state(caller_state)
    deterministic = DirectionSensitiveIntersector()

    actual = preprocess_module._contains_points_with_fixed_fallback(
        deterministic, points
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [True, True, False])
    assert len(legacy.fallback_directions) == 1
    assert len(deterministic.fallback_directions) == 1
    np.testing.assert_array_equal(
        deterministic.fallback_directions[0], legacy.fallback_directions[0]
    )


def test_broken_ray_fallback_uses_an_explicit_fixed_non_degenerate_direction(monkeypatch):
    class BrokenRayIntersector:
        mesh = SimpleNamespace(
            bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        )

        def intersects_location(self, ray_origins, ray_directions):
            assert ray_origins.shape == (2, 3)
            assert ray_directions.shape == (2, 3)
            # One forward hit and two backward hits make the parity disagree,
            # with neither direction classified as free space.
            return np.zeros((3, 3)), np.array([0, 1, 1]), np.zeros(3, dtype=np.int64)

    observed = []

    def explicit_contains(_intersector, points, check_direction=None):
        observed.append(np.asarray(check_direction, dtype=np.float64))
        assert points.shape == (1, 3)
        return np.ones(1, dtype=bool)

    monkeypatch.setattr("trimesh.ray.ray_util.contains_points", explicit_contains)

    inside = preprocess_module._contains_points_with_fixed_fallback(
        BrokenRayIntersector(), np.zeros((1, 3), dtype=np.float64)
    )

    assert inside.tolist() == [True]
    assert len(observed) == 1
    assert observed[0].shape == (3,)
    assert np.isfinite(observed[0]).all()
    assert np.linalg.norm(observed[0]) == pytest.approx(1.0, abs=1e-15)


def _forked_surface_distance(queue):
    try:
        import trimesh

        mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        result = preprocess_module._deterministic_signed_distance(
            mesh, np.array([[0.15, 0.01, 0.02]], dtype=np.float64)
        )
        queue.put(bool(np.isfinite(result).all()))
    except BaseException as error:
        queue.put(f"{type(error).__name__}: {error}")


def test_surface_query_has_no_module_lock_and_forked_child_completes():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method is unavailable")
    assert not hasattr(preprocess_module, "_SIGNED_DISTANCE_RNG_LOCK")
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_forked_surface_distance, args=(queue,))
    try:
        process.start()
        process.join(timeout=2.0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)

    assert process.exitcode == 0
    assert queue.get(timeout=1.0) is True


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
