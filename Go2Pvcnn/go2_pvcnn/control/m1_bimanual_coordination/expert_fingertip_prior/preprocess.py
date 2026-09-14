"""Geometry-only conversion of audited DexManipNet hand sequences.

The module is deliberately offline: it uses the pinned source-hand kinematics and
object collision geometry, and has no Isaac, task-label, or network dependency.
"""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path, PurePosixPath
import stat
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np
from scipy.interpolate import CubicSpline
import torch
import trimesh

from .contracts import ExpertWindow, LEFT_REFLECTION, PRIOR_HORIZON, PriorPhase
from .sources import SOURCE_HANDS

if TYPE_CHECKING:
    from .dexmanipnet import LoadedHandSequence
    from .urdf_fk import UrdfKinematicTree


_REFLECTION = np.asarray(LEFT_REFLECTION, dtype=np.float64)
_CONTAINS_DEFAULT_DIRECTION = np.array(
    [0.4395064455, 0.617598629942, 0.652231566745], dtype=np.float64
)
# Exact normalized direction drawn by RandomState(0).random_sample(3) - 0.5.
# Keeping the literal removes all dependency on NumPy's process-global RNG.
_CONTAINS_FALLBACK_DIRECTION = np.array(
    [0.20053838696390053, 0.8840530785980371, 0.4221782912174071],
    dtype=np.float64,
)


def _finite_geometry(values: np.ndarray, *, name: str, trailing: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.shape[-len(trailing) :] != trailing:
        raise ValueError(f"{name} must have trailing shape {trailing}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def canonicalize_left(values: np.ndarray) -> np.ndarray:
    """Reflect left palm-frame xyz geometry into the frozen right-hand convention."""

    array = _finite_geometry(values, name="left geometry", trailing=(3,))
    return np.einsum("...j,ij->...i", array, _REFLECTION)


def decanonicalize_left(values: np.ndarray) -> np.ndarray:
    """Undo :func:`canonicalize_left`; the frozen reflection is an involution."""

    return canonicalize_left(values)


def _rate(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real sampling rate")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def resample_fingertips_with_velocity(
    points: np.ndarray,
    *,
    source_hz: float = 60,
    target_hz: float = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic-spline resample five fingertips and evaluate its analytic derivative."""

    values = _finite_geometry(points, name="fingertips", trailing=(5, 3)).astype(
        np.float64, copy=False
    )
    if values.ndim != 3:
        raise ValueError("fingertips must have shape (frames, 5, 3)")
    if values.shape[0] < 2:
        raise ValueError("fingertips need at least two source frames")
    source_rate = _rate("source_hz", source_hz)
    target_rate = _rate("target_hz", target_hz)
    source_time = np.arange(values.shape[0], dtype=np.float64) / source_rate
    duration = source_time[-1]
    last_index = int(np.floor(np.nextafter(duration * target_rate, np.inf)))
    target_time = np.arange(last_index + 1, dtype=np.float64) / target_rate
    target_time = target_time[target_time <= np.nextafter(duration, np.inf)]
    spline = CubicSpline(source_time, values, axis=0)
    positions = np.asarray(spline(target_time), dtype=np.float64)
    velocities = np.asarray(spline(target_time, 1), dtype=np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("resampled fingertip geometry must be finite")
    return positions, velocities, target_time


def resample_fingertips(
    points: np.ndarray,
    *,
    source_hz: float = 60,
    target_hz: float = 100,
) -> np.ndarray:
    """Return the position component of deterministic cubic resampling."""

    return resample_fingertips_with_velocity(
        points, source_hz=source_hz, target_hz=target_hz
    )[0]


def infer_contact_hysteresis(
    signed_distance: np.ndarray,
    relative_normal_speed: np.ndarray,
    *,
    enter_m: float = 0.002,
    exit_m: float = 0.005,
) -> np.ndarray:
    """Infer per-tip contact from geometry with deterministic enter/exit hysteresis.

    Distance is positive outside the object and negative in penetration.  A new
    contact needs proximity and a non-separating relative normal velocity.  Once
    active, it remains active until the wider exit distance is reached.
    """

    distance = _finite_geometry(
        signed_distance, name="signed distance", trailing=(5,)
    ).astype(np.float64, copy=False)
    speed = _finite_geometry(
        relative_normal_speed, name="relative normal speed", trailing=(5,)
    ).astype(np.float64, copy=False)
    if distance.ndim != 2 or speed.shape != distance.shape:
        raise ValueError("distance and normal speed must share shape (frames, 5)")
    for name, value in (("enter_m", enter_m), ("exit_m", exit_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if enter_m < 0.0 or exit_m <= enter_m:
        raise ValueError("contact thresholds must satisfy 0 <= enter_m < exit_m")

    result = np.zeros(distance.shape, dtype=np.bool_)
    active = np.zeros(5, dtype=np.bool_)
    for frame in range(distance.shape[0]):
        active &= distance[frame] < exit_m
        entering = (distance[frame] <= 0.0) | (
            (distance[frame] <= enter_m) & (speed[frame] <= 0.0)
        )
        active |= (~active) & entering
        result[frame] = active
    return result


def infer_prior_phase(
    contact: np.ndarray,
    fingertip_speed: np.ndarray,
    *,
    hold_speed_m_s: float = 0.002,
    manipulate_speed_m_s: float = 0.02,
) -> np.ndarray:
    """Infer seven task-independent phases from contact transitions and speed."""

    mask = np.asarray(contact)
    speed = np.asarray(fingertip_speed)
    if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[1] != 5:
        raise ValueError("contact must be a bool array with shape (frames, 5)")
    if speed.dtype.kind not in "iuf" or speed.shape != (mask.shape[0],):
        raise ValueError("fingertip_speed must have shape (frames,)")
    if not np.isfinite(speed).all() or np.any(speed < 0.0):
        raise ValueError("fingertip_speed must be finite and non-negative")
    if not (0.0 <= hold_speed_m_s < manipulate_speed_m_s):
        raise ValueError("phase speed thresholds are invalid")

    phases = np.full(mask.shape[0], int(PriorPhase.UNKNOWN), dtype=np.uint8)
    counts = np.count_nonzero(mask, axis=1)
    for frame, count in enumerate(counts):
        previous = counts[frame - 1] if frame else 0
        if count == 0:
            if previous > 0:
                phases[frame] = PriorPhase.RELEASE
            elif speed[frame] >= hold_speed_m_s:
                phases[frame] = PriorPhase.APPROACH
        elif previous == 0:
            phases[frame] = PriorPhase.PRELOAD
        elif count > previous:
            phases[frame] = PriorPhase.GRASP
        elif speed[frame] <= hold_speed_m_s:
            phases[frame] = PriorPhase.HOLD
        elif speed[frame] >= manipulate_speed_m_s:
            phases[frame] = PriorPhase.MANIPULATE
        else:
            phases[frame] = PriorPhase.GRASP
    return phases


def windows_from_sequence(
    fingertip_position_palm: np.ndarray,
    fingertip_velocity_palm: np.ndarray,
    contact_mask: np.ndarray,
    phase: np.ndarray,
    *,
    source_group: str,
    source_sha256: str,
) -> tuple[ExpertWindow, ...]:
    """Build non-crossing examples with 20 strictly future 100 Hz velocity nodes."""

    position = _finite_geometry(
        fingertip_position_palm, name="fingertip positions", trailing=(5, 3)
    )
    velocity = _finite_geometry(
        fingertip_velocity_palm, name="fingertip velocities", trailing=(5, 3)
    )
    contact = np.asarray(contact_mask)
    phases = np.asarray(phase)
    frames = position.shape[0]
    if position.ndim != 3 or velocity.shape != position.shape:
        raise ValueError("position and velocity must share shape (frames, 5, 3)")
    if contact.dtype != np.bool_ or contact.shape != (frames, 5):
        raise ValueError("contact_mask must be bool with shape (frames, 5)")
    if phases.shape != (frames,) or phases.dtype.kind not in "iu":
        raise ValueError("phase must be an integer array with shape (frames,)")
    if np.any(phases < 0) or np.any(phases >= len(PriorPhase)):
        raise ValueError("phase contains an unknown label")
    if type(source_group) is not str or not source_group:
        raise ValueError("source_group must be non-empty")
    if (
        type(source_sha256) is not str
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256")

    windows: list[ExpertWindow] = []
    for frame in range(max(0, frames - PRIOR_HORIZON)):
        windows.append(
            ExpertWindow(
                fingertip_position_palm=torch.from_numpy(
                    np.array(position[frame], dtype=np.float32, copy=True)
                ),
                fingertip_velocity_palm=torch.from_numpy(
                    np.array(velocity[frame], dtype=np.float32, copy=True)
                ),
                contact_mask=torch.from_numpy(np.array(contact[frame], dtype=np.bool_, copy=True)),
                phase=PriorPhase(int(phases[frame])),
                future_fingertip_velocity_palm=torch.from_numpy(
                    np.array(
                        velocity[frame + 1 : frame + PRIOR_HORIZON + 1],
                        dtype=np.float32,
                        copy=True,
                    )
                ),
                source_group=source_group,
                source_sha256=source_sha256,
            )
        )
    return tuple(windows)


def _vector(raw: str | None, *, default: tuple[float, float, float], label: str) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float64)
    fields = raw.split()
    if len(fields) != 3:
        raise ValueError(f"object geometry {label} must have three values")
    try:
        values = np.asarray([float(field) for field in fields], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"object geometry {label} must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"object geometry {label} must be finite")
    return values


def _rpy_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    result[:3, 3] = xyz
    return result


def _safe_mesh_path(urdf: Path, raw: str) -> Path:
    posix = PurePosixPath(raw)
    if not raw or posix.is_absolute() or ".." in posix.parts:
        raise ValueError("object geometry mesh path is unsafe")
    if raw.startswith(("package://", "file://")):
        raise ValueError("object geometry mesh path is unsafe")
    candidate = urdf.parent
    for part in posix.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as error:
            raise ValueError("object geometry mesh is missing") from error
        if stat.S_ISLNK(mode):
            raise ValueError("object geometry mesh path is unsafe")
    if not candidate.is_file():
        raise ValueError("object geometry mesh is missing")
    return candidate


def _collision_mesh(geometry: ElementTree.Element, urdf: Path) -> trimesh.Trimesh:
    children = list(geometry)
    if len(children) != 1:
        raise ValueError("object geometry collision must contain exactly one shape")
    shape = children[0]
    if shape.tag == "box":
        size = _vector(shape.get("size"), default=(0.0, 0.0, 0.0), label="box size")
        if np.any(size <= 0.0):
            raise ValueError("object geometry box size must be positive")
        return trimesh.creation.box(extents=size)
    if shape.tag == "sphere":
        try:
            radius = float(shape.get("radius", "nan"))
        except ValueError as error:
            raise ValueError("object geometry sphere radius must be numeric") from error
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("object geometry sphere radius must be positive")
        return trimesh.creation.icosphere(subdivisions=3, radius=radius)
    if shape.tag == "cylinder":
        try:
            radius = float(shape.get("radius", "nan"))
            length = float(shape.get("length", "nan"))
        except ValueError as error:
            raise ValueError("object geometry cylinder dimensions must be numeric") from error
        if not all(math.isfinite(value) and value > 0.0 for value in (radius, length)):
            raise ValueError("object geometry cylinder dimensions must be positive")
        return trimesh.creation.cylinder(radius=radius, height=length, sections=64)
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if filename is None:
            raise ValueError("object geometry mesh filename is missing")
        path = _safe_mesh_path(urdf, filename)
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except BaseException as error:
            raise ValueError("object geometry mesh is unusable") from error
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("object geometry mesh is unusable")
        scale = _vector(shape.get("scale"), default=(1.0, 1.0, 1.0), label="mesh scale")
        if np.any(scale <= 0.0):
            raise ValueError("object geometry mesh scale must be positive")
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        return mesh
    raise ValueError(f"object geometry shape is unsupported: {shape.tag}")


def _outward_collision_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return consistently wound, positive-volume collision geometry."""

    result = mesh.copy()
    if (
        result.vertices.size == 0
        or result.faces.size == 0
        or not np.isfinite(result.vertices).all()
        or not result.is_watertight
    ):
        raise ValueError("object geometry collision mesh is unusable")
    if not result.is_winding_consistent:
        raise ValueError("object geometry collision mesh has inconsistent winding")
    volume = float(result.volume)
    if not math.isfinite(volume) or volume == 0.0:
        raise ValueError("object geometry collision mesh has unusable volume")
    if volume < 0.0:
        result.invert()
    if not result.is_winding_consistent or not math.isfinite(float(result.volume)) or result.volume <= 0.0:
        raise ValueError("object geometry collision mesh orientation is unusable")
    return result


def _parse_object_urdf(urdf: Path) -> ElementTree.Element:
    """Parse a local object URDF after removing only leading ASCII whitespace."""

    try:
        return ElementTree.fromstring(urdf.read_bytes().lstrip(b" \t\r\n"))
    except (ElementTree.ParseError, OSError) as error:
        raise ValueError("object geometry URDF is unusable") from error


def load_object_collision_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a finite, watertight object collision surface from a local URDF."""

    urdf = Path(path)
    if not urdf.is_file() or urdf.is_symlink():
        raise ValueError("object geometry URDF is missing or unsafe")
    root = _parse_object_urdf(urdf)
    if root.tag != "robot":
        raise ValueError("object geometry URDF root must be robot")
    meshes: list[trimesh.Trimesh] = []
    for collision in root.findall(".//collision"):
        geometries = collision.findall("geometry")
        if len(geometries) != 1:
            raise ValueError("object geometry collision must contain one geometry")
        mesh = _collision_mesh(geometries[0], urdf)
        origins = collision.findall("origin")
        if len(origins) > 1:
            raise ValueError("object geometry collision has multiple origins")
        origin = origins[0] if origins else None
        transform = _rpy_transform(
            _vector(None if origin is None else origin.get("xyz"), default=(0, 0, 0), label="xyz"),
            _vector(None if origin is None else origin.get("rpy"), default=(0, 0, 0), label="rpy"),
        )
        mesh.apply_transform(transform)
        meshes.append(_outward_collision_mesh(mesh))
    if not meshes:
        raise ValueError("object geometry URDF has no collision geometry")
    combined = trimesh.util.concatenate(meshes)
    return _outward_collision_mesh(combined)


def _quaternion_xyzw_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps) or not np.isfinite(norm).all():
        raise ValueError("root-state quaternion is unusable")
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _object_relative_points(
    fingertip_palm: np.ndarray,
    root_state: np.ndarray,
    object_state: np.ndarray,
) -> np.ndarray:
    hand_rotation = _quaternion_xyzw_matrix(root_state[:, 3:7])
    world = np.einsum("tij,tfj->tfi", hand_rotation, fingertip_palm) + root_state[:, None, :3]
    object_rotation = _quaternion_xyzw_matrix(object_state[:, 3:7])
    return np.einsum(
        "tji,tfj->tfi", object_rotation, world - object_state[:, None, :3]
    )


def _contains_points_with_fixed_fallback(intersector: object, points: np.ndarray) -> np.ndarray:
    """Match Trimesh's contains query with an explicit broken-ray fallback."""

    query = np.asanyarray(points, dtype=np.float64)
    if not trimesh.util.is_shape(query, (-1, 3)):
        raise ValueError("points must be (n,3)")
    contains = np.zeros(len(query), dtype=bool)
    inside_aabb = trimesh.bounds.contains(intersector.mesh.bounds, query)
    if not inside_aabb.any():
        return contains

    directions = np.tile(_CONTAINS_DEFAULT_DIRECTION, (inside_aabb.sum(), 1))
    _, ray_index, _ = intersector.intersects_location(
        np.vstack((query[inside_aabb], query[inside_aabb])),
        np.vstack((directions, -directions)),
    )
    if len(ray_index) == 0:
        return contains
    bidirectional_hits = np.bincount(
        ray_index, minlength=len(directions) * 2
    ).reshape((2, -1))
    bidirectional_contains = np.mod(bidirectional_hits, 2) == 1
    agree = np.equal(*bidirectional_contains)
    mask = inside_aabb.copy()
    mask[mask] = agree
    contains[mask] = bidirectional_contains[0][agree]

    one_freespace = (bidirectional_hits == 0).any(axis=0)
    broken = np.logical_and(np.logical_not(agree), np.logical_not(one_freespace))
    if broken.any():
        mask = inside_aabb.copy()
        mask[mask] = broken
        contains[mask] = trimesh.ray.ray_util.contains_points(
            intersector,
            query[inside_aabb][broken],
            check_direction=_CONTAINS_FALLBACK_DIRECTION,
        )
    return contains


def _deterministic_signed_distance(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> np.ndarray:
    """Match Trimesh signed distance without its process-global random fallback."""

    query = np.asanyarray(points, dtype=np.float64)
    closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, query)
    nonzero_mask = distance > trimesh.constants.tol.merge
    if not nonzero_mask.any():
        return distance

    nonzero = np.where(nonzero_mask)[0]
    normals = mesh.face_normals[triangle_id]
    projection = (
        query[nonzero]
        - (
            normals[nonzero].T
            * np.einsum(
                "ij,ij->i", query[nonzero] - closest[nonzero], normals[nonzero]
            )
        ).T
    )
    barycentric = trimesh.triangles.points_to_barycentric(
        mesh.triangles[triangle_id[nonzero]], projection
    )
    on_triangle = ~(
        (
            (barycentric < -trimesh.constants.tol.merge)
            | (barycentric > 1 + trimesh.constants.tol.merge)
        ).any(axis=1)
    )
    on_triangle_nonzero = nonzero[on_triangle]
    sign = np.sign(
        np.einsum(
            "ij,ij->i",
            normals[on_triangle_nonzero],
            query[on_triangle_nonzero] - projection[on_triangle],
        )
    )
    distance[on_triangle_nonzero] *= -1.0 * sign

    off_triangle_nonzero = nonzero[~on_triangle]
    inside = _contains_points_with_fixed_fallback(mesh.ray, query[off_triangle_nonzero])
    distance[off_triangle_nonzero] *= inside.astype(int) * 2 - 1.0
    return distance


def object_relative_surface_kinematics(
    fingertip_palm: np.ndarray,
    root_state: np.ndarray,
    object_state: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    source_hz: float = 60,
    target_hz: float = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query target-rate surface distance and analytic object-relative normal speed."""

    points = _finite_geometry(
        fingertip_palm, name="fingertip positions", trailing=(5, 3)
    ).astype(np.float64, copy=False)
    if points.ndim != 3 or points.shape[0] < 2:
        raise ValueError("fingertip positions must have shape (frames, 5, 3)")
    for name, states in (("root_state", root_state), ("object_state", object_state)):
        values = _finite_geometry(states, name=name, trailing=(13,))
        if values.shape != (points.shape[0], 13):
            raise ValueError(f"{name} must have shape (frames, 13)")
    source_rate = _rate("source_hz", source_hz)
    target_rate = _rate("target_hz", target_hz)
    oriented_mesh = _outward_collision_mesh(mesh)
    source_local = _object_relative_points(points, root_state, object_state)
    source_time = np.arange(points.shape[0], dtype=np.float64) / source_rate
    duration = source_time[-1]
    last_index = int(np.floor(np.nextafter(duration * target_rate, np.inf)))
    target_time = np.arange(last_index + 1, dtype=np.float64) / target_rate
    target_time = target_time[target_time <= np.nextafter(duration, np.inf)]
    spline = CubicSpline(source_time, source_local, axis=0)
    target_local = np.asarray(spline(target_time), dtype=np.float64)
    target_local_velocity = np.asarray(spline(target_time, 1), dtype=np.float64)
    flat = target_local.reshape(-1, 3)
    try:
        signed = -_deterministic_signed_distance(oriented_mesh, flat).reshape(
            target_local.shape[:2]
        )
        _, _, triangle = trimesh.proximity.closest_point_naive(oriented_mesh, flat)
    except BaseException as error:
        raise ValueError("object geometry distance query failed") from error
    normals = np.asarray(oriented_mesh.face_normals[triangle]).reshape(target_local.shape)
    normal_speed = np.sum(target_local_velocity * normals, axis=-1)
    if not np.isfinite(signed).all() or not np.isfinite(normal_speed).all():
        raise ValueError("object geometry distance query is non-finite")
    return signed, normal_speed, target_time


def convert_loaded_sequence(
    sequence: LoadedHandSequence,
    tree: UrdfKinematicTree,
    *,
    source_hz: float = 60,
    target_hz: float = 100,
    enter_m: float = 0.002,
    exit_m: float = 0.005,
) -> tuple[ExpertWindow, ...]:
    """Atomically convert one audited rollout into canonical 100 Hz windows."""

    source_hand = SOURCE_HANDS[sequence.source_hand_key]
    try:
        source_points = tree.palm_relative_fingertips(sequence.q, source_hand)
        mesh = load_object_collision_mesh(sequence.object_geometry_path)
        positions, velocities, target_time = resample_fingertips_with_velocity(
            source_points, source_hz=source_hz, target_hz=target_hz
        )
        distance, normal_speed, contact_time = object_relative_surface_kinematics(
            source_points,
            sequence.root_state,
            sequence.object_state,
            mesh,
            source_hz=source_hz,
            target_hz=target_hz,
        )
        if not np.array_equal(contact_time, target_time):
            raise RuntimeError("position and contact target timestamps diverged")
        contact = infer_contact_hysteresis(
            distance, normal_speed, enter_m=enter_m, exit_m=exit_m
        )
        if sequence.side == "lh":
            positions = canonicalize_left(positions)
            velocities = canonicalize_left(velocities)
        scalar_speed = np.sqrt(np.mean(np.square(velocities), axis=(1, 2)))
        phase = infer_prior_phase(contact, scalar_speed)
        return windows_from_sequence(
            positions,
            velocities,
            contact,
            phase,
            source_group=f"{sequence.source}/{sequence.sequence}/{sequence.side}",
            source_sha256=sequence.source_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sequence conversion rejected: {error}") from error


def object_geometry_sha256(path: str | Path) -> str:
    """Hash the URDF and every safe mesh file it references for audit provenance."""

    urdf = Path(path)
    load_object_collision_mesh(urdf)
    root = _parse_object_urdf(urdf)
    files = [urdf]
    for shape in root.findall(".//collision/geometry/mesh"):
        filename = shape.get("filename")
        if filename is None:
            raise ValueError("object geometry mesh filename is missing")
        files.append(_safe_mesh_path(urdf, filename))
    digest = sha256()
    for file in sorted(files, key=lambda item: item.relative_to(urdf.parent).as_posix()):
        relative = file.relative_to(urdf.parent).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


__all__ = [
    "canonicalize_left",
    "convert_loaded_sequence",
    "decanonicalize_left",
    "infer_contact_hysteresis",
    "infer_prior_phase",
    "load_object_collision_mesh",
    "object_geometry_sha256",
    "object_relative_surface_kinematics",
    "resample_fingertips",
    "resample_fingertips_with_velocity",
    "windows_from_sequence",
]
