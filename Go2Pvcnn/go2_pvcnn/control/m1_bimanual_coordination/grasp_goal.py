"""Deterministic, perception-facing bimanual grasp-goal generation.

This module deliberately ends at a task-goal boundary.  It does not own a
camera, object tracker, inverse kinematics solver, or MPC configuration.  The
input geometry is expressed in the object's frame and the returned targets
are expressed in the same world/base frame as ``object_pose``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np


_PROFILES = frozenset(
    {
        "cylindrical",
        "rim_or_body",
        "open_container",
        "thin_two_hand",
        "symmetric_two_hand",
        "polygonal_two_hand",
        "generic",
    }
)
_PROFILE_ALIASES = {
    "stable": "generic",
    "two_hand_stable": "symmetric_two_hand",
}
_PROFILE_NAMES = tuple(sorted(_PROFILES))
_PROFILE_ALIAS_NAMES = tuple(
    f"{alias}={canonical}" for alias, canonical in sorted(_PROFILE_ALIASES.items())
)
_PROFILE_CLASS = {
    "cylindrical": "bottle",
    "rim_or_body": "cup",
    "open_container": "bowl",
    "thin_two_hand": "book",
    "symmetric_two_hand": "cube",
    "polygonal_two_hand": "cylinder",
}
_PROFILE_FORCE = {
    "cylindrical": 5.0,
    "rim_or_body": 4.0,
    "open_container": 4.0,
    "thin_two_hand": 3.0,
    "symmetric_two_hand": 5.0,
    "polygonal_two_hand": 5.0,
    "generic": 4.0,
}


def normalize_grasp_profile(value: object) -> str:
    """Return the canonical goal profile for a catalog or perception label."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "grasp_profile must be a non-empty string; supported profiles are "
            f"{', '.join(_PROFILE_NAMES)}"
        )
    canonical = _PROFILE_ALIASES.get(value, value)
    if canonical not in _PROFILES:
        aliases = ", ".join(_PROFILE_ALIAS_NAMES)
        raise ValueError(
            f"unsupported grasp_profile: {value!r}; supported profiles are "
            f"{', '.join(_PROFILE_NAMES)}; aliases: {aliases}"
        )
    return canonical


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _vector(name: str, value: Sequence[object], length: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of {length} numbers")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of {length} numbers") from error
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return np.asarray([_finite_float(name, item) for item in values], dtype=float)


def _matrix(name: str, value: Sequence[Sequence[object]]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite 3x3 matrix") from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return matrix


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    # This branch-stable conversion is sufficient for the orthonormal frames
    # constructed below and avoids a dependency on a robotics transform lib.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _pose(value: Sequence[object]) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError("object_pose must be a sequence of 3, 6, or 7 numbers") from error
    if len(values) not in (3, 6, 7):
        raise ValueError("object_pose must contain 3, 6, or 7 numbers")
    translation = _vector("object_pose", values[:3], 3)
    if len(values) == 3:
        quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
    elif len(values) == 6:
        quaternion = _quaternion_from_rpy(*_vector("object_pose", values[3:], 3))
    else:
        quaternion = _vector("object_pose", values[3:], 4)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise ValueError("object_pose quaternion must be non-zero")
        quaternion /= norm
    rotation = _rotation_matrix_from_quaternion(quaternion)
    return translation, rotation, tuple(float(value) for value in (*translation, *quaternion))


def _canonicalize_axes(axes: np.ndarray) -> np.ndarray:
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise ValueError("obb_axes must be orthonormal")
    if abs(float(np.linalg.det(axes)) - 1.0) > 1.0e-6:
        raise ValueError("obb_axes must be right-handed")
    # Explicit OBB frames are already part of the perception contract; retain
    # their signs so a caller's frame convention is not silently changed.
    return axes.copy()


def _derive_obb(point_cloud: Sequence[Sequence[object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        points = np.asarray(point_cloud, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("point_cloud must be a finite Nx3 array") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 4:
        raise ValueError("point_cloud must contain at least four Nx3 points")
    if not np.isfinite(points).all():
        raise ValueError("point_cloud must contain only finite values")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1.0e-10) < 3:
        raise ValueError("point_cloud geometry must have non-zero 3-D extent")
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    projected = centered @ axes
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    center = points.mean(axis=0) + axes @ ((lower + upper) / 2.0)
    dimensions = upper - lower
    if np.any(dimensions <= 1.0e-9):
        raise ValueError("point_cloud geometry must have positive extent on every axis")
    return center, dimensions, axes


@dataclass(frozen=True)
class ContactTarget:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent: tuple[float, float, float]


@dataclass(frozen=True)
class OrientedBoundingBox:
    """Object-frame OBB supplied by a perception frontend."""

    dimensions: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ContactSet:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    targets: tuple[ContactTarget, ...]


@dataclass(frozen=True)
class SideGraspTarget:
    palm_position: tuple[float, float, float]
    palm_orientation: tuple[float, float, float, float]
    fingertip_positions: tuple[tuple[float, float, float], ...]
    contact: ContactSet

    @property
    def contact_targets(self) -> tuple[ContactTarget, ...]:
        return self.contact.targets


@dataclass(frozen=True)
class ClampCriteria:
    closure_width_m: float
    min_normal_force_n: float
    max_normal_force_n: float
    min_contact_count_per_hand: int = 2
    max_slip_speed_m_s: float = 0.03
    min_normal_alignment: float = 0.8


@dataclass(frozen=True)
class LiftCriteria:
    height_m: float = 0.10
    min_vertical_force_n: float = 4.0
    max_tilt_rad: float = 0.25
    hold_time_s: float = 3.0


@dataclass(frozen=True)
class BimanualGraspGoal:
    object_pose: tuple[float, ...]
    object_center: tuple[float, float, float]
    object_class: str
    grasp_profile: str
    obb_dimensions: tuple[float, float, float]
    obb_axes: tuple[tuple[float, float, float], ...]
    left: SideGraspTarget
    right: SideGraspTarget
    clamp: ClampCriteria
    lift: LiftCriteria

    @property
    def left_palm_target(self) -> tuple[float, float, float]:
        return self.left.palm_position

    @property
    def right_palm_target(self) -> tuple[float, float, float]:
        return self.right.palm_position

    @property
    def left_fingertip_targets(self) -> tuple[tuple[float, float, float], ...]:
        return self.left.fingertip_positions

    @property
    def right_fingertip_targets(self) -> tuple[tuple[float, float, float], ...]:
        return self.right.fingertip_positions

    @property
    def left_contact_targets(self) -> tuple[ContactTarget, ...]:
        return self.left.contact_targets

    @property
    def right_contact_targets(self) -> tuple[ContactTarget, ...]:
        return self.right.contact_targets

    @property
    def clamp_criteria(self) -> ClampCriteria:
        return self.clamp

    @property
    def lift_criteria(self) -> LiftCriteria:
        return self.lift


def generate_bimanual_grasp_goal(
    object_pose: Sequence[object],
    dimensions: Sequence[object] | None = None,
    *,
    obb_axes: Sequence[Sequence[object]] | None = None,
    obb: OrientedBoundingBox | None = None,
    point_cloud: Sequence[Sequence[object]] | None = None,
    grasp_profile: str = "generic",
    object_class: str | None = None,
    lift_height_m: float = 0.10,
    hold_time_s: float = 3.0,
) -> BimanualGraspGoal:
    """Build symmetric side-clamp targets from an object pose and OBB.

    ``dimensions`` and ``obb_axes`` are object-frame OBB values.  When a
    point cloud is supplied, its deterministic PCA OBB is used instead.  A
    cloud is expected in the object's frame; camera registration belongs to a
    caller.  The profile changes conservative clamp force and clearance only;
    it never changes the left/right symmetry contract.
    """
    grasp_profile = normalize_grasp_profile(grasp_profile)
    if object_class is not None and (not isinstance(object_class, str) or not object_class.strip()):
        raise ValueError("object_class must be a non-empty string when provided")
    if obb is not None and not isinstance(obb, OrientedBoundingBox):
        raise TypeError("obb must be an OrientedBoundingBox")
    if obb is not None and (dimensions is not None or obb_axes is not None or point_cloud is not None):
        raise ValueError("obb cannot be combined with dimensions, obb_axes, or point_cloud")
    translation, object_rotation, normalized_pose = _pose(object_pose)
    if obb is not None:
        local_center = _vector("obb.center", obb.center, 3)
        obb_dimensions = _vector("obb.dimensions", obb.dimensions, 3)
        if np.any(obb_dimensions <= 1.0e-9):
            raise ValueError("obb dimensions must be finite and strictly positive")
        local_axes = _canonicalize_axes(_matrix("obb.axes", obb.axes))
    elif point_cloud is not None:
        local_center, obb_dimensions, local_axes = _derive_obb(point_cloud)
        if dimensions is not None:
            requested = _vector("dimensions", dimensions, 3)
            if not np.allclose(requested, obb_dimensions, atol=1.0e-6, rtol=1.0e-5):
                raise ValueError("dimensions do not match point-cloud OBB")
    else:
        if dimensions is None:
            raise ValueError("one of dimensions or point_cloud is required")
        local_center = np.zeros(3, dtype=float)
        obb_dimensions = _vector("dimensions", dimensions, 3)
        if np.any(obb_dimensions <= 1.0e-9):
            raise ValueError("dimensions must be finite and strictly positive")
        local_axes = np.eye(3) if obb_axes is None else _canonicalize_axes(_matrix("obb_axes", obb_axes))
    if obb_axes is not None and point_cloud is not None:
        # Explicit axes are only meaningful when dimensions are explicit.  A
        # second competing OBB convention would make perception non-deterministic.
        raise ValueError("obb_axes cannot be combined with point_cloud")

    center = translation + object_rotation @ local_center
    center[np.abs(center) < 1.0e-15] = 0.0
    world_axes = object_rotation @ local_axes
    lateral = world_axes[:, 0]
    tangent = world_axes[:, 1]
    vertical = world_axes[:, 2]
    half_extent = float(obb_dimensions[0] / 2.0)
    # Clearance scales for unseen object sizes but remains bounded for large
    # catalog geometry and strictly positive for stable contact separation.
    contact_margin = max(0.002, min(0.02, 0.08 * float(np.min(obb_dimensions))))
    palm_clearance = max(0.004, 0.5 * contact_margin)
    contact_half = float(obb_dimensions[1] / 2.0)
    contact_height = float(obb_dimensions[2] / 2.0)
    offsets = (
        (0.0, 0.0),
        (-0.55 * contact_half, 0.0),
        (0.55 * contact_half, 0.0),
        (0.0, -0.55 * contact_height),
        (0.0, 0.55 * contact_height),
    )

    def make_side(sign: float) -> SideGraspTarget:
        # ``sign=-1`` is the left lane; both normals point into the object.
        outward = sign * lateral
        inward = -outward
        contact_position = center + outward * half_extent
        palm_position = center + outward * (half_extent + palm_clearance)
        fingertip_positions = tuple(
            tuple(
                float(value)
                for value in contact_position + tangent * (-sign) * tangent_offset + vertical * (-sign) * vertical_offset
            )
            for tangent_offset, vertical_offset in offsets
        )
        targets = tuple(
            ContactTarget(position=position, normal=tuple(float(v) for v in inward), tangent=tuple(float(v) for v in tangent))
            for position in fingertip_positions
        )
        frame = np.column_stack((inward, tangent, np.cross(inward, tangent)))
        return SideGraspTarget(
            palm_position=tuple(float(value) for value in palm_position),
            palm_orientation=_quaternion_from_matrix(frame),
            fingertip_positions=fingertip_positions,
            contact=ContactSet(
                position=tuple(float(value) for value in contact_position),
                normal=tuple(float(value) for value in inward),
                targets=targets,
            ),
        )

    left = make_side(-1.0)
    right = make_side(1.0)
    min_force = _PROFILE_FORCE[grasp_profile]
    max_force = max(2.0 * min_force, min_force + 5.0)
    clamp = ClampCriteria(
        closure_width_m=float(2.0 * (half_extent + contact_margin)),
        min_normal_force_n=min_force,
        max_normal_force_n=max_force,
    )
    lift_height = _finite_float("lift_height_m", lift_height_m)
    hold_time = _finite_float("hold_time_s", hold_time_s)
    if lift_height <= 0.0 or hold_time <= 0.0:
        raise ValueError("lift_height_m and hold_time_s must be positive")
    lift = LiftCriteria(
        height_m=lift_height,
        min_vertical_force_n=max(2.0, 0.5 * min_force),
        hold_time_s=hold_time,
    )
    return BimanualGraspGoal(
        object_pose=normalized_pose,
        object_center=tuple(float(value) for value in center),
        object_class=object_class or _PROFILE_CLASS.get(grasp_profile, "generic"),
        grasp_profile=grasp_profile,
        obb_dimensions=tuple(float(value) for value in obb_dimensions),
        obb_axes=tuple(tuple(float(value) for value in local_axes[:, col]) for col in range(3)),
        left=left,
        right=right,
        clamp=clamp,
        lift=lift,
    )


# Short aliases make this adapter convenient at script/task boundaries while
# keeping one implementation and one deterministic output contract.
build_bimanual_grasp_goal = generate_bimanual_grasp_goal
generate_grasp_goal = generate_bimanual_grasp_goal


__all__ = [
    "BimanualGraspGoal",
    "ClampCriteria",
    "ContactSet",
    "ContactTarget",
    "LiftCriteria",
    "OrientedBoundingBox",
    "SideGraspTarget",
    "build_bimanual_grasp_goal",
    "generate_bimanual_grasp_goal",
    "generate_grasp_goal",
    "normalize_grasp_profile",
]
