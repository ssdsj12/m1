"""Deterministic NumPy forward kinematics for pinned source-hand URDFs.

This offline module deliberately depends only on the Python standard library and
NumPy.  It is not an Isaac runtime URDF loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np

if TYPE_CHECKING:
    from .sources import SourceHandSpec


_MOVING_JOINT_TYPES = frozenset({"continuous", "prismatic", "revolute"})
_SUPPORTED_JOINT_TYPES = _MOVING_JOINT_TYPES | {"fixed"}


@dataclass(frozen=True)
class JointLimit:
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass(frozen=True)
class Mimic:
    joint: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit: JointLimit
    mimic: Mimic | None


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Return rotation matrices for one axis and an arbitrary batch of angles."""

    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("axis must be a finite three-vector")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    if not np.isfinite(angle).all():
        raise ValueError("angle must be finite")
    axis = axis / norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return (
        eye
        + np.sin(angle)[..., None, None] * skew
        + (1.0 - np.cos(angle))[..., None, None] * (skew @ skew)
    )


def _required_name(element: ElementTree.Element, kind: str) -> str:
    name = element.get("name")
    if name is None or not name.strip():
        raise ValueError(f"{kind} name must be non-empty")
    return name


def _single_child(element: ElementTree.Element, tag: str, *, required: bool) -> ElementTree.Element | None:
    matches = element.findall(tag)
    if len(matches) > 1:
        raise ValueError(f"{element.tag} must contain at most one {tag}")
    if required and not matches:
        raise ValueError(f"{element.tag} is missing {tag}")
    return matches[0] if matches else None


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _vector(attribute: str | None, default: tuple[float, float, float], label: str) -> np.ndarray:
    if attribute is None:
        return np.asarray(default, dtype=np.float64)
    fields = attribute.split()
    if len(fields) != 3:
        raise ValueError(f"{label} must contain three values")
    values = np.asarray([_finite_float(field, label) for field in fields], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(element: ElementTree.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is None:
        return result
    result[:3, :3] = _rpy_matrix(_vector(element.get("rpy"), (0.0, 0.0, 0.0), "origin rpy"))
    result[:3, 3] = _vector(element.get("xyz"), (0.0, 0.0, 0.0), "origin xyz")
    return result


def _optional_float(element: ElementTree.Element | None, attribute: str, label: str) -> float | None:
    if element is None or element.get(attribute) is None:
        return None
    return _finite_float(element.get(attribute), label)  # type: ignore[arg-type]


def _parse_joint(element: ElementTree.Element) -> UrdfJoint:
    name = _required_name(element, "joint")
    joint_type = element.get("type")
    if joint_type not in _SUPPORTED_JOINT_TYPES:
        raise ValueError(f"unknown joint type for {name}: {joint_type!r}")

    parent_element = _single_child(element, "parent", required=True)
    child_element = _single_child(element, "child", required=True)
    assert parent_element is not None and child_element is not None
    parent = parent_element.get("link")
    child = child_element.get("link")
    if parent is None or not parent.strip():
        raise ValueError(f"joint {name} parent link must be non-empty")
    if child is None or not child.strip():
        raise ValueError(f"joint {name} child link must be non-empty")

    origin = _origin_matrix(_single_child(element, "origin", required=False))
    axis_element = _single_child(element, "axis", required=False)
    axis = _vector(
        None if axis_element is None else axis_element.get("xyz"),
        (1.0, 0.0, 0.0),
        f"joint {name} axis",
    )
    if joint_type in _MOVING_JOINT_TYPES:
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"joint {name} axis must be finite and non-zero")
        axis = axis / norm

    limit_element = _single_child(element, "limit", required=False)
    limit = JointLimit(
        lower=_optional_float(limit_element, "lower", f"joint {name} lower limit"),
        upper=_optional_float(limit_element, "upper", f"joint {name} upper limit"),
        effort=_optional_float(limit_element, "effort", f"joint {name} effort limit"),
        velocity=_optional_float(limit_element, "velocity", f"joint {name} velocity limit"),
    )
    if limit.lower is not None and limit.upper is not None and limit.lower > limit.upper:
        raise ValueError(f"joint {name} lower limit exceeds upper limit")

    mimic_element = _single_child(element, "mimic", required=False)
    mimic = None
    if mimic_element is not None:
        master = mimic_element.get("joint")
        if master is None or not master.strip():
            raise ValueError(f"joint {name} mimic master must be non-empty")
        mimic = Mimic(
            joint=master,
            multiplier=_optional_float(mimic_element, "multiplier", f"joint {name} mimic multiplier")
            if mimic_element.get("multiplier") is not None
            else 1.0,
            offset=_optional_float(mimic_element, "offset", f"joint {name} mimic offset")
            if mimic_element.get("offset") is not None
            else 0.0,
        )

    return UrdfJoint(
        name=name,
        joint_type=joint_type,
        parent=parent,
        child=child,
        origin=origin,
        axis=axis,
        limit=limit,
        mimic=mimic,
    )


class UrdfKinematicTree:
    """A validated, single-rooted URDF link tree with batched FK."""

    def __init__(
        self,
        *,
        links: tuple[str, ...],
        joints: Mapping[str, UrdfJoint],
        root_link: str,
        topological_joint_names: tuple[str, ...],
    ) -> None:
        self.links = links
        self.joints = MappingProxyType(dict(joints))
        self.root_link = root_link
        self.topological_joint_names = topological_joint_names

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> "UrdfKinematicTree":
        try:
            document = ElementTree.parse(Path(path))
        except (ElementTree.ParseError, OSError) as error:
            raise ValueError(f"failed to parse URDF: {path}") from error
        root = document.getroot()
        if root.tag != "robot":
            raise ValueError("URDF root element must be robot")

        link_names: list[str] = []
        link_set: set[str] = set()
        for element in root.findall("link"):
            name = _required_name(element, "link")
            if name in link_set:
                raise ValueError(f"duplicate link name: {name}")
            link_names.append(name)
            link_set.add(name)
        if not link_names:
            raise ValueError("URDF must contain at least one link")

        joints: dict[str, UrdfJoint] = {}
        parent_by_child: dict[str, str] = {}
        children_by_parent: dict[str, list[UrdfJoint]] = {name: [] for name in link_names}
        for element in root.findall("joint"):
            joint = _parse_joint(element)
            if joint.name in joints:
                raise ValueError(f"duplicate joint name: {joint.name}")
            if joint.parent not in link_set:
                raise ValueError(f"joint {joint.name} has unknown parent link: {joint.parent}")
            if joint.child not in link_set:
                raise ValueError(f"joint {joint.name} has unknown child link: {joint.child}")
            if joint.child in parent_by_child:
                raise ValueError(f"link {joint.child} has multiple parents")
            joints[joint.name] = joint
            parent_by_child[joint.child] = joint.parent
            children_by_parent[joint.parent].append(joint)

        roots = sorted(link_set - set(parent_by_child))
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link; found {len(roots)}")
        root_link = roots[0]

        ordered_joint_names: list[str] = []
        visited_links = {root_link}
        pending = [root_link]
        while pending:
            parent = pending.pop(0)
            for joint in sorted(children_by_parent[parent], key=lambda item: (item.name, item.child)):
                if joint.child in visited_links:
                    raise ValueError("joint topology contains a cycle")
                ordered_joint_names.append(joint.name)
                visited_links.add(joint.child)
                pending.append(joint.child)
        if len(visited_links) != len(link_names) or len(ordered_joint_names) != len(joints):
            raise ValueError("joint topology contains a cycle or disconnected links")

        for joint in joints.values():
            if joint.mimic is not None and joint.mimic.joint not in joints:
                raise ValueError(f"joint {joint.name} mimics unknown joint: {joint.mimic.joint}")
        cls._validate_mimic_graph(joints)

        return cls(
            links=tuple(link_names),
            joints=joints,
            root_link=root_link,
            topological_joint_names=tuple(ordered_joint_names),
        )

    @staticmethod
    def _validate_mimic_graph(joints: Mapping[str, UrdfJoint]) -> None:
        complete: set[str] = set()
        active: set[str] = set()

        def visit(name: str) -> None:
            if name in complete:
                return
            if name in active:
                raise ValueError("mimic cycle detected")
            active.add(name)
            mimic = joints[name].mimic
            if mimic is not None:
                visit(mimic.joint)
            active.remove(name)
            complete.add(name)

        for name in sorted(joints):
            visit(name)

    def expanded_joint_positions(
        self,
        positions: Mapping[str, object],
        *,
        mimic_override_joints: Sequence[str] = (),
    ) -> dict[str, object]:
        """Expand independent joint positions to fixed and recursive mimic joints."""

        if not isinstance(positions, Mapping):
            raise TypeError("joint positions must be a mapping")
        if isinstance(mimic_override_joints, (str, bytes)):
            raise TypeError("mimic_override_joints must be a sequence of joint names")
        overrides = tuple(mimic_override_joints)
        if len(overrides) != len(set(overrides)):
            raise ValueError("mimic_override_joints must contain unique names")
        unknown_overrides = sorted(set(overrides) - set(self.joints))
        if unknown_overrides:
            raise ValueError(f"unknown mimic override joint: {unknown_overrides}")
        non_mimics = sorted(name for name in overrides if self.joints[name].mimic is None)
        if non_mimics:
            raise ValueError(f"mimic override joint is not a mimic: {non_mimics}")
        override_set = set(overrides)
        unknown = sorted(set(positions) - set(self.joints))
        if unknown:
            raise ValueError(f"unknown joint positions: {unknown}")
        supplied: dict[str, object] = {}
        for name, value in positions.items():
            joint = self.joints[name]
            if joint.joint_type == "fixed":
                raise ValueError(f"fixed joint {name} cannot be supplied")
            if joint.mimic is not None and name not in override_set:
                raise ValueError(f"mimic joint {name} cannot be supplied directly")
            array = np.asarray(value, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("joint positions must be finite")
            supplied[name] = float(array) if array.ndim == 0 else array

        expanded: dict[str, object] = {}

        def resolve(name: str) -> object:
            if name in expanded:
                return expanded[name]
            joint = self.joints[name]
            if joint.joint_type == "fixed":
                value: object = 0.0
            elif name in supplied:
                value = supplied[name]
            elif joint.mimic is not None:
                value = joint.mimic.multiplier * resolve(joint.mimic.joint) + joint.mimic.offset
            else:
                raise ValueError(f"missing independent joint position: {name}")
            expanded[name] = value
            return value

        for name in self.topological_joint_names:
            resolve(name)
        return expanded

    def forward_links(
        self,
        q: np.ndarray,
        joint_order: Sequence[str],
        links: Sequence[str],
        *,
        mimic_override_joints: Sequence[str] = (),
    ) -> np.ndarray:
        """Return batched root-to-link transforms with shape ``(B, L, 4, 4)``."""

        values = np.asarray(q, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("q must be a two-dimensional batch")
        if not np.isfinite(values).all():
            raise ValueError("q values must be finite")
        if isinstance(joint_order, (str, bytes)):
            raise TypeError("joint_order must be a sequence of joint names")
        order = tuple(joint_order)
        if len(order) != len(set(order)):
            raise ValueError("joint_order must contain unique names")
        if values.shape[1] != len(order):
            raise ValueError("q columns must match joint_order")
        for name in order:
            if name not in self.joints:
                raise ValueError(f"unknown joint in joint_order: {name}")
        requested_links = tuple(links)
        missing_links = sorted(set(requested_links) - set(self.links))
        if missing_links:
            raise ValueError(f"missing requested link: {missing_links}")

        independent = self.expanded_joint_positions(
            {name: values[:, column] for column, name in enumerate(order)},
            mimic_override_joints=mimic_override_joints,
        )
        batch = values.shape[0]
        root_transform = np.broadcast_to(np.eye(4, dtype=np.float64), (batch, 4, 4)).copy()
        transforms: dict[str, np.ndarray] = {self.root_link: root_transform}
        for name in self.topological_joint_names:
            joint = self.joints[name]
            motion = np.broadcast_to(np.eye(4, dtype=np.float64), (batch, 4, 4)).copy()
            position = np.asarray(independent[name], dtype=np.float64)
            if joint.joint_type in {"continuous", "revolute"}:
                motion[:, :3, :3] = axis_angle_matrix(joint.axis, position)
            elif joint.joint_type == "prismatic":
                motion[:, :3, 3] = position[:, None] * joint.axis
            transforms[joint.child] = transforms[joint.parent] @ joint.origin[None, :, :] @ motion
        return np.stack([transforms[name] for name in requested_links], axis=1)

    def palm_relative_fingertip_transforms(
        self,
        q: np.ndarray,
        source_hand: SourceHandSpec,
    ) -> np.ndarray:
        """Return ``inv(T_palm) @ T_tip`` in the source manifest's five-tip order."""

        names = (source_hand.palm_link, *source_hand.fingertip_links)
        transforms = self.forward_links(
            q,
            source_hand.joint_order,
            names,
            mimic_override_joints=source_hand.mimic_override_joints,
        )
        palm_inverse = np.linalg.inv(transforms[:, 0])
        return palm_inverse[:, None, :, :] @ transforms[:, 1:, :, :]

    def palm_relative_fingertips(
        self,
        q: np.ndarray,
        source_hand: SourceHandSpec,
    ) -> np.ndarray:
        """Return palm-relative fingertip positions with shape ``(B, 5, 3)``."""

        return self.palm_relative_fingertip_transforms(q, source_hand)[:, :, :3, 3]


__all__ = [
    "JointLimit",
    "Mimic",
    "UrdfJoint",
    "UrdfKinematicTree",
    "axis_angle_matrix",
]
