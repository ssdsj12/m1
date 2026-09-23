"""Pure task-goal selection for catalog-backed bimanual control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Mapping

from go2_pvcnn.control.m1_bimanual_coordination.grasp_goal import (
    BimanualGraspGoal,
    OrientedBoundingBox,
    generate_bimanual_grasp_goal,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectInstance,
)


# These values are deliberately not silently applied.  They are conservative
# CPU contract fallbacks for catalogs that carry class/profile metadata but no
# measured dimensions; callers must pass ``allow_default_dimensions=True``.
DEFAULT_CATALOG_DIMENSIONS: Mapping[str, tuple[float, float, float]] = {
    "book": (0.30, 0.21, 0.035),
    "bottle": (0.075, 0.075, 0.24),
    "cup": (0.10, 0.10, 0.11),
    "bowl": (0.22, 0.22, 0.10),
    "cube": (0.12, 0.12, 0.12),
    "cylinder": (0.10, 0.10, 0.20),
}


def resolve_target_object_ids(
    instances: Sequence[ObjectInstance], target_object_id: str | None
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve one explicit target and deterministic enabled obstacles.

    ``None`` preserves the legacy ``/Box`` target.  Catalog instances remain
    visible as obstacles so omission never silently changes the play default.
    """

    if isinstance(instances, (str, bytes)):
        raise TypeError("instances must be a sequence of ObjectInstance values")
    checked: list[ObjectInstance] = []
    seen: set[str] = set()
    for instance in instances:
        if not isinstance(instance, ObjectInstance):
            raise TypeError("instances must contain ObjectInstance values")
        if instance.object_id in seen:
            raise ValueError(f"duplicate object_id: {instance.object_id!r}")
        seen.add(instance.object_id)
        if instance.enabled:
            checked.append(instance)
    ordered_ids = tuple(
        instance.object_id for instance in sorted(checked, key=lambda item: item.object_id)
    )
    if target_object_id is None:
        return None, ordered_ids
    if not isinstance(target_object_id, str) or not target_object_id:
        raise TypeError("target_object_id must be a non-empty string or None")
    if target_object_id == "box":
        return "box", ordered_ids
    if target_object_id not in ordered_ids:
        raise ValueError(f"unknown target object_id: {target_object_id!r}")
    return target_object_id, tuple(
        object_id for object_id in ordered_ids if object_id != target_object_id
    )


def build_catalog_grasp_goal(
    object_pose: Sequence[object],
    *,
    object_class: str,
    catalog: ObjectCatalog | None = None,
    dimensions: Sequence[object] | None = None,
    obb: OrientedBoundingBox | None = None,
    point_cloud: Sequence[Sequence[object]] | None = None,
    allow_default_dimensions: bool = False,
    default_dimensions: Mapping[str, Sequence[object]] | None = None,
) -> BimanualGraspGoal:
    """Build an injectable catalog-object task goal from measured geometry.

    Explicit ``dimensions``, ``obb``, and ``point_cloud`` values are preferred
    in that order.  A catalog's optional measured ``dimensions`` field is the
    next source.  The final class defaults are only available when the caller
    explicitly opts in with ``allow_default_dimensions=True``; this prevents a
    missing perception/geometry input from masquerading as a live sensor path.
    """

    if not isinstance(object_class, str) or not object_class.strip():
        raise ValueError("object_class must be a non-empty string")
    if obb is not None and (dimensions is not None or point_cloud is not None):
        raise ValueError("obb cannot be combined with dimensions or point_cloud")
    record = None
    if catalog is not None:
        if not isinstance(catalog, ObjectCatalog):
            raise TypeError("catalog must be an ObjectCatalog or None")
        record = catalog.resolve(object_class)
    profile = record.grasp_profile if record is not None else "generic"
    chosen_dimensions = dimensions
    if chosen_dimensions is None and obb is None and point_cloud is None and record is not None:
        chosen_dimensions = record.dimensions
    if chosen_dimensions is None and obb is None and point_cloud is None:
        if not allow_default_dimensions:
            raise ValueError(
                "catalog geometry is unavailable; provide dimensions, obb, or point_cloud, "
                "or explicitly enable default dimensions"
            )
        defaults = DEFAULT_CATALOG_DIMENSIONS if default_dimensions is None else default_dimensions
        try:
            chosen_dimensions = defaults[object_class]
        except KeyError as error:
            raise ValueError(
                f"no default dimensions are registered for object class {object_class!r}"
            ) from error
    return generate_bimanual_grasp_goal(
        object_pose,
        chosen_dimensions,
        obb=obb,
        point_cloud=point_cloud,
        grasp_profile=profile,
        object_class=object_class,
    )


def build_catalog_grasp_goal_provider(
    catalog: ObjectCatalog,
    instances: Sequence[ObjectInstance],
    target_object_id: str,
    *,
    geometry_provider: Callable[[object, ObjectInstance], object] | None = None,
    allow_default_dimensions: bool = False,
    default_dimensions: Mapping[str, Sequence[object]] | None = None,
) -> Callable[[object], BimanualGraspGoal]:
    """Create a runtime-injectable provider for one catalog target.

    ``geometry_provider`` is intentionally opaque to this module: it may
    return an :class:`OrientedBoundingBox`, a three-value dimensions sequence,
    or a mapping with ``dimensions``, ``obb``, and/or ``point_cloud`` keys.
    This keeps camera/point-cloud acquisition outside the control runtime.
    """

    if not isinstance(catalog, ObjectCatalog):
        raise TypeError("catalog must be an ObjectCatalog")
    ordered = tuple(instance for instance in catalog.validate_instances(instances) if instance.enabled)
    target = next((instance for instance in ordered if instance.object_id == target_object_id), None)
    if target is None:
        raise ValueError(f"target_object_id is not an enabled instance: {target_object_id!r}")
    catalog.resolve(target.object_class)

    def provider(snapshot: object) -> BimanualGraspGoal:
        geometry = None if geometry_provider is None else geometry_provider(snapshot, target)
        kwargs: dict[str, object] = {}
        if isinstance(geometry, OrientedBoundingBox):
            kwargs["obb"] = geometry
        elif isinstance(geometry, Mapping):
            unknown = set(geometry).difference({"dimensions", "obb", "point_cloud"})
            if unknown:
                raise ValueError(f"unsupported grasp geometry keys: {sorted(unknown)!r}")
            kwargs.update(geometry)
        elif geometry is not None:
            kwargs["dimensions"] = geometry
        pose = getattr(getattr(snapshot, "box", None), "pose_b", None)
        if pose is None:
            raise TypeError("snapshot must expose box.pose_b for catalog grasp-goal generation")
        return build_catalog_grasp_goal(
            pose.tolist() if hasattr(pose, "tolist") else pose,
            object_class=target.object_class,
            catalog=catalog,
            allow_default_dimensions=allow_default_dimensions,
            default_dimensions=default_dimensions,
            **kwargs,
        )

    return provider


__all__ = [
    "DEFAULT_CATALOG_DIMENSIONS",
    "build_catalog_grasp_goal",
    "build_catalog_grasp_goal_provider",
    "resolve_target_object_ids",
]
