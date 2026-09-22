"""Isaac Lab configuration helpers for catalog-backed object instances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectInstance,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCfg


def _ordered_enabled_instances(
    instances: Sequence[ObjectInstance],
) -> tuple[ObjectInstance, ...]:
    """Validate an instance sequence and return enabled instances by ID."""

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
    return tuple(sorted(checked, key=lambda item: item.object_id))


def select_target_and_obstacles(
    instances: Sequence[ObjectInstance], target_object_id: str | None
) -> tuple[ObjectInstance | None, tuple[ObjectInstance, ...]]:
    """Partition enabled instances into one optional target and obstacles.

    A ``None`` target ID deliberately leaves every enabled instance in the
    obstacle tuple.  This makes target selection explicit and avoids silently
    selecting an arbitrary object from a catalog.
    """

    ordered = _ordered_enabled_instances(instances)
    if target_object_id is None:
        return None, ordered
    if not isinstance(target_object_id, str) or not target_object_id:
        raise TypeError("target_object_id must be a non-empty string or None")
    target = next(
        (instance for instance in ordered if instance.object_id == target_object_id),
        None,
    )
    if target is None:
        raise ValueError(f"target_object_id is not an enabled instance: {target_object_id!r}")
    obstacles = tuple(
        instance for instance in ordered if instance.object_id != target_object_id
    )
    return target, obstacles


def _pose_parts(instance: ObjectInstance) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert an XYZ(+quaternion) catalog pose to Isaac Lab state fields."""

    if len(instance.pose) == 3:
        return instance.pose, (1.0, 0.0, 0.0, 0.0)
    if len(instance.pose) != 7:
        raise ValueError(
            f"pose for {instance.object_id!r} must contain XYZ or XYZ+quaternion values"
        )
    return instance.pose[:3], instance.pose[3:]


def build_object_scene_cfg(
    catalog: ObjectCatalog,
    instances: Sequence[ObjectInstance],
    env_regex: str,
) -> dict[str, RigidObjectCfg]:
    """Build deterministic one-asset-per-instance rigid object configs.

    Catalog validation owns class/path/hash checks.  This helper adds stable
    object IDs and scene-specific fields without changing the legacy ``Box``
    configuration used when no catalog is supplied.
    """

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if not isinstance(catalog, ObjectCatalog):
        raise TypeError("catalog must be an ObjectCatalog")
    if not isinstance(env_regex, str) or not env_regex.strip():
        raise ValueError("env_regex must be a non-empty string")

    # ObjectCatalog.validate_instances provides the catalog class checks and
    # stable ordering.  Keep disabled instances out of the scene entirely.
    ordered = _ordered_enabled_instances(catalog.validate_instances(instances))
    prim_prefix = env_regex.rstrip("/")
    configs: dict[str, RigidObjectCfg] = {}
    for instance in ordered:
        record = catalog.resolve(instance.object_class)
        pos, rot = _pose_parts(instance)
        configs[instance.object_id] = RigidObjectCfg(
            prim_path=f"{prim_prefix}/Objects/{instance.object_id}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(record.usd_path),
                scale=(record.scale, record.scale, record.scale),
                mass_props=sim_utils.MassPropertiesCfg(mass=record.mass_kg),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )
    return configs


__all__ = ["build_object_scene_cfg", "select_target_and_obstacles"]
