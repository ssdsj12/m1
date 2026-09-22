"""Pure task-goal selection for catalog-backed bimanual control."""

from __future__ import annotations

from collections.abc import Sequence

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import ObjectInstance


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


__all__ = ["resolve_target_object_ids"]
