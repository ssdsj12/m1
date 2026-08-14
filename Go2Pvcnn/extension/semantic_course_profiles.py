"""Pure shape-profile overrides for semantic course generation."""

from __future__ import annotations


def resolve_cuboid_size_override(
    *,
    semantic_class: str,
    default_shape_kind: str,
    default_shape_params: dict,
    cuboid_size_overrides: dict[str, tuple[float, float, float]] | None,
) -> tuple[str, dict]:
    """Replace one semantic class with an explicitly sized cuboid when configured."""
    if not cuboid_size_overrides or semantic_class not in cuboid_size_overrides:
        return default_shape_kind, default_shape_params
    size = tuple(float(value) for value in cuboid_size_overrides[semantic_class])
    if len(size) != 3 or any(value <= 0.0 for value in size):
        raise ValueError(f"Cuboid size must contain three positive values, got {size!r}")
    return "cuboid", {"size": size}
