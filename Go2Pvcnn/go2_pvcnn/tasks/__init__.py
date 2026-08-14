"""Active M1 task registration for this adapted copy."""

try:
    import go2_pvcnn.tasks.register_m1_envs  # noqa: F401
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith(("isaaclab", "pxr")):
        raise

__all__: list[str] = []
