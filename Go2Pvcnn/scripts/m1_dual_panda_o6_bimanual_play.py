"""GUI playback of the deterministic M1 dual-Panda O6 MPC mission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections.abc import Sequence

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectInstance,
    load_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_CATALOG = ROOT / "config" / "m1_object_catalog.json"
DEFAULT_OBJECT_ASSET_ROOT = ROOT / "assets" / "m1_objects" / "rialto"

_DEFAULT_OBJECT_POSES: dict[str, tuple[float, float, float]] = {
    "bottle": (0.65, 0.00, 1.20),
    "cup": (0.65, 0.28, 1.20),
    "bowl": (0.65, -0.28, 1.20),
    "book": (0.88, 0.00, 1.20),
    "cube": (0.88, 0.28, 1.20),
    "cylinder": (0.88, -0.28, 1.20),
}
_DEFAULT_OBJECT_IDS = frozenset(
    f"{object_class}_000" for object_class in _DEFAULT_OBJECT_POSES
)


def _metadata_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "value must be 64 lowercase hexadecimal characters"
        )
    return value


def _validate_prior_pair(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        args.fingertip_prior_artifact is not None
        and args.fingertip_prior_metadata_sha256 is None
    ):
        parser.error(
            "--fingertip-prior-artifact requires "
            "--fingertip-prior-metadata-sha256"
        )
    if (
        args.fingertip_prior_artifact is None
        and args.fingertip_prior_metadata_sha256 is not None
    ):
        parser.error(
            "--fingertip-prior-metadata-sha256 requires "
            "--fingertip-prior-artifact"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpc-backend", choices=("reference-cpu", "bimanual-rti-cuda", "auto"), default="reference-cpu")
    parser.add_argument("--qp-backend", choices=("reference-cpu", "osqp-cuda", "auto"), default="reference-cpu")
    parser.add_argument("--control-device", default="cuda:0", help="requested control device; reference-cpu always uses CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--object-id", default=None)
    parser.add_argument(
        "--object-catalog",
        type=Path,
        default=None,
        help=(
            "optional object catalog JSON; omitted keeps the legacy /Box scene, "
            "while --object-id uses the default catalog"
        ),
    )
    parser.add_argument(
        "--object-assets-root",
        type=Path,
        default=DEFAULT_OBJECT_ASSET_ROOT,
        help="local prepared object asset root used with --object-catalog",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--mode", choices=("teacher", "latent"), default="teacher")
    parser.add_argument("--fingertip-prior-artifact", type=Path, default=None)
    parser.add_argument(
        "--fingertip-prior-metadata-sha256", type=_metadata_sha256, default=None
    )
    return parser


def _catalog_class_names(path: Path) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"object catalog does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read object catalog: {path}") from error
    classes = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(classes, dict) or not classes:
        raise ValueError("object catalog classes must be a non-empty object")
    for object_class in classes:
        if not isinstance(object_class, str) or not object_class:
            raise ValueError("object catalog class names must be non-empty strings")
    return tuple(classes)


def _fallback_pose(index: int) -> tuple[float, float, float]:
    x = 0.65 + 0.23 * (index // 3)
    y = (0.28, -0.28, 0.56)[index % 3]
    return (x, y, 1.20)


def default_object_instances(
    object_classes: Sequence[str],
) -> tuple[ObjectInstance, ...]:
    """Build one deterministic enabled instance per catalog class."""

    if isinstance(object_classes, (str, bytes)):
        raise TypeError("object_classes must be a sequence of class names")
    instances: list[ObjectInstance] = []
    seen: set[str] = set()
    for index, object_class in enumerate(object_classes):
        if not isinstance(object_class, str) or not object_class:
            raise ValueError("object class names must be non-empty strings")
        if object_class in seen:
            raise ValueError(f"duplicate object class: {object_class!r}")
        seen.add(object_class)
        instances.append(
            ObjectInstance(
                f"{object_class}_000",
                object_class,
                _DEFAULT_OBJECT_POSES.get(object_class, _fallback_pose(index)),
            )
        )
    return tuple(instances)


def _validate_target_object_id(
    object_id: str | None, instances: Sequence[ObjectInstance]
) -> None:
    if object_id is None or object_id == "box":
        return
    if not isinstance(object_id, str) or not object_id:
        raise TypeError("target object_id must be a non-empty string or None")
    valid_ids = {instance.object_id for instance in instances if instance.enabled}
    if object_id not in valid_ids:
        available = ", ".join(sorted(valid_ids))
        raise ValueError(
            f"unknown target object_id: {object_id!r}; available catalog object IDs: "
            f"{available}"
        )


def _uses_catalog_scene(object_id: str | None, object_catalog: Path | None) -> bool:
    return object_catalog is not None or (object_id not in {None, "box"})


def _prepare_object_scene(
    *,
    object_id: str | None,
    object_catalog: Path | None,
    object_assets_root: Path,
) -> tuple[ObjectCatalog | None, tuple[ObjectInstance, ...]]:
    if not _uses_catalog_scene(object_id, object_catalog):
        return None, ()
    catalog_path = Path(object_catalog) if object_catalog is not None else DEFAULT_OBJECT_CATALOG
    instances = default_object_instances(_catalog_class_names(catalog_path))
    _validate_target_object_id(object_id, instances)
    catalog = load_catalog(catalog_path, Path(object_assets_root))
    return catalog, instances


def _load_env_cfg_type():
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import (
        M1DualPandaO6BimanualEnvCfg,
    )

    return M1DualPandaO6BimanualEnvCfg


def build_play_config(
    *,
    seed: int,
    object_id: str | None,
    object_catalog: Path | None = None,
    object_assets_root: Path = DEFAULT_OBJECT_ASSET_ROOT,
    catalog: ObjectCatalog | None = None,
    object_instances: tuple[ObjectInstance, ...] | None = None,
):
    """Construct the env cfg used by shipped play before ``gym.make``."""

    if catalog is None and object_instances is None and not _uses_catalog_scene(
        object_id, object_catalog
    ):
        object_instances = ()
    elif (
        catalog is None
        and object_instances is None
        and object_catalog is None
        and object_id not in _DEFAULT_OBJECT_IDS
    ):
        available = ", ".join(sorted(_DEFAULT_OBJECT_IDS))
        raise ValueError(
            f"unknown target object_id: {object_id!r}; available catalog object IDs: "
            f"{available}"
        )
    elif catalog is None and object_instances is None:
        catalog, object_instances = _prepare_object_scene(
            object_id=object_id,
            object_catalog=object_catalog,
            object_assets_root=object_assets_root,
        )
    elif (
        catalog is None
        and object_instances == ()
        and object_catalog is None
        and object_id in {None, "box"}
    ):
        pass
    elif catalog is None or object_instances is None:
        raise ValueError("catalog and object_instances must be supplied together")
    _validate_target_object_id(object_id, object_instances)

    cfg_type = _load_env_cfg_type()
    if catalog is None:
        cfg = cfg_type()
    else:
        cfg = cfg_type(
            object_catalog=catalog,
            object_instances=tuple(object_instances),
        )
    cfg.scene.num_envs = 1
    cfg.seed = seed
    return cfg


def main() -> int:
    parser = _parser()
    early_args, _unknown_launcher_args = parser.parse_known_args()
    _validate_prior_pair(parser, early_args)
    from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.config import resolve_backend_selection

    try:
        backend_selection = resolve_backend_selection(early_args.mpc_backend, early_args.qp_backend, early_args.control_device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    parser.set_defaults(backend_selection=backend_selection)
    binding = None
    try:
        if early_args.fingertip_prior_artifact is not None:
            from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime_binding import (
                PreparedO6FingertipPriors,
            )

            binding = PreparedO6FingertipPriors.from_artifact(
                early_args.fingertip_prior_artifact,
                expected_metadata_sha256=early_args.fingertip_prior_metadata_sha256,
            )
        return _run_with_fingertip_prior(parser, binding)
    finally:
        if binding is not None:
            binding.close()


def _run_with_fingertip_prior(parser, binding) -> int:
    # The pinned left/right workers are already bound. Never reload their
    # mutable artifact path after crossing the launcher/scene boundary.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        catalog, object_instances = _prepare_object_scene(
            object_id=args.object_id,
            object_catalog=args.object_catalog,
            object_assets_root=args.object_assets_root,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.diagnostics:
        print(f"backend_selection={args.backend_selection}", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import gymnasium as gym

    import go2_pvcnn.tasks  # noqa: F401
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import (
        M1DualPandaO6BimanualWrapper,
    )

    cfg = build_play_config(
        seed=args.seed,
        object_id=args.object_id,
        catalog=catalog,
        object_instances=object_instances,
    )
    env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
    with M1DualPandaO6BimanualWrapper(
        env,
        mode=args.mode,
        object_id=args.object_id,
        fingertip_prior_artifact=args.fingertip_prior_artifact,
        fingertip_prior_metadata_sha256=args.fingertip_prior_metadata_sha256,
        fingertip_prior_binding=binding,
    ) as wrapper:
        wrapper.reset(seed=args.seed)
        previous_phase = wrapper.runtime.mission.phase.name
        for step in range(args.max_steps):
            wrapper.step()
            phase = wrapper.runtime.mission.phase.name
            if args.diagnostics and (phase != previous_phase or phase in {"DONE", "TERMINATED"}):
                latest = wrapper.runtime.latest_solutions
                reasons = {
                    key: None if value is None else value.diagnostics.fallback_reason
                    for key, value in latest.items()
                    if key != "arm"
                }
                print(
                    f"step={step + 1} phase={phase} feasible={wrapper.last_command.feasible} "
                    f"reasons={reasons} base_state={wrapper.last_snapshot.base_state.tolist()}",
                    flush=True,
                )
            previous_phase = phase
            if phase in {"DONE", "TERMINATED"}:
                break
            if not simulation_app.is_running():
                break
        print(f"final_phase={wrapper.runtime.mission.phase.name}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
