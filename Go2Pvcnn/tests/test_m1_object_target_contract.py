from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualPhase,
    BimanualSnapshot,
    BoxState,
    ObjectMpcInput,
    ObjectState,
    SideArmState,
    SideHandState,
)
from go2_pvcnn.control.m1_bimanual_coordination.task_goal import (
    resolve_target_object_ids,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import ObjectInstance
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    AssetPreparationRequiredError,
)


DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[1]


def _arm() -> SideArmState:
    return SideArmState(
        q=torch.zeros(7, dtype=DTYPE),
        qd=torch.zeros(7, dtype=DTYPE),
        palm_pose_b=torch.zeros(6, dtype=DTYPE),
        palm_twist_b=torch.zeros(6, dtype=DTYPE),
        jacobian_b=torch.eye(6, 7, dtype=DTYPE),
        mass_matrix=torch.eye(7, dtype=DTYPE),
        bias=torch.zeros(7, dtype=DTYPE),
    )


def _hand() -> SideHandState:
    return SideHandState(
        q=torch.zeros(6, dtype=DTYPE),
        qd=torch.zeros(6, dtype=DTYPE),
        fingertip_forces_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_jacobian_b=torch.zeros(15, 6, dtype=DTYPE),
        contact_mask=torch.zeros(5, dtype=torch.bool),
    )


def _snapshot() -> BimanualSnapshot:
    return BimanualSnapshot(
        timestamp_ns=1,
        base_state=torch.zeros(13, dtype=DTYPE),
        m1_q=torch.zeros(16, dtype=DTYPE),
        m1_qd=torch.zeros(16, dtype=DTYPE),
        platform_q_qd=torch.zeros(2, dtype=DTYPE),
        left_arm=_arm(),
        right_arm=_arm(),
        left_hand=_hand(),
        right_hand=_hand(),
        box=BoxState(
            pose_b=torch.zeros(6, dtype=DTYPE),
            twist_b=torch.zeros(6, dtype=DTYPE),
            mass=torch.tensor(0.5, dtype=DTYPE),
            inertia_b=torch.eye(3, dtype=DTYPE),
            supported=True,
        ),
        obstacle_objects=(
            ObjectState(
                pose_b=torch.tensor(
                    [0.3, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=DTYPE
                ),
                twist_b=torch.zeros(6, dtype=DTYPE),
            ),
        ),
    )


def test_play_parser_keeps_box_default_and_accepts_explicit_object_id():
    import importlib.util

    path = ROOT / "scripts/m1_dual_panda_o6_bimanual_play.py"
    spec = importlib.util.spec_from_file_location("m1_o6_play", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._parser().parse_args([]).object_id is None
    assert module._parser().parse_args(["--object-id", "cup_000"]).object_id == "cup_000"
    assert module._parser().parse_args([]).object_catalog is None


def _load_play_module():
    import importlib.util

    path = ROOT / "scripts/m1_dual_panda_o6_bimanual_play.py"
    spec = importlib.util.spec_from_file_location("m1_o6_play_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_play_config_omission_keeps_legacy_box_and_does_not_load_catalog(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_play_module()
    calls: list[object] = []
    monkeypatch.setattr(module, "_prepare_object_scene", lambda **kwargs: calls.append(kwargs))

    class FakeCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.object_catalog = kwargs.get("object_catalog")
            self.object_instances = kwargs.get("object_instances", ())
            self.scene = type("Scene", (), {"num_envs": 99})()
            self.seed = None

    monkeypatch.setattr(module, "_load_env_cfg_type", lambda: FakeCfg)
    cfg = module.build_play_config(seed=7, object_id=None)

    assert cfg.object_catalog is None
    assert cfg.object_instances == ()
    assert cfg.scene.num_envs == 1
    assert cfg.seed == 7
    assert calls == []


def test_play_config_accepts_prepared_legacy_empty_scene(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_play_module()

    class FakeCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.object_catalog = kwargs.get("object_catalog")
            self.object_instances = kwargs.get("object_instances", ())
            self.scene = type("Scene", (), {"num_envs": 99})()
            self.seed = None

    monkeypatch.setattr(module, "_load_env_cfg_type", lambda: FakeCfg)
    cfg = module.build_play_config(seed=5, object_id=None, catalog=None, object_instances=())

    assert cfg.kwargs == {}
    assert cfg.object_catalog is None
    assert cfg.object_instances == ()
    assert cfg.scene.num_envs == 1
    assert cfg.seed == 5


def test_play_config_builds_catalog_scene_for_valid_object_id(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_play_module()
    catalog = object()
    instances = (
        ObjectInstance("bottle_000", "bottle", (0.65, 0.0, 1.20)),
        ObjectInstance("cube_000", "cube", (0.65, 0.30, 1.20)),
    )
    monkeypatch.setattr(
        module,
        "_prepare_object_scene",
        lambda **kwargs: (catalog, instances),
    )

    class FakeCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.object_catalog = kwargs["object_catalog"]
            self.object_instances = kwargs["object_instances"]
            self.scene = type("Scene", (), {"num_envs": 99})()
            self.seed = None

    monkeypatch.setattr(module, "_load_env_cfg_type", lambda: FakeCfg)
    cfg = module.build_play_config(seed=11, object_id="bottle_000")

    assert cfg.kwargs == {
        "object_catalog": catalog,
        "object_instances": instances,
    }
    assert cfg.object_catalog is catalog
    assert tuple(instance.object_id for instance in cfg.object_instances) == (
        "bottle_000",
        "cube_000",
    )
    assert cfg.scene.num_envs == 1
    assert cfg.seed == 11


def test_unknown_play_object_id_is_rejected_before_catalog_or_cfg(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_play_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_prepare_object_scene",
        lambda **kwargs: calls.append("catalog") or (_ for _ in ()).throw(
            AssertionError("catalog loader was reached")
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_env_cfg_type",
        lambda: (_ for _ in ()).throw(AssertionError("cfg constructor was reached")),
    )

    with pytest.raises(ValueError, match="unknown target object_id"):
        module.build_play_config(seed=42, object_id="missing_000")

    assert calls == []


def _write_play_catalog(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    classes: dict[str, dict[str, object]] = {}
    for object_class in ("bottle", "cube"):
        asset_path = asset_root / f"{object_class}.usd"
        asset_path.write_bytes(f"{object_class}-usd".encode())
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        classes[object_class] = {
            "usd_path": f"{object_class}.usd",
            "source_sha256": digest,
            "resolved_sha256": digest,
            "mass_kg": 0.5,
            "scale": 1.0,
            "collision_profile": "convex_decomposition",
            "grasp_profile": "two_hand_stable",
        }
    config = tmp_path / "catalog.json"
    config.write_text(
        json.dumps({"schema_version": 1, "classes": classes}, indent=2),
        encoding="utf-8",
    )
    return config, asset_root


def test_prepare_object_scene_accepts_valid_catalog_object_id(tmp_path: Path):
    module = _load_play_module()
    config, asset_root = _write_play_catalog(tmp_path)

    catalog, instances = module._prepare_object_scene(
        object_id="bottle_000",
        object_catalog=config,
        object_assets_root=asset_root,
    )

    assert tuple(catalog.classes) == ("bottle",)
    assert tuple(instance.object_id for instance in instances) == ("bottle_000",)


def _write_partial_play_catalog(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "partial-assets"
    asset_root.mkdir()
    book_path = asset_root / "book.usd"
    book_path.write_bytes(b"book-usd")
    book_digest = hashlib.sha256(book_path.read_bytes()).hexdigest()
    classes = {
        "book": {
            "usd_path": "book.usd",
            "source_sha256": book_digest,
            "resolved_sha256": book_digest,
            "mass_kg": 0.5,
            "scale": 1.0,
            "collision_profile": "convex_decomposition",
            "grasp_profile": "two_hand_stable",
        },
        "cup": {
            "usd_path": "cup.usd",
            "source_sha256": "0" * 64,
            "resolved_sha256": "0" * 64,
            "mass_kg": 0.5,
            "scale": 1.0,
            "collision_profile": "convex_decomposition",
            "grasp_profile": "two_hand_stable",
        },
    }
    config = tmp_path / "partial-catalog.json"
    config.write_text(
        json.dumps({"schema_version": 1, "classes": classes}, indent=2),
        encoding="utf-8",
    )
    return config, asset_root


def test_prepare_object_scene_only_validates_selected_book_asset(
    tmp_path: Path,
):
    module = _load_play_module()
    config, asset_root = _write_partial_play_catalog(tmp_path)

    catalog, instances = module._prepare_object_scene(
        object_id="book_000",
        object_catalog=config,
        object_assets_root=asset_root,
    )

    assert tuple(catalog.classes) == ("book",)
    assert tuple(instance.object_id for instance in instances) == ("book_000",)


def test_prepare_object_scene_still_requires_selected_cup_asset(
    tmp_path: Path,
):
    module = _load_play_module()
    config, asset_root = _write_partial_play_catalog(tmp_path)

    with pytest.raises(AssetPreparationRequiredError, match="preparation required.*cup"):
        module._prepare_object_scene(
            object_id="cup_000",
            object_catalog=config,
            object_assets_root=asset_root,
        )


def test_default_catalog_object_id_fails_with_preparation_required_error():
    module = _load_play_module()

    with pytest.raises(AssetPreparationRequiredError, match="preparation required.*cup"):
        module._prepare_object_scene(
            object_id="cup_000",
            object_catalog=None,
            object_assets_root=module.DEFAULT_OBJECT_ASSET_ROOT,
        )


def test_unknown_target_id_is_rejected_before_scene_startup():
    instances = (
        ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),
        ObjectInstance("bottle_000", "bottle", (0.5, 0.0, 0.5)),
    )

    with pytest.raises(ValueError, match="unknown target object_id"):
        resolve_target_object_ids(instances, "missing")


def test_explicit_target_takes_precedence_and_exposes_other_enabled_obstacles():
    instances = (
        ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),
        ObjectInstance("bottle_000", "bottle", (0.5, 0.0, 0.5)),
        ObjectInstance("disabled", "cube", (1.0, 0.0, 0.5), enabled=False),
    )

    target_id, obstacle_ids = resolve_target_object_ids(instances, "bottle_000")

    assert target_id == "bottle_000"
    assert obstacle_ids == ("cube_000",)


def test_legacy_omission_keeps_box_target_and_exposes_catalog_objects_as_obstacles():
    instances = (ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),)

    target_id, obstacle_ids = resolve_target_object_ids(instances, None)

    assert target_id is None
    assert obstacle_ids == ("cube_000",)


def test_mpc_goal_boundary_carries_selected_target_and_obstacle_poses():
    snapshot = _snapshot()
    target = torch.zeros((25, 6), dtype=DTYPE)
    target[:, 2] = 0.1
    obstacle = snapshot.obstacle_objects[0].pose_b.repeat(25, 1)

    sample = ObjectMpcInput(
        snapshot=snapshot,
        target_object_pose_b=target,
        obstacle_object_poses_b=(obstacle,),
        phase=BimanualPhase.APPROACH,
    )

    assert torch.equal(sample.target_object_pose_b, target)
    assert torch.equal(sample.target_box_pose_b, target)
    assert len(sample.obstacle_object_poses_b) == 1
    assert torch.equal(sample.obstacle_object_poses_b[0], obstacle)


def test_snapshot_target_alias_remains_box_compatible():
    snapshot = _snapshot()
    assert snapshot.target_object is snapshot.box
    assert len(snapshot.obstacle_objects) == 1
