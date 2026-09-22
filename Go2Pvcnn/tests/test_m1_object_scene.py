"""Tests for deterministic multi-instance RialTo scene configuration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectClassRecord,
    ObjectInstance,
)


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RigidObjectCfg(_Cfg):
    class InitialStateCfg(_Cfg):
        pass


def _install_isaaclab_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = types.ModuleType("isaaclab.sim")
    for name in (
        "UsdFileCfg",
        "RigidBodyPropertiesCfg",
        "MassPropertiesCfg",
        "CollisionPropertiesCfg",
    ):
        setattr(sim, name, _Cfg)
    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObjectCfg = _RigidObjectCfg
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.sim = sim
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)


def _catalog(tmp_path: Path) -> ObjectCatalog:
    records = {}
    for name, mass, scale in (("bottle", 0.4, 1.0), ("cube", 1.2, 0.75)):
        path = tmp_path / f"{name}.usd"
        path.write_text("#usda", encoding="utf-8")
        records[name] = ObjectClassRecord(
            object_class=name,
            usd_path=path,
            source_sha256="a" * 64,
            resolved_sha256="b" * 64,
            mass_kg=mass,
            scale=scale,
            collision_profile="convex_decomposition",
            grasp_profile="two_hand_stable",
        )
    return ObjectCatalog(records, {})


def test_explicit_target_selection_partitions_enabled_obstacles() -> None:
    from go2_pvcnn.tasks.m1_object_scene import select_target_and_obstacles

    target = ObjectInstance("target", "cube", (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0))
    obstacle = ObjectInstance("obstacle", "bottle", (1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0))
    disabled = ObjectInstance(
        "disabled", "bottle", (2.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0), enabled=False
    )

    selected, obstacles = select_target_and_obstacles(
        (obstacle, disabled, target), "target"
    )

    assert selected == target
    assert obstacles == (obstacle,)


def test_target_is_optional_and_missing_target_id_is_rejected() -> None:
    from go2_pvcnn.tasks.m1_object_scene import select_target_and_obstacles

    instances = (ObjectInstance("a", "cube", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),)
    selected, obstacles = select_target_and_obstacles(instances, None)
    assert selected is None
    assert obstacles == instances

    with pytest.raises(ValueError, match="target_object_id"):
        select_target_and_obstacles(instances, "missing")


def test_build_object_scene_cfg_uses_unique_deterministic_paths_and_class_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_isaaclab_stubs(monkeypatch)
    sys.modules.pop("go2_pvcnn.tasks.m1_object_scene", None)
    scene = importlib.import_module("go2_pvcnn.tasks.m1_object_scene")
    catalog = _catalog(tmp_path)
    instances = (
        ObjectInstance("cube-02", "cube", (0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)),
        ObjectInstance("bottle-01", "bottle", (0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)),
        ObjectInstance("disabled", "cube", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), enabled=False),
    )

    configs = scene.build_object_scene_cfg(catalog, instances, "{ENV_REGEX_NS}")

    assert tuple(configs) == ("bottle-01", "cube-02")
    assert len({cfg.prim_path for cfg in configs.values()}) == 2
    assert configs["bottle-01"].prim_path == "{ENV_REGEX_NS}/Objects/bottle-01"
    assert configs["cube-02"].prim_path == "{ENV_REGEX_NS}/Objects/cube-02"
    assert configs["bottle-01"].spawn.usd_path == str(catalog.resolve("bottle").usd_path)
    assert configs["bottle-01"].spawn.mass_props.mass == pytest.approx(0.4)
    assert configs["cube-02"].spawn.scale == (0.75, 0.75, 0.75)
    assert configs["cube-02"].init_state.pos == (0.2, 0.0, 0.5)
    assert configs["cube-02"].init_state.rot == (1.0, 0.0, 0.0, 0.0)


def test_legacy_catalog_does_not_create_object_entries(tmp_path: Path, monkeypatch):
    _install_isaaclab_stubs(monkeypatch)
    sys.modules.pop("go2_pvcnn.tasks.m1_object_scene", None)
    scene = importlib.import_module("go2_pvcnn.tasks.m1_object_scene")

    legacy = ObjectCatalog({}, {}, uses_legacy_box=True)

    assert scene.build_object_scene_cfg(legacy, (), "{ENV_REGEX_NS}") == {}


@pytest.mark.parametrize(
    "object_id",
    ("box", "robot", "left_arm", "right_arm", "o6_contacts"),
)
def test_scene_object_ids_reject_existing_scene_fields(object_id: str) -> None:
    from go2_pvcnn.tasks.m1_object_scene import validate_scene_object_ids

    class _Scene:
        box = object()
        robot = object()
        left_arm = object()
        right_arm = object()
        o6_contacts = object()

    with pytest.raises(ValueError, match=object_id):
        validate_scene_object_ids(_Scene(), (object_id,))


def test_bimanual_env_cfg_keeps_legacy_box_when_catalog_is_unspecified() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "object_catalog: ObjectCatalog | None = None" in source
    assert "object_instances: tuple[ObjectInstance, ...] = ()" in source
    assert "if self.object_catalog is not None:" in source
    assert source.index(
        "validate_scene_object_ids(self.scene, object_configs)"
    ) < source.index("setattr(self.scene, object_id, object_cfg)")
    assert 'prim_path="{ENV_REGEX_NS}/Box"' in source
    assert 'box = RigidObjectCfg(' in source
