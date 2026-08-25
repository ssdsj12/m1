from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py"
EVENTS = ROOT / "go2_pvcnn/mdp/events.py"
CURRICULUM = ROOT / "go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py"


class _SceneEntityCfg:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


def _configure_helper():
    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "configure_folded_load_stage"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    spec = importlib.util.spec_from_file_location("folded_curriculum_for_cfg", CURRICULUM)
    curriculum = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = curriculum
    spec.loader.exec_module(curriculum)
    namespace = {"SceneEntityCfg": _SceneEntityCfg, "stage_spec": curriculum.stage_spec}
    exec(compile(module, str(CFG), "exec"), namespace)
    return namespace["configure_folded_load_stage"]


def _cfg():
    return SimpleNamespace(events=SimpleNamespace(
        reset_base=SimpleNamespace(params={}),
        reset_robot_joints=SimpleNamespace(params={}),
        physics_material=SimpleNamespace(params={}),
    ))


def _reset_helper():
    tree = ast.parse(EVENTS.read_text(encoding="utf-8"))
    names = {
        "_finite_ordered_range",
        "_canonical_joint_ids",
        "reset_folded_load_joints_by_offset",
    }
    nodes = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": _SceneEntityCfg,
        "Articulation": object,
    }
    exec(compile(module, str(EVENTS), "exec"), namespace)
    return namespace["reset_folded_load_joints_by_offset"]


class _Robot:
    def __init__(self, leg_names):
        self.device = "cpu"
        self.joint_names = list(leg_names) + [f"wheel{i}" for i in range(4)] + [f"panda{i}" for i in range(7)] + ["finger1", "finger2"]
        count = len(self.joint_names)
        default_pos = torch.arange(2 * count, dtype=torch.float32).reshape(2, count) / 100.0
        default_vel = torch.arange(2 * count, dtype=torch.float32).reshape(2, count) / 1000.0
        self.data = SimpleNamespace(
            default_joint_pos=default_pos,
            default_joint_vel=default_vel,
            soft_joint_pos_limits=torch.stack(
                (torch.full_like(default_pos, -10.0), torch.full_like(default_pos, 10.0)), dim=-1
            ),
        )
        self.write = None

    def find_joints(self, names, preserve_order=False):
        return [self.joint_names.index(name) for name in names], list(names)

    def write_joint_state_to_sim(self, joint_pos, joint_vel, *, env_ids):
        self.write = (joint_pos.clone(), joint_vel.clone(), env_ids.clone())


def test_stage_config_sets_exact_d1_d2_d3_ranges_and_protected_zeros():
    configure = _configure_helper()
    expected = {
        "L2-D1": (0.005, (0.01, 0.01, 0.01), 0.01, 0.02, 0.005, (0.95, 1.05)),
        "L2-D2": (0.01, (0.015, 0.015, 0.025), 0.025, 0.05, 0.01, (0.90, 1.10)),
        "L2-D3": (0.02, (0.03, 0.03, 0.05), 0.05, 0.10, 0.02, (0.80, 1.20)),
    }
    for stage, (xy, rpy, linear, angular, leg, friction) in expected.items():
        cfg = _cfg(); configure(cfg, stage)
        pose = cfg.events.reset_base.params["pose_range"]
        velocity = cfg.events.reset_base.params["velocity_range"]
        assert pose == {"x": (-xy, xy), "y": (-xy, xy), "z": (0.0, 0.0),
                        "roll": (-rpy[0], rpy[0]), "pitch": (-rpy[1], rpy[1]), "yaw": (-rpy[2], rpy[2])}
        assert velocity == {"x": (-linear, linear), "y": (-linear, linear), "z": (-linear, linear),
                            "roll": (-angular, angular), "pitch": (-angular, angular), "yaw": (-angular, angular)}
        assert cfg.events.reset_robot_joints.params["leg_position_range"] == (-leg, leg)
        assert "arm_position_range" not in cfg.events.reset_robot_joints.params
        assert "velocity_range" not in cfg.events.reset_robot_joints.params
        assert cfg.events.physics_material.params["static_friction_range"] == friction
        assert cfg.events.physics_material.params["dynamic_friction_range"] == friction
        assert cfg.events.physics_material.params["restitution_range"] == (0.0, 0.0)


def test_l0_l1_stage_config_is_deterministic():
    configure = _configure_helper(); cfg = _cfg(); configure(cfg, "L1-C4")
    assert all(value == (0.0, 0.0) for value in cfg.events.reset_base.params["pose_range"].values())
    assert all(value == (0.0, 0.0) for value in cfg.events.reset_base.params["velocity_range"].values())
    assert cfg.events.reset_robot_joints.params["leg_position_range"] == (0.0, 0.0)
    assert cfg.events.physics_material.params["static_friction_range"] == (1.0, 1.0)


def test_source_uses_leg_only_reset_and_never_external_wrench():
    cfg_source = CFG.read_text(encoding="utf-8")
    wrapper_source = (ROOT / "go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py").read_text(encoding="utf-8")
    events_source = EVENTS.read_text(encoding="utf-8")
    assert "reset_folded_load_joints_by_offset" in cfg_source
    assert "def reset_folded_load_joints_by_offset" in events_source
    assert "set_external_force_and_torque" not in wrapper_source
    assert "apply_external_force_torque" not in wrapper_source


def test_leg_only_reset_writes_selected_env_and_preserves_wheel_panda_velocity(monkeypatch):
    leg_names = tuple(f"leg{i}" for i in range(12))
    assets = types.ModuleType("go2_pvcnn.assets")
    assets.M1_LEG_JOINT_NAMES = leg_names
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets", assets)
    robot = _Robot(leg_names)
    env = SimpleNamespace(scene={"robot": robot})

    torch.manual_seed(5)
    _reset_helper()(env, torch.tensor([1]), (-0.02, 0.02), _SceneEntityCfg("robot"))

    joint_pos, joint_vel, env_ids = robot.write
    assert env_ids.tolist() == [1]
    assert joint_pos.shape == joint_vel.shape == (1, 25)
    offsets = joint_pos - robot.data.default_joint_pos[env_ids]
    assert offsets[:, :12].abs().max() <= 0.02
    assert offsets[:, 12:].eq(0.0).all()
    torch.testing.assert_close(joint_vel, robot.data.default_joint_vel[env_ids])
