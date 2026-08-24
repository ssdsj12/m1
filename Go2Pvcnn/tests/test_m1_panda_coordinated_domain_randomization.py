from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "go2_pvcnn/mdp/events.py"
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py"

M1_LEG_JOINT_NAMES = tuple(f"LEG_{index}" for index in range(12))
M1_WHEEL_JOINT_NAMES = tuple(f"WHEEL_{index}" for index in range(4))
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))


class SceneEntityCfg:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs


@pytest.fixture
def reset_coordinated_joints_by_offset(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    managers = types.ModuleType("isaaclab.managers")
    terrains = types.ModuleType("isaaclab.terrains")
    assets.RigidObjectCollection = type("RigidObjectCollection", (), {})
    assets.Articulation = type("Articulation", (), {})
    managers.SceneEntityCfg = SceneEntityCfg
    terrains.TerrainImporter = type("TerrainImporter", (), {})
    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.terrains = terrains
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrains)

    go2_pvcnn = types.ModuleType("go2_pvcnn")
    robot_assets = types.ModuleType("go2_pvcnn.assets")
    panda_assets = types.ModuleType("go2_pvcnn.assets.m1_panda")
    robot_assets.M1_LEG_JOINT_NAMES = M1_LEG_JOINT_NAMES
    robot_assets.M1_WHEEL_JOINT_NAMES = M1_WHEEL_JOINT_NAMES
    panda_assets.M1_PANDA_WBC_CONTROLLED_JOINT_NAMES = (
        M1_LEG_JOINT_NAMES + M1_WHEEL_JOINT_NAMES + PANDA_ARM_JOINT_NAMES
    )
    monkeypatch.setitem(sys.modules, "go2_pvcnn", go2_pvcnn)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets", robot_assets)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets.m1_panda", panda_assets)

    spec = importlib.util.spec_from_file_location("coordinated_events_under_test", EVENTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reset_coordinated_joints_by_offset


def _load_configure_helper():
    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "configure_coordinated_training_domain_randomization"
    ]
    assert len(functions) == 1, "missing domain-randomization configuration helper"
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"SceneEntityCfg": SceneEntityCfg}
    exec(compile(module, str(CFG), "exec"), namespace)
    return namespace["configure_coordinated_training_domain_randomization"]


def _fake_cfg():
    return SimpleNamespace(
        events=SimpleNamespace(
            reset_base=SimpleNamespace(params={}),
            reset_robot_joints=SimpleNamespace(params={}),
            physics_material=SimpleNamespace(params={}),
        )
    )


class FakeRobot:
    def __init__(self, *, missing_joint: str | None = None) -> None:
        names = (
            list(M1_LEG_JOINT_NAMES)
            + list(M1_WHEEL_JOINT_NAMES)
            + list(PANDA_ARM_JOINT_NAMES)
            + ["panda_finger_joint1", "panda_finger_joint2"]
        )
        if missing_joint is not None:
            names.remove(missing_joint)
        self.joint_names = names
        self.device = "cpu"
        joint_count = len(names)
        default_pos = torch.linspace(-0.2, 0.2, joint_count).repeat(2, 1)
        default_vel = torch.zeros(2, joint_count)
        self.data = SimpleNamespace(
            default_joint_pos=default_pos,
            default_joint_vel=default_vel,
            soft_joint_pos_limits=torch.stack(
                (
                    torch.full_like(default_pos, -0.21),
                    torch.full_like(default_pos, 0.21),
                ),
                dim=-1,
            ),
            soft_joint_vel_limits=torch.full_like(default_vel, 0.04),
        )
        self.write_count = 0
        self.last_written_state = None

    def find_joints(self, requested, preserve_order=False):
        found = [name for name in requested if name in self.joint_names]
        ids = [self.joint_names.index(name) for name in found]
        return ids, found

    def write_joint_state_to_sim(self, joint_pos, joint_vel, *, env_ids):
        self.write_count += 1
        self.last_written_state = SimpleNamespace(
            joint_pos=joint_pos.clone(),
            joint_vel=joint_vel.clone(),
            env_ids=env_ids.clone(),
        )


class FakeEnv:
    def __init__(self, *, missing_joint: str | None = None) -> None:
        self.robot = FakeRobot(missing_joint=missing_joint)
        self.scene = {"robot": self.robot}


def _ids(robot: FakeRobot, names) -> torch.Tensor:
    return torch.tensor([robot.joint_names.index(name) for name in names])


def test_coordinated_joint_reset_uses_separate_ranges_and_keeps_wheels_default(
    reset_coordinated_joints_by_offset,
) -> None:
    torch.manual_seed(7)
    env = FakeEnv()
    env_ids = torch.tensor([0])

    reset_coordinated_joints_by_offset(
        env,
        env_ids,
        (-0.02, 0.02),
        (-0.03, 0.03),
        (-0.05, 0.05),
        SceneEntityCfg("robot"),
    )

    written = env.robot.last_written_state
    assert env.robot.write_count == 1
    leg_ids = _ids(env.robot, M1_LEG_JOINT_NAMES)
    wheel_ids = _ids(env.robot, M1_WHEEL_JOINT_NAMES)
    arm_ids = _ids(env.robot, PANDA_ARM_JOINT_NAMES)
    controlled_ids = torch.cat((leg_ids, wheel_ids, arm_ids))
    offsets = written.joint_pos - env.robot.data.default_joint_pos[env_ids]
    assert torch.all(offsets[:, leg_ids].abs() <= 0.02 + 1.0e-7)
    assert torch.all(offsets[:, arm_ids].abs() <= 0.03 + 1.0e-7)
    assert torch.equal(
        written.joint_pos[:, wheel_ids],
        env.robot.data.default_joint_pos[env_ids][:, wheel_ids],
    )
    assert torch.all(written.joint_vel[:, controlled_ids].abs() <= 0.04)
    assert torch.isfinite(written.joint_pos).all()
    assert torch.isfinite(written.joint_vel).all()


def test_invalid_range_or_missing_joint_fails_before_atomic_write(
    reset_coordinated_joints_by_offset,
) -> None:
    invalid = FakeEnv()
    with pytest.raises(ValueError, match="finite ordered range"):
        reset_coordinated_joints_by_offset(
            invalid,
            torch.tensor([0]),
            (float("nan"), 0.02),
            (-0.03, 0.03),
            (-0.05, 0.05),
            SceneEntityCfg("robot"),
        )
    assert invalid.robot.write_count == 0

    missing = FakeEnv(missing_joint=PANDA_ARM_JOINT_NAMES[-1])
    with pytest.raises(ValueError, match="canonical joints"):
        reset_coordinated_joints_by_offset(
            missing,
            torch.tensor([0]),
            (-0.02, 0.02),
            (-0.03, 0.03),
            (-0.05, 0.05),
            SceneEntityCfg("robot"),
        )
    assert missing.robot.write_count == 0


def test_training_dr_helper_sets_exact_ranges() -> None:
    configure_coordinated_training_domain_randomization = _load_configure_helper()
    cfg = _fake_cfg()
    configure_coordinated_training_domain_randomization(cfg, True)

    pose = cfg.events.reset_base.params["pose_range"]
    velocity = cfg.events.reset_base.params["velocity_range"]
    joints = cfg.events.reset_robot_joints.params
    material = cfg.events.physics_material.params
    assert pose == {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (0.0, 0.0),
        "roll": (-0.03, 0.03),
        "pitch": (-0.03, 0.03),
        "yaw": (-0.05, 0.05),
    }
    assert velocity == {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
        "roll": (-0.10, 0.10),
        "pitch": (-0.10, 0.10),
        "yaw": (-0.10, 0.10),
    }
    assert joints["leg_position_range"] == (-0.02, 0.02)
    assert joints["arm_position_range"] == (-0.03, 0.03)
    assert joints["velocity_range"] == (-0.05, 0.05)
    assert material["static_friction_range"] == (0.8, 1.2)
    assert material["dynamic_friction_range"] == (0.8, 1.2)
    assert material["restitution_range"] == (0.0, 0.0)
    assert material["num_buckets"] == 64


def test_defaults_and_disabled_helper_are_deterministic_and_instance_local() -> None:
    configure_coordinated_training_domain_randomization = _load_configure_helper()
    cfg = _fake_cfg()
    other = _fake_cfg()
    configure_coordinated_training_domain_randomization(cfg, False)
    configure_coordinated_training_domain_randomization(other, False)
    configure_coordinated_training_domain_randomization(cfg, True)
    configure_coordinated_training_domain_randomization(cfg, False)

    assert all(value == (0.0, 0.0) for value in cfg.events.reset_base.params["pose_range"].values())
    assert all(value == (0.0, 0.0) for value in cfg.events.reset_base.params["velocity_range"].values())
    assert cfg.events.reset_robot_joints.params["leg_position_range"] == (0.0, 0.0)
    assert cfg.events.reset_robot_joints.params["arm_position_range"] == (0.0, 0.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert cfg.events.physics_material.params["static_friction_range"] == (1.0, 1.0)
    assert other.events.reset_robot_joints.params["leg_position_range"] == (0.0, 0.0)
