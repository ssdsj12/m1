from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import sys
import types

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualCommand,
    BimanualPhase,
    BimanualSnapshot,
    BoxState,
    SideArmState,
    SideHandState,
    validate_monotonic_snapshot,
)


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result

    def copy(self):
        return copy.deepcopy(self)


def _arm_state() -> SideArmState:
    return SideArmState(
        q=torch.zeros(7, dtype=torch.float64),
        qd=torch.zeros(7, dtype=torch.float64),
        palm_pose_b=torch.zeros(6, dtype=torch.float64),
        palm_twist_b=torch.zeros(6, dtype=torch.float64),
        jacobian_b=torch.zeros(6, 7, dtype=torch.float64),
        mass_matrix=torch.eye(7, dtype=torch.float64),
        bias=torch.zeros(7, dtype=torch.float64),
    )


def _hand_state() -> SideHandState:
    return SideHandState(
        q=torch.zeros(6, dtype=torch.float64),
        qd=torch.zeros(6, dtype=torch.float64),
        fingertip_forces_b=torch.zeros(5, 3, dtype=torch.float64),
        fingertip_positions_b=torch.zeros(5, 3, dtype=torch.float64),
        fingertip_jacobian_b=torch.zeros(15, 6, dtype=torch.float64),
        contact_mask=torch.zeros(5, dtype=torch.bool),
    )


def _snapshot(timestamp_ns: int = 10) -> BimanualSnapshot:
    return BimanualSnapshot(
        timestamp_ns=timestamp_ns,
        base_state=torch.zeros(13, dtype=torch.float64),
        m1_q=torch.zeros(16, dtype=torch.float64),
        m1_qd=torch.zeros(16, dtype=torch.float64),
        platform_q_qd=torch.zeros(2, dtype=torch.float64),
        left_arm=_arm_state(),
        right_arm=_arm_state(),
        left_hand=_hand_state(),
        right_hand=_hand_state(),
        box=BoxState(
            pose_b=torch.zeros(6, dtype=torch.float64),
            twist_b=torch.zeros(6, dtype=torch.float64),
            mass=torch.tensor(0.5, dtype=torch.float64),
            inertia_b=torch.eye(3, dtype=torch.float64),
            supported=True,
        ),
    )


@pytest.fixture()
def contract(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.UsdFileCfg = _Cfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg

    actuators = types.ModuleType("isaaclab.actuators")
    actuators.DCMotorCfg = _Cfg
    actuators.ImplicitActuatorCfg = _Cfg

    assets_pkg = types.ModuleType("isaaclab.assets")
    articulation = types.ModuleType("isaaclab.assets.articulation")

    class _ArticulationCfg(_Cfg):
        InitialStateCfg = _Cfg

    articulation.ArticulationCfg = _ArticulationCfg
    utils_pkg = types.ModuleType("isaaclab.utils")
    utils_assets = types.ModuleType("isaaclab.utils.assets")
    utils_assets.ISAACLAB_NUCLEUS_DIR = "/Isaac/Nucleus"
    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.sim": sim,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets_pkg,
        "isaaclab.assets.articulation": articulation,
        "isaaclab.utils": utils_pkg,
        "isaaclab.utils.assets": utils_assets,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    for name in tuple(sys.modules):
        if name == "go2_pvcnn.assets" or name.startswith("go2_pvcnn.assets."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("go2_pvcnn.assets.m1_dual_panda_o6")


def test_active_order_is_16_plus_1_plus_14_plus_12(contract):
    names = contract.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES
    assert contract.M1_DUAL_PANDA_O6_ACTIVE_DOF_COUNT == 43
    assert len(names) == len(set(names)) == 43
    assert names[:16] == contract.M1_BASE_ACTIVE_JOINT_NAMES
    assert names[16] == "dual_arm_platform_yaw_joint"
    assert names[17:24] == tuple(f"left_panda_joint{i}" for i in range(1, 8))
    assert names[24:31] == tuple(f"right_panda_joint{i}" for i in range(1, 8))
    assert names[31:37] == contract.LEFT_O6_ACTIVE_JOINT_NAMES
    assert names[37:43] == contract.RIGHT_O6_ACTIVE_JOINT_NAMES


def test_side_specific_arm_and_hand_orders_are_explicit(contract):
    assert contract.LEFT_PANDA_ACTIVE_JOINT_NAMES == tuple(
        f"left_panda_joint{i}" for i in range(1, 8)
    )
    assert contract.RIGHT_PANDA_ACTIVE_JOINT_NAMES == tuple(
        f"right_panda_joint{i}" for i in range(1, 8)
    )
    expected_o6 = (
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )
    assert contract.LEFT_O6_ACTIVE_JOINT_NAMES == tuple(f"left_{name}" for name in expected_o6)
    assert contract.RIGHT_O6_ACTIVE_JOINT_NAMES == tuple(f"right_{name}" for name in expected_o6)


def test_o6_mimics_are_metadata_not_active_channels(contract):
    assert contract.O6_MIMIC_MAP == {
        "thumb_ip": ("thumb_cmc_pitch", 1.86, 0.0),
        "index_dip": ("index_mcp_pitch", 0.89, 0.0),
        "middle_dip": ("middle_mcp_pitch", 0.89, 0.0),
        "ring_dip": ("ring_mcp_pitch", 0.89, 0.0),
        "pinky_dip": ("pinky_mcp_pitch", 0.89, 0.0),
    }
    for side in ("left", "right"):
        for mimic_name in contract.O6_MIMIC_MAP:
            assert f"{side}_{mimic_name}" not in contract.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES


def test_runtime_body_names_match_side_prefixed_asset_contract(contract):
    assert contract.M1_DUAL_PANDA_O6_BASE_BODY_NAME == "BASE_LINK"
    assert contract.M1_DUAL_PANDA_O6_PLATFORM_BODY_NAME == "platform"
    assert contract.LEFT_PANDA_WRIST_BODY_NAME == "left_panda_link8"
    assert contract.RIGHT_PANDA_WRIST_BODY_NAME == "right_panda_link8"
    assert contract.LEFT_O6_PALM_BODY_NAME == "left_hand_base_link"
    assert contract.RIGHT_O6_PALM_BODY_NAME == "right_hand_base_link"
    assert contract.LEFT_O6_FINGERTIP_BODY_NAMES == tuple(
        f"left_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
    )
    assert contract.RIGHT_O6_FINGERTIP_BODY_NAMES == tuple(
        f"right_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
    )


def test_cfg_is_project_owned_and_has_isolated_actuator_groups(contract):
    assert contract.M1_DUAL_PANDA_O6_USD_PATH.endswith(
        "assets/m1_dual_panda_o6/m1_dual_panda_o6.usd"
    )
    cfg = contract.M1_DUAL_PANDA_O6_CFG
    assert cfg.spawn.usd_path == contract.M1_DUAL_PANDA_O6_USD_PATH
    assert set(cfg.actuators) == {
        "legs",
        "wheels",
        "platform",
        "left_panda_shoulder",
        "left_panda_forearm",
        "right_panda_shoulder",
        "right_panda_forearm",
        "left_o6",
        "right_o6",
    }
    platform = cfg.actuators["platform"]
    assert platform.joint_names_expr == [contract.M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME]
    assert platform.velocity_limit_sim == pytest.approx(0.25)
    for side in ("left", "right"):
        hand = cfg.actuators[f"{side}_o6"]
        expected = (
            contract.LEFT_O6_ACTIVE_JOINT_NAMES
            if side == "left"
            else contract.RIGHT_O6_ACTIVE_JOINT_NAMES
        )
        assert tuple(hand.joint_names_expr) == expected
        assert hand.velocity_limit_sim == pytest.approx(1.0)
        assert hand.effort_limit_sim < 100.0


def test_platform_initial_position_and_soft_limit_contract(contract):
    cfg = contract.M1_DUAL_PANDA_O6_CFG
    assert cfg.init_state.joint_pos[contract.M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME] == pytest.approx(0.0)
    assert cfg.init_state.joint_pos["left_(thumb|index|middle|ring|pinky)_.*"] >= 0.15
    assert cfg.init_state.joint_pos["right_(thumb|index|middle|ring|pinky)_.*"] >= 0.15
    assert cfg.soft_joint_pos_limit_factor == pytest.approx(0.9)
    assert contract.M1_DUAL_PANDA_O6_PLATFORM_YAW_LIMIT_RAD == pytest.approx(
        (-1.5707963267948966, 1.5707963267948966)
    )


def test_runtime_joint_mapping_preserves_canonical_order(contract):
    runtime_names = tuple(reversed(contract.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES))
    ids = contract.resolve_active_joint_ids(runtime_names)
    assert tuple(runtime_names[index] for index in ids) == contract.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES

    with pytest.raises(ValueError, match="missing"):
        contract.resolve_active_joint_ids(runtime_names[1:])
    with pytest.raises(ValueError, match="duplicate"):
        contract.resolve_active_joint_ids(runtime_names + (runtime_names[-1],))


def test_package_lazily_reexports_primary_contract(contract):
    package = importlib.import_module("go2_pvcnn.assets")
    assert package.M1_DUAL_PANDA_O6_CFG is contract.M1_DUAL_PANDA_O6_CFG
    assert package.M1_DUAL_PANDA_O6_ACTIVE_DOF_COUNT == 43
    assert package.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES == (
        contract.M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES
    )


def test_snapshot_rejects_nonmonotonic_and_nonfinite_hand_state():
    valid = _snapshot(timestamp_ns=10)
    with pytest.raises(ValueError, match="monotonic"):
        validate_monotonic_snapshot(valid, replace(valid, timestamp_ns=10))
    with pytest.raises(ValueError, match="finite"):
        replace(
            valid,
            left_hand=replace(
                valid.left_hand,
                q=torch.full((6,), float("nan"), dtype=torch.float64),
            ),
        )


def test_snapshot_tensors_are_exact_cpu_float64_and_caller_isolated():
    source = torch.arange(7, dtype=torch.float64)
    arm = replace(_arm_state(), q=source)
    source.add_(100.0)
    assert torch.equal(arm.q, torch.arange(7, dtype=torch.float64))
    assert arm.q.data_ptr() != source.data_ptr()

    with pytest.raises(TypeError, match="float64"):
        replace(arm, q=torch.zeros(7, dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        replace(arm, jacobian_b=torch.zeros(7, 6, dtype=torch.float64))
    with pytest.raises(TypeError, match="bool"):
        replace(_hand_state(), contact_mask=torch.zeros(5, dtype=torch.float64))


def test_box_mass_and_snapshot_timestamp_are_strict():
    valid = _snapshot()
    with pytest.raises(ValueError, match="positive"):
        replace(valid.box, mass=torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="symmetric"):
        replace(
            valid.box,
            inertia_b=torch.tensor(
                [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
        )
    with pytest.raises(ValueError, match="positive integer"):
        replace(valid, timestamp_ns=0)


def test_command_has_exact_43_channel_order_and_is_caller_isolated():
    source = torch.arange(43, dtype=torch.float64)
    command = BimanualCommand(
        timestamp_ns=11,
        effort=source,
        feasible=True,
        fallback_reasons=(),
    )
    source.zero_()
    assert command.effort.shape == (43,)
    assert command.timestamp_ns > 0
    assert command.effort[42].item() == pytest.approx(42.0)
    with pytest.raises(ValueError, match="shape"):
        replace(command, effort=torch.zeros(42, dtype=torch.float64))
    with pytest.raises(ValueError, match="fallback_reasons"):
        replace(command, fallback_reasons=("",))


def test_bimanual_phase_includes_normal_and_safe_paths():
    assert tuple(phase.name for phase in BimanualPhase) == (
        "APPROACH",
        "PRELOAD",
        "GRASP",
        "LIFT",
        "HOLD",
        "LOWER",
        "RELEASE",
        "DONE",
        "HOLD_SAFE",
        "LOWER_SAFE",
        "SAFE_RELEASE",
        "TERMINATED",
    )
