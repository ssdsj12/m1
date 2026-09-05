from __future__ import annotations

import copy
import importlib
import sys
import types

import pytest


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result

    def copy(self):
        return copy.deepcopy(self)


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
    return importlib.import_module("go2_pvcnn.assets.m1_single_panda_o6")


def test_active_order_is_16_plus_7_plus_6(contract):
    names = contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    assert contract.M1_SINGLE_PANDA_O6_ACTIVE_DOF_COUNT == 29
    assert len(names) == len(set(names)) == 29
    assert names[:16] == contract.M1_BASE_ACTIVE_JOINT_NAMES
    assert names[16:23] == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert names[23:29] == contract.RIGHT_O6_ACTIVE_JOINT_NAMES


def test_mimics_are_metadata_not_active_channels(contract):
    assert contract.RIGHT_O6_MIMIC_MAP == {
        "right_thumb_ip": ("right_thumb_cmc_pitch", 1.86, 0.0),
        "right_index_dip": ("right_index_mcp_pitch", 0.89, 0.0),
        "right_middle_dip": ("right_middle_mcp_pitch", 0.89, 0.0),
        "right_ring_dip": ("right_ring_mcp_pitch", 0.89, 0.0),
        "right_pinky_dip": ("right_pinky_mcp_pitch", 0.89, 0.0),
    }
    assert not set(contract.RIGHT_O6_MIMIC_MAP) & set(
        contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    )


def test_runtime_names_and_actuator_groups_are_isolated(contract):
    assert contract.M1_SINGLE_PANDA_O6_BASE_BODY_NAME == "BASE_LINK"
    assert contract.PANDA_WRIST_BODY_NAME == "panda_link8"
    assert contract.RIGHT_O6_PALM_BODY_NAME == "right_hand_base_link"
    assert contract.RIGHT_O6_FINGERTIP_BODY_NAMES == tuple(
        f"right_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
    )
    assert set(contract.M1_SINGLE_PANDA_O6_CFG.actuators) == {
        "legs",
        "wheels",
        "panda_shoulder",
        "panda_forearm",
        "right_o6",
    }


def test_runtime_mapping_rejects_missing_and_duplicate_names(contract):
    runtime = tuple(reversed(contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES))
    ids = contract.resolve_active_joint_ids(runtime)
    assert tuple(runtime[index] for index in ids) == contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    with pytest.raises(ValueError, match="missing"):
        contract.resolve_active_joint_ids(runtime[1:])
    with pytest.raises(ValueError, match="duplicate"):
        contract.resolve_active_joint_ids(runtime + (runtime[-1],))
