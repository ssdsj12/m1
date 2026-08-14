from pathlib import Path
import sys
import types


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        values = dict(self.__dict__)
        values.update(kwargs)
        return self.__class__(**values)


def _install_isaaclab_asset_stubs(monkeypatch):
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

    modules = {
        "isaaclab": isaaclab,
        "isaaclab.sim": sim,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets_pkg,
        "isaaclab.assets.articulation": articulation,
        "isaaclab.utils": utils_pkg,
        "isaaclab.utils.assets": utils_assets,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_m1_asset_cfg_uses_floating_usd_and_16_controlled_joints(monkeypatch):
    _install_isaaclab_asset_stubs(monkeypatch)

    from go2_pvcnn.assets import (
        M1_BASE_BODY_NAME,
        M1_CFG,
        M1_FOOT_BODY_NAMES,
        M1_JOINT_NAMES,
        M1_LEG_JOINT_NAMES,
        M1_ROLLING_MODE,
        M1_USD_PATH,
        M1_WAVE_MODE,
        M1_WHEEL_JOINT_NAMES,
    )

    assert Path(M1_USD_PATH).is_file()
    assert M1_USD_PATH.endswith("ZJ_V3_URDF_V1_0_floating.usda")
    usd_text = Path(M1_USD_PATH).read_text()
    assert 'defaultPrim = "ZJ_V3_URDF_V1_0"' in usd_text
    assert 'over "root_joint" (' in usd_text
    assert "active = false" in usd_text
    assert 'over "FAR_FOOT_JOINT"' in usd_text
    assert 'over "FBL_FOOT_JOINT"' in usd_text
    assert 'over "RAR_FOOT_JOINT"' in usd_text
    assert 'over "RBL_FOOT_JOINT"' in usd_text
    assert "physics:lowerLimit = -1000000" in usd_text
    assert "physics:upperLimit = 1000000" in usd_text
    assert "drive:angular:physics:stiffness = 0" in usd_text
    assert "drive:angular:physics:maxForce = 500" in usd_text
    assert usd_text.count('def Cylinder "wheel_collision"') == 4
    assert "double radius = 0.0959" in usd_text
    assert 'uniform token axis = "Y"' in usd_text
    assert M1_BASE_BODY_NAME == "BASE_LINK"
    assert M1_FOOT_BODY_NAMES == (
        "FAR_FOOT_LINK",
        "FBL_FOOT_LINK",
        "RAR_FOOT_LINK",
        "RBL_FOOT_LINK",
    )
    assert M1_JOINT_NAMES == (
        "FAR_ABAD_JOINT",
        "FAR_HIP_JOINT",
        "FAR_KNEE_JOINT",
        "FAR_FOOT_JOINT",
        "FBL_ABAD_JOINT",
        "FBL_HIP_JOINT",
        "FBL_KNEE_JOINT",
        "FBL_FOOT_JOINT",
        "RAR_ABAD_JOINT",
        "RAR_HIP_JOINT",
        "RAR_KNEE_JOINT",
        "RAR_FOOT_JOINT",
        "RBL_ABAD_JOINT",
        "RBL_HIP_JOINT",
        "RBL_KNEE_JOINT",
        "RBL_FOOT_JOINT",
    )
    assert M1_CFG.spawn.usd_path == M1_USD_PATH
    assert M1_CFG.spawn.activate_contact_sensors is True
    assert M1_CFG.spawn.articulation_props.solver_velocity_iteration_count == 2
    assert M1_CFG.init_state.pos == (0.0, 0.0, 0.62)
    assert M1_CFG.init_state.joint_pos["FAR_HIP_JOINT"] == 0.30
    assert M1_CFG.init_state.joint_pos["FBL_HIP_JOINT"] == 0.30
    assert M1_CFG.init_state.joint_pos["FAR_KNEE_JOINT"] == -0.60
    assert M1_CFG.init_state.joint_pos["FBL_KNEE_JOINT"] == -0.60
    assert M1_CFG.init_state.joint_pos["RAR_HIP_JOINT"] == -0.30
    assert M1_CFG.init_state.joint_pos["RBL_HIP_JOINT"] == -0.30
    assert M1_CFG.init_state.joint_pos["RAR_KNEE_JOINT"] == 0.60
    assert M1_CFG.init_state.joint_pos["RBL_KNEE_JOINT"] == 0.60
    assert M1_CFG.actuators["legs"].joint_names_expr == list(M1_LEG_JOINT_NAMES)
    assert M1_CFG.actuators["legs"].stiffness == 120.0
    assert M1_CFG.actuators["legs"].damping == 5.5
    assert M1_CFG.actuators["wheels"].joint_names_expr == list(M1_WHEEL_JOINT_NAMES)
    assert M1_CFG.actuators["wheels"].stiffness == 0.0
    assert M1_CFG.actuators["wheels"].damping == 30.0
    assert M1_CFG.actuators["wheels"].effort_limit_sim == 200.0
    assert M1_CFG.actuators["wheels"].velocity_limit_sim == 20.0
    assert M1_LEG_JOINT_NAMES == tuple(name for name in M1_JOINT_NAMES if "FOOT_JOINT" not in name)
    assert M1_WHEEL_JOINT_NAMES == tuple(name for name in M1_JOINT_NAMES if "FOOT_JOINT" in name)
    assert len(M1_LEG_JOINT_NAMES) == 12
    assert len(M1_WHEEL_JOINT_NAMES) == 4
    assert M1_ROLLING_MODE == "rolling"
    assert M1_WAVE_MODE == "wave"
