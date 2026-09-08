"""Asset and active-control contract for M1 with dual Panda and dual O6 hands."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from isaaclab.actuators import ImplicitActuatorCfg

from go2_pvcnn.assets import M1_CFG


M1_DUAL_PANDA_O6_USD_PATH = str(
    Path(__file__).resolve().parents[2] / "assets/m1_dual_panda_o6/m1_dual_panda_o6.usd"
)

M1_DUAL_PANDA_O6_BASE_BODY_NAME = "BASE_LINK"
M1_DUAL_PANDA_O6_PLATFORM_BODY_NAME = "platform"
M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME = "dual_arm_platform_yaw_joint"
M1_DUAL_PANDA_O6_PLATFORM_YAW_LIMIT_RAD = (-1.5707963267948966, 1.5707963267948966)

LEFT_PANDA_WRIST_BODY_NAME = "left_panda_link8"
RIGHT_PANDA_WRIST_BODY_NAME = "right_panda_link8"
LEFT_O6_PALM_BODY_NAME = "left_hand_base_link"
RIGHT_O6_PALM_BODY_NAME = "right_hand_base_link"
LEFT_O6_FINGERTIP_BODY_NAMES = tuple(
    f"left_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
)
RIGHT_O6_FINGERTIP_BODY_NAMES = tuple(
    f"right_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
)

M1_BASE_ACTIVE_JOINT_NAMES = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
    "FAR_FOOT_JOINT",
    "FBL_FOOT_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_FOOT_JOINT",
)
LEFT_PANDA_ACTIVE_JOINT_NAMES = tuple(f"left_panda_joint{i}" for i in range(1, 8))
RIGHT_PANDA_ACTIVE_JOINT_NAMES = tuple(f"right_panda_joint{i}" for i in range(1, 8))

_O6_ACTIVE_JOINT_SUFFIXES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
LEFT_O6_ACTIVE_JOINT_NAMES = tuple(f"left_{name}" for name in _O6_ACTIVE_JOINT_SUFFIXES)
RIGHT_O6_ACTIVE_JOINT_NAMES = tuple(f"right_{name}" for name in _O6_ACTIVE_JOINT_SUFFIXES)

# Values are copied from the supplied left/right O6 URDFs.  Mimic joints are
# physical simulation DOFs, but never independent optimizer/control channels.
O6_MIMIC_MAP = {
    "thumb_ip": ("thumb_cmc_pitch", 1.86, 0.0),
    "index_dip": ("index_mcp_pitch", 0.89, 0.0),
    "middle_dip": ("middle_mcp_pitch", 0.89, 0.0),
    "ring_dip": ("ring_mcp_pitch", 0.89, 0.0),
    "pinky_dip": ("pinky_mcp_pitch", 0.89, 0.0),
}

M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES = (
    *M1_BASE_ACTIVE_JOINT_NAMES,
    M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME,
    *LEFT_PANDA_ACTIVE_JOINT_NAMES,
    *RIGHT_PANDA_ACTIVE_JOINT_NAMES,
    *LEFT_O6_ACTIVE_JOINT_NAMES,
    *RIGHT_O6_ACTIVE_JOINT_NAMES,
)
M1_DUAL_PANDA_O6_ACTIVE_DOF_COUNT = 43
assert len(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES) == M1_DUAL_PANDA_O6_ACTIVE_DOF_COUNT


def resolve_active_joint_ids(runtime_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Map canonical active order to an unambiguous Isaac runtime joint order."""

    name_to_id: dict[str, int] = {}
    duplicates: set[str] = set()
    for joint_id, name in enumerate(runtime_joint_names):
        if name in name_to_id:
            duplicates.add(name)
        else:
            name_to_id[name] = joint_id
    if duplicates:
        raise ValueError(f"duplicate runtime joint names: {sorted(duplicates)}")
    missing = [name for name in M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES if name not in name_to_id]
    if missing:
        raise ValueError(f"missing active runtime joint names: {missing}")
    return tuple(name_to_id[name] for name in M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES)


M1_DUAL_PANDA_O6_CFG = M1_CFG.copy()
M1_DUAL_PANDA_O6_CFG.spawn = M1_DUAL_PANDA_O6_CFG.spawn.replace(
    usd_path=M1_DUAL_PANDA_O6_USD_PATH
)
M1_DUAL_PANDA_O6_CFG.init_state.joint_pos.update(
    {
        M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME: 0.0,
        "left_panda_joint1": 0.0,
        "left_panda_joint2": -0.569,
        "left_panda_joint3": 0.0,
        "left_panda_joint4": -2.650,
        "left_panda_joint5": 0.0,
        "left_panda_joint6": 3.037,
        "left_panda_joint7": 0.741,
        "right_panda_joint1": 0.0,
        "right_panda_joint2": -0.569,
        "right_panda_joint3": 0.0,
        "right_panda_joint4": -2.650,
        "right_panda_joint5": 0.0,
        "right_panda_joint6": 3.037,
        "right_panda_joint7": 0.741,
        "left_(thumb|index|middle|ring|pinky)_.*": 0.25,
        "right_(thumb|index|middle|ring|pinky)_.*": 0.25,
    }
)

# O6's supplied URDF declares 1 rad/s for every joint.  The smaller simulated
# effort and moderate position gains below are controller limits; the vendor's
# uniform URDF effort=100 is not treated as a validated hardware safety limit.
M1_DUAL_PANDA_O6_CFG.actuators.update(
    {
        "platform": ImplicitActuatorCfg(
            joint_names_expr=[M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME],
            effort_limit_sim=50.0,
            velocity_limit_sim=0.25,
            stiffness=120.0,
            damping=20.0,
        ),
        "left_panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "left_panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "right_panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "right_panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "left_o6": ImplicitActuatorCfg(
            joint_names_expr=list(LEFT_O6_ACTIVE_JOINT_NAMES),
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=20.0,
            damping=1.0,
        ),
        "right_o6": ImplicitActuatorCfg(
            joint_names_expr=list(RIGHT_O6_ACTIVE_JOINT_NAMES),
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=20.0,
            damping=1.0,
        ),
    }
)
