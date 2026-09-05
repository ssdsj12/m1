"""Asset and active-control contract for M1 with one Panda and right O6."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from isaaclab.actuators import ImplicitActuatorCfg

from go2_pvcnn.assets import M1_CFG


M1_SINGLE_PANDA_O6_USD_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets/m1_single_panda_o6/m1_single_panda_o6.usd"
)
M1_SINGLE_PANDA_O6_BASE_BODY_NAME = "BASE_LINK"
PANDA_WRIST_BODY_NAME = "panda_link8"
RIGHT_O6_PALM_BODY_NAME = "right_hand_base_link"
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
PANDA_ACTIVE_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
RIGHT_O6_ACTIVE_JOINT_NAMES = tuple(
    f"right_{name}"
    for name in (
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )
)
RIGHT_O6_MIMIC_MAP = {
    "right_thumb_ip": ("right_thumb_cmc_pitch", 1.86, 0.0),
    "right_index_dip": ("right_index_mcp_pitch", 0.89, 0.0),
    "right_middle_dip": ("right_middle_mcp_pitch", 0.89, 0.0),
    "right_ring_dip": ("right_ring_mcp_pitch", 0.89, 0.0),
    "right_pinky_dip": ("right_pinky_mcp_pitch", 0.89, 0.0),
}
M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES = (
    *M1_BASE_ACTIVE_JOINT_NAMES,
    *PANDA_ACTIVE_JOINT_NAMES,
    *RIGHT_O6_ACTIVE_JOINT_NAMES,
)
M1_SINGLE_PANDA_O6_ACTIVE_DOF_COUNT = 29


def resolve_active_joint_ids(runtime_joint_names: Sequence[str]) -> tuple[int, ...]:
    name_to_id: dict[str, int] = {}
    duplicates: set[str] = set()
    for joint_id, name in enumerate(runtime_joint_names):
        if name in name_to_id:
            duplicates.add(name)
        else:
            name_to_id[name] = joint_id
    if duplicates:
        raise ValueError(f"duplicate runtime joint names: {sorted(duplicates)}")
    missing = [
        name
        for name in M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
        if name not in name_to_id
    ]
    if missing:
        raise ValueError(f"missing active runtime joint names: {missing}")
    return tuple(
        name_to_id[name] for name in M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    )


M1_SINGLE_PANDA_O6_CFG = M1_CFG.copy()
M1_SINGLE_PANDA_O6_CFG.spawn = M1_SINGLE_PANDA_O6_CFG.spawn.replace(
    usd_path=M1_SINGLE_PANDA_O6_USD_PATH
)
M1_SINGLE_PANDA_O6_CFG.init_state.joint_pos.update(
    {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.650,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "right_(thumb|index|middle|ring|pinky)_.*": 0.1,
    }
)
M1_SINGLE_PANDA_O6_CFG.actuators.update(
    {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
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
