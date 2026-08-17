"""Configuration for the combined M1 and Panda articulation."""

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg

from go2_pvcnn.assets import M1_CFG


M1_PANDA_USD_PATH = str(Path(__file__).resolve().parents[2] / "assets/m1_panda/m1_panda.usd")
M1_PANDA_BASE_BODY_NAME = "BASE_LINK"
M1_PANDA_MOUNT_BODY_NAME = "panda_link0"
M1_PANDA_DOF_COUNT = 25

M1_PANDA_CFG = M1_CFG.copy()
M1_PANDA_CFG.spawn = M1_PANDA_CFG.spawn.replace(usd_path=M1_PANDA_USD_PATH)
M1_PANDA_CFG.init_state.joint_pos.update(
    {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
)
M1_PANDA_CFG.actuators.update(
    {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
    }
)

# Explicit order used by the C0 whole-body controller.  The two gripper joints
# deliberately remain outside this effort-control contract.
M1_PANDA_WBC_CONTROLLED_JOINT_NAMES = (
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
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

# Keep the legacy combined asset untouched.  C0 writes torques directly to the
# 23 controlled joints, while the fingers retain their existing position hold.
M1_PANDA_WBC_CFG = M1_PANDA_CFG.copy()
M1_PANDA_WBC_CFG.actuators = {
    "legs": M1_PANDA_CFG.actuators["legs"].replace(stiffness=0.0, damping=0.0),
    "wheels": M1_PANDA_CFG.actuators["wheels"].replace(stiffness=0.0, damping=0.0),
    "panda_shoulder": M1_PANDA_CFG.actuators["panda_shoulder"].replace(stiffness=0.0, damping=0.0),
    "panda_forearm": M1_PANDA_CFG.actuators["panda_forearm"].replace(stiffness=0.0, damping=0.0),
    "panda_hand": M1_PANDA_CFG.actuators["panda_hand"].copy(),
}
