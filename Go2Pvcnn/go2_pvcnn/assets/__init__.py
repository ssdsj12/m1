"""Configuration for Unitree Go2 robot for PVCNN project.

This module provides robot configurations for the Go2 quadruped robot.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

M1_USD_PATH = (
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_usd/"
    "ZJ_V3_URDF_V1_0_floating.usda"
)
M1_BASE_BODY_NAME = "BASE_LINK"
M1_FOOT_BODY_NAMES = (
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
)
M1_JOINT_NAMES = (
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
M1_LEG_JOINT_NAMES = tuple(joint_name for joint_name in M1_JOINT_NAMES if "FOOT_JOINT" not in joint_name)
M1_WHEEL_JOINT_NAMES = tuple(joint_name for joint_name in M1_JOINT_NAMES if "FOOT_JOINT" in joint_name)
M1_ROLLING_MODE = "rolling"
M1_WAVE_MODE = "wave"

M1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=M1_USD_PATH,
        activate_contact_sensors=True,
        visible=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={
            ".*ABAD_JOINT": 0.0,
            "FAR_HIP_JOINT": 0.30,
            "FBL_HIP_JOINT": 0.30,
            "FAR_KNEE_JOINT": -0.60,
            "FBL_KNEE_JOINT": -0.60,
            "RAR_HIP_JOINT": -0.30,
            "RBL_HIP_JOINT": -0.30,
            "RAR_KNEE_JOINT": 0.60,
            "RBL_KNEE_JOINT": 0.60,
            ".*FOOT_JOINT": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=list(M1_LEG_JOINT_NAMES),
            effort_limit=150.0,
            saturation_effort=150.0,
            velocity_limit=20.0,
            stiffness=120.0,
            damping=5.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(M1_WHEEL_JOINT_NAMES),
            effort_limit_sim=200.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=30.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Genisomai M1 with a simple DC motor actuator model."""

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        visible=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.7,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 with DC motor actuator model."""


UNITREE_GO2_CFG_SIMPLE = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.7,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
        ),
    },
)
"""Configuration of Unitree Go2 with implicit (simple) actuator model."""
