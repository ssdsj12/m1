"""Combined M1 + Panda coordinated mission environment configuration."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_pvcnn.assets import M1_FOOT_BODY_NAMES
from go2_pvcnn.assets.m1_panda import (
    M1_PANDA_BASE_BODY_NAME,
    M1_PANDA_CFG,
    M1_PANDA_MOUNT_BODY_NAME,
    M1_PANDA_WBC_CONTROLLED_JOINT_NAMES,
)
import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg import M1PandaWbcRollTeacherEnvCfg


PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
PANDA_HAND_BODY_NAME = "panda_hand"
COORDINATED_POLICY_OBSERVATION_DIM = 103
COORDINATED_POLICY_OBSERVATION_WIDTHS = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("controlled_joint_pos", 23),
    ("controlled_joint_vel", 23),
    ("base_target_error_b", 3),
    ("ee_pose_error_b", 6),
    ("desired_twist_b", 6),
    ("wheel_contact", 4),
    ("mount_wrench_b", 6),
    ("previous_action", 23),
)


@configclass
class M1PandaCoordinatedObservationsCfg:
    """Exact 103-value whole-body Teacher policy observation."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=0.0, n_max=0.0))
        controlled_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES), preserve_order=True)},
        )
        controlled_joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES), preserve_order=True)},
        )
        base_target_error_b = ObsTerm(func=mdp.coordinated_base_target_error_b)
        ee_pose_error_b = ObsTerm(
            func=mdp.coordinated_ee_pose_error_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[M1_PANDA_BASE_BODY_NAME, PANDA_HAND_BODY_NAME], preserve_order=True)},
        )
        desired_twist_b = ObsTerm(func=mdp.coordinated_desired_twist_b)
        wheel_contact = ObsTerm(
            func=mdp.coordinated_wheel_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES), preserve_order=True)},
        )
        mount_wrench_b = ObsTerm(
            func=mdp.m1_panda_mount_wrench_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=M1_PANDA_MOUNT_BODY_NAME),
                "mount_body_name": M1_PANDA_MOUNT_BODY_NAME,
                "base_body_name": M1_PANDA_BASE_BODY_NAME,
            },
        )
        previous_action = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1PandaCoordinatedRewardsCfg:
    """Two-stage base-navigation then end-effector objective."""

    base_target = RewTerm(func=mdp.coordinated_base_tracking_reward, weight=3.0)
    folded_arm = RewTerm(
        func=mdp.coordinated_folded_arm_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(PANDA_ARM_JOINT_NAMES), preserve_order=True)},
    )
    ee_tracking = RewTerm(
        func=mdp.coordinated_ee_tracking_reward,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[M1_PANDA_BASE_BODY_NAME, PANDA_HAND_BODY_NAME], preserve_order=True)},
    )
    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-200.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-12.0,
        params={"target_height": 0.6115},
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.selected_joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES), preserve_order=True)},
    )


@configclass
class M1PandaCoordinatedEnvCfg(M1PandaWbcRollTeacherEnvCfg):
    """Combined 25-DOF articulation with a 16+7 coordinated action boundary."""

    observations: M1PandaCoordinatedObservationsCfg = M1PandaCoordinatedObservationsCfg()
    rewards: M1PandaCoordinatedRewardsCfg = M1PandaCoordinatedRewardsCfg()

    # mount_wrench_b remains the unchanged six-value
    # [Fx, Fy, Fz, Tx, Ty, Tz] base-frame signal.

    mission_target_base_pose: tuple[float, float, float] = (0.5, 0.0, 0.0)
    mission_ee_target_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
    mission_ee_target_offset_b: tuple[float, float, float] = (0.05, 0.0, 0.0)
    mission_folded_arm_target: tuple[float, ...] = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0)
    mission_arrival_position_tolerance_m: float = 0.08
    mission_arrival_yaw_tolerance_rad: float = 0.10
    combined_action_dim: int = 23
    mount_wrench_dim: int = 6

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M1_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 30.0


__all__ = [
    "COORDINATED_POLICY_OBSERVATION_DIM",
    "COORDINATED_POLICY_OBSERVATION_WIDTHS",
    "M1PandaCoordinatedEnvCfg",
]
