"""Combined M1 + Panda coordinated mission environment configuration."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_pvcnn.assets import (
    M1_FOOT_BODY_NAMES,
    M1_LEG_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
)
from go2_pvcnn.assets.m1_panda import (
    M1_PANDA_BASE_BODY_NAME,
    M1_PANDA_CFG,
    M1_PANDA_MOUNT_BODY_NAME,
    M1_PANDA_WBC_CONTROLLED_JOINT_NAMES,
)
import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg import M1PandaWbcRollTeacherEnvCfg
from go2_pvcnn.tasks.m1_smoke_env_cfg import M1SmokeEventsCfg


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
class M1PandaCoordinatedActionsCfg:
    """Canonical 12+4+7 residual effort segments with physical authority."""

    leg_effort = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=list(M1_LEG_JOINT_NAMES),
        scale=5.0,
        preserve_order=True,
    )
    wheel_effort = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=list(M1_WHEEL_JOINT_NAMES),
        scale=50.0,
        preserve_order=True,
    )
    arm_effort = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=list(PANDA_ARM_JOINT_NAMES),
        scale=2.0,
        preserve_order=True,
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
    base_velocity_target = RewTerm(
        func=mdp.coordinated_base_velocity_tracking_reward,
        weight=2.0,
    )
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
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-10000.0)
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
    action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-0.1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.selected_joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES), preserve_order=True)},
    )


@configclass
class M1PandaCoordinatedEventsCfg(M1SmokeEventsCfg):
    """Deterministic defaults that the coordinated train entrypoint may randomize."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_coordinated_joints_by_offset,
        mode="reset",
        params={
            "leg_position_range": (0.0, 0.0),
            "arm_position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class M1PandaCoordinatedEnvCfg(M1PandaWbcRollTeacherEnvCfg):
    """Combined 25-DOF articulation with a 16+7 coordinated action boundary."""

    actions: M1PandaCoordinatedActionsCfg = M1PandaCoordinatedActionsCfg()
    observations: M1PandaCoordinatedObservationsCfg = M1PandaCoordinatedObservationsCfg()
    rewards: M1PandaCoordinatedRewardsCfg = M1PandaCoordinatedRewardsCfg()
    events: M1PandaCoordinatedEventsCfg = M1PandaCoordinatedEventsCfg()

    # mount_wrench_b remains the unchanged six-value
    # [Fx, Fy, Fz, Tx, Ty, Tz] base-frame signal.

    mission_target_base_pose: tuple[float, float, float] = (0.5, 0.0, 0.0)
    mission_ee_target_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
    mission_ee_target_offset_b: tuple[float, float, float] = (0.05, 0.0, 0.0)
    mission_folded_arm_target: tuple[float, ...] = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0)
    mission_arrival_position_tolerance_m: float = 0.08
    mission_arrival_yaw_tolerance_rad: float = 0.10
    mission_balance_target_height_m: float = 0.6115
    mission_base_linear_speed_limit_mps: float = 0.10
    mission_base_yaw_rate_limit_rad_s: float = 0.50
    mission_wheel_radius_m: float = 0.095
    mission_wheel_damping_nm_per_rad_s: float = 30.0
    mission_wheel_action_scale_nm: float = 50.0
    combined_action_dim: int = 23
    mount_wrench_dim: int = 6

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M1_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 30.0


def configure_coordinated_training_domain_randomization(cfg, enabled: bool) -> None:
    """Set the exact approved train ranges or restore deterministic defaults."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if enabled:
        pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.05, 0.05),
        }
        root_velocity_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
        }
        leg_position_range = (-0.02, 0.02)
        arm_position_range = (-0.03, 0.03)
        joint_velocity_range = (-0.05, 0.05)
        friction_range = (0.8, 1.2)
    else:
        pose_range = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        root_velocity_range = dict(pose_range)
        leg_position_range = (0.0, 0.0)
        arm_position_range = (0.0, 0.0)
        joint_velocity_range = (0.0, 0.0)
        friction_range = (1.0, 1.0)

    cfg.events.reset_base.params = {
        "pose_range": pose_range,
        "velocity_range": root_velocity_range,
    }
    cfg.events.reset_robot_joints.params = {
        "leg_position_range": leg_position_range,
        "arm_position_range": arm_position_range,
        "velocity_range": joint_velocity_range,
        "asset_cfg": cfg.events.reset_robot_joints.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        ),
    }
    cfg.events.physics_material.params = {
        "asset_cfg": cfg.events.physics_material.params.get(
            "asset_cfg", SceneEntityCfg("robot", body_names=".*")
        ),
        "static_friction_range": friction_range,
        "dynamic_friction_range": friction_range,
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    }


__all__ = [
    "COORDINATED_POLICY_OBSERVATION_DIM",
    "COORDINATED_POLICY_OBSERVATION_WIDTHS",
    "M1PandaCoordinatedEventsCfg",
    "M1PandaCoordinatedEnvCfg",
    "PANDA_ARM_JOINT_NAMES",
    "configure_coordinated_training_domain_randomization",
]
