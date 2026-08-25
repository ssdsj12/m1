"""M1 locomotion while carrying the dynamically simulated, PD-folded Panda."""

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
FOLDED_LOAD_POLICY_OBSERVATION_DIM = 103
FOLDED_LOAD_POLICY_OBSERVATION_WIDTHS = (
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
class M1PandaFoldedLoadActionsCfg:
    """Canonical 12-leg, 4-wheel, 7-arm effort boundary."""

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
class M1PandaFoldedLoadObservationsCfg:
    """Checkpoint-compatible 103-value locomotion observation."""

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
        base_target_error_b = ObsTerm(func=mdp.folded_load_compat_base_error_b)
        ee_pose_error_b = ObsTerm(func=mdp.folded_load_compat_ee_error_b)
        desired_twist_b = ObsTerm(func=mdp.folded_load_desired_twist_b)
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
class M1PandaFoldedLoadRewardsCfg:
    """Balance-first body-X and yaw-rate locomotion objective."""

    track_vx = RewTerm(
        func=mdp.folded_load_track_vx,
        weight=2.0,
    )
    track_wz = RewTerm(
        func=mdp.folded_load_track_wz,
        weight=1.0,
    )
    lateral_velocity = RewTerm(
        func=mdp.folded_load_lateral_velocity_l2,
        weight=-0.5,
    )
    alive = RewTerm(
        func=isaac_mdp.is_alive,
        weight=1.0,
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-12.0,
        params={"target_height": 0.6115},
    )
    base_linear_velocity = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )
    base_angular_velocity = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
    active_action_l2 = RewTerm(
        func=mdp.folded_load_active_action_l2,
        weight=-0.02,
    )
    active_action_rate = RewTerm(
        func=mdp.folded_load_active_action_rate_l2,
        weight=-0.01,
    )
    joint_torques = RewTerm(
        func=mdp.selected_joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES), preserve_order=True)},
    )
    termination_penalty = RewTerm(
        func=isaac_mdp.is_terminated,
        weight=-10000.0,
    )


@configclass
class M1PandaFoldedLoadEventsCfg(M1SmokeEventsCfg):
    """Deterministic L0/L1 reset and material defaults."""

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
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "velocity_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
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
class M1PandaFoldedLoadEnvCfg(M1PandaWbcRollTeacherEnvCfg):
    """Single dynamic articulation with inactive Panda policy coordinates."""

    actions: M1PandaFoldedLoadActionsCfg = M1PandaFoldedLoadActionsCfg()
    observations: M1PandaFoldedLoadObservationsCfg = M1PandaFoldedLoadObservationsCfg()
    rewards: M1PandaFoldedLoadRewardsCfg = M1PandaFoldedLoadRewardsCfg()
    events: M1PandaFoldedLoadEventsCfg = M1PandaFoldedLoadEventsCfg()
    combined_action_dim: int = 23

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M1_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = 1
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.episode_length_s = 30.0


__all__ = [
    "FOLDED_LOAD_POLICY_OBSERVATION_DIM",
    "FOLDED_LOAD_POLICY_OBSERVATION_WIDTHS",
    "M1PandaFoldedLoadEnvCfg",
]
