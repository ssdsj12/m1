"""Trainable M1 walking environment.

This task keeps the smoke task available for open-loop wheel checks, but gives
RSL-RL a locomotion objective: track a forward velocity command with the 12 leg
position actions while the four wheel joints are held at zero velocity.
"""

from __future__ import annotations

import math

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_pvcnn.assets import M1_BASE_BODY_NAME, M1_FOOT_BODY_NAMES, M1_LEG_JOINT_NAMES, M1_WHEEL_JOINT_NAMES
from go2_pvcnn.tasks.m1_smoke_env_cfg import (
    M1SmokeActionsCfg,
    M1SmokeCommandsCfg,
    M1SmokeEnvCfg,
    M1SmokeObservationsCfg,
    M1SmokeRewardsCfg,
)
import go2_pvcnn.mdp as mdp


@configclass
class M1WalkCommandsCfg(M1SmokeCommandsCfg):
    """Forward walking command for early M1 training."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.5,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.08),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.08),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class M1WalkActionsCfg(M1SmokeActionsCfg):
    """Train the legs; hold wheel joints at zero velocity."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(M1_LEG_JOINT_NAMES),
        scale=0.05,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(M1_WHEEL_JOINT_NAMES),
        scale=0.0,
        use_default_offset=True,
        clip={".*": (0.0, 0.0)},
    )


@configclass
class M1WalkObservationsCfg(M1SmokeObservationsCfg):
    """Proprioception plus the commanded forward velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1WalkRewardsCfg(M1SmokeRewardsCfg):
    """Reward terms for stable forward walking."""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=2.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.20)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-12.0, params={"target_height": 0.60})
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-8)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    front_rear_joint_speed = RewTerm(
        func=mdp.paired_joint_speed_mismatch,
        weight=-0.08,
        params={
            "front_asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FAR_ABAD_JOINT",
                    "FAR_HIP_JOINT",
                    "FAR_KNEE_JOINT",
                    "FBL_ABAD_JOINT",
                    "FBL_HIP_JOINT",
                    "FBL_KNEE_JOINT",
                ],
            ),
            "rear_asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "RAR_ABAD_JOINT",
                    "RAR_HIP_JOINT",
                    "RAR_KNEE_JOINT",
                    "RBL_ABAD_JOINT",
                    "RBL_HIP_JOINT",
                    "RBL_KNEE_JOINT",
                ],
            ),
        },
    )
    front_rear_action = RewTerm(
        func=mdp.paired_action_mismatch,
        weight=-1.0,
        params={
            "front_action_ids": (0, 1, 2, 3, 4, 5),
            "rear_action_ids": (6, 7, 8, 9, 10, 11),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_LEG_JOINT_NAMES)),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.2,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES))},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=M1_BASE_BODY_NAME),
        },
    )


@configclass
class M1WalkEnvCfg(M1SmokeEnvCfg):
    """M1 walking task for RSL-RL training."""

    observations: M1WalkObservationsCfg = M1WalkObservationsCfg()
    actions: M1WalkActionsCfg = M1WalkActionsCfg()
    commands: M1WalkCommandsCfg = M1WalkCommandsCfg()
    rewards: M1WalkRewardsCfg = M1WalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.terminations.base_contact = None
