"""Trainable M1 rolling environment.

This is the first long-running M1 locomotion stage: keep the leg posture fixed
and train the four wheel velocity actions to track forward commands.
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
class M1RollCommandsCfg(M1SmokeCommandsCfg):
    """Slow forward commands for the first stable rolling curriculum."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.02, 0.04),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.02, 0.04),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class M1RollActionsCfg(M1SmokeActionsCfg):
    """Lock leg position actions; train wheel velocity actions."""

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
        scale=1.0,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class M1RollObservationsCfg(M1SmokeObservationsCfg):
    """Low-dimensional proprioception for rolling control."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1RollRewardsCfg(M1SmokeRewardsCfg):
    """Rolling rewards: move forward without tipping or scraping the body."""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.5)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.03)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.20)},
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-8.0, params={"target_height": 0.55})
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)
    lateral_velocity = RewTerm(func=mdp.track_lin_vel_y_l2, weight=-2.0, params={"command_name": "base_velocity"})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-8)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    wheel_action_match = RewTerm(
        func=mdp.paired_action_mismatch,
        weight=-0.25,
        params={
            "front_action_ids": (12, 13),
            "rear_action_ids": (14, 15),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=M1_BASE_BODY_NAME),
        },
    )


@configclass
class M1RollEnvCfg(M1SmokeEnvCfg):
    """M1 rolling task for the first autonomous locomotion stage."""

    observations: M1RollObservationsCfg = M1RollObservationsCfg()
    actions: M1RollActionsCfg = M1RollActionsCfg()
    commands: M1RollCommandsCfg = M1RollCommandsCfg()
    rewards: M1RollRewardsCfg = M1RollRewardsCfg()
    roll_equal_wheel_actions: bool = True
    roll_sync_actual_wheel_velocity: bool = True
    roll_wheel_sync_gain: float = 0.50
    roll_wheel_sync_integral_gain: float = 1.0
    roll_wheel_sync_integral_limit: float = 0.50
    roll_wheel_sync_max_correction: float = 0.50
    roll_wheel_equalize_gain: float = 2.0
    roll_wheel_equalize_max_correction: float = 0.50

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.terminations.base_contact = None
