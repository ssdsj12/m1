"""Privileged Teacher balance environments for the combined M1 + Panda asset."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from go2_pvcnn.assets import M1_FOOT_BODY_NAMES, M1_JOINT_NAMES, M1_WHEEL_JOINT_NAMES
import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_panda_smoke_env_cfg import M1PandaSmokeEnvCfg


@configclass
class M1PandaTeacherRewardsCfg:
    """Stationary balance objective shared by A0 and A1."""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-12.0,
        params={"target_height": 0.60},
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    base_xy_drift = RewTerm(
        func=mdp.base_xy_drift_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    wheel_speed = RewTerm(
        func=mdp.selected_joint_velocity_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(M1_WHEEL_JOINT_NAMES)
            )
        },
    )
    residual = RewTerm(func=mdp.teacher_residual_l2, weight=-0.02)
    residual_rate = RewTerm(func=mdp.teacher_residual_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.selected_joint_torques_l2,
        weight=-5.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_JOINT_NAMES))
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(M1_FOOT_BODY_NAMES)
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=list(M1_FOOT_BODY_NAMES)
            ),
        },
    )


@configclass
class M1PandaTeacherBaseEnvCfg(M1PandaSmokeEnvCfg):
    """Shared exact 60-observation/16-action Teacher environment."""

    rewards: M1PandaTeacherRewardsCfg = M1PandaTeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0


@configclass
class M1PandaTeacherA0EnvCfg(M1PandaTeacherBaseEnvCfg):
    """Zero-base Teacher stage with small quasi-static disturbances."""

    teacher_stage: str = "A0"
    teacher_force_limit_n: tuple[float, float, float] = (10.0, 10.0, 10.0)
    teacher_torque_limit_nm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    teacher_hold_time_s: tuple[float, float] = (1.0, 2.0)
    teacher_curriculum_start_scale: float = 0.25
    teacher_curriculum_steps: int = 50_000
    teacher_mode_probabilities: tuple[float, float, float] = (1.0, 0.0, 0.0)
    teacher_pulse_on_fraction: float = 0.20


@configclass
class M1PandaTeacherA1EnvCfg(M1PandaTeacherBaseEnvCfg):
    """Frozen-A0 Teacher stage with stronger dynamic disturbances."""

    teacher_stage: str = "A1"
    teacher_force_limit_n: tuple[float, float, float] = (20.0, 20.0, 20.0)
    teacher_torque_limit_nm: tuple[float, float, float] = (5.0, 5.0, 5.0)
    teacher_hold_time_s: tuple[float, float] = (0.25, 1.0)
    teacher_curriculum_start_scale: float = 0.25
    teacher_curriculum_steps: int = 75_000
    teacher_mode_probabilities: tuple[float, float, float] = (0.50, 0.30, 0.20)
    teacher_pulse_on_fraction: float = 0.20
