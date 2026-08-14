"""Isolated smoke environment for the combined M1 and Panda articulation."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_pvcnn.assets import M1_JOINT_NAMES, M1_LEG_JOINT_NAMES, M1_WHEEL_JOINT_NAMES
from go2_pvcnn.assets.m1_panda import M1_PANDA_BASE_BODY_NAME, M1_PANDA_CFG, M1_PANDA_MOUNT_BODY_NAME
import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_smoke_env_cfg import M1SmokeEnvCfg, M1SmokeSceneCfg


@configclass
class M1PandaSmokeSceneCfg(M1SmokeSceneCfg):
    """Flat smoke scene using the combined articulation."""

    robot = M1_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class M1PandaSmokeActionsCfg:
    """Exactly 12 M1 leg-position plus four M1 wheel-velocity actions."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(M1_LEG_JOINT_NAMES),
        scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)},
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=list(M1_WHEEL_JOINT_NAMES),
        scale=8.0, use_default_offset=True, clip={".*": (-8.0, 8.0)},
    )


@configclass
class M1PandaSmokeObservationsCfg:
    """M1-only proprioception; Panda joints stay outside policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_JOINT_NAMES))},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_JOINT_NAMES))},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        actions = ObsTerm(func=isaac_mdp.last_action)
        mount_wrench_b = ObsTerm(
            func=mdp.m1_panda_mount_wrench_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=M1_PANDA_MOUNT_BODY_NAME),
                "mount_body_name": M1_PANDA_MOUNT_BODY_NAME,
                "base_body_name": M1_PANDA_BASE_BODY_NAME,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1PandaSmokeEnvCfg(M1SmokeEnvCfg):
    """No-planner smoke cfg for the combined 25-DOF articulation."""

    scene: M1PandaSmokeSceneCfg = M1PandaSmokeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: M1PandaSmokeObservationsCfg = M1PandaSmokeObservationsCfg()
    actions: M1PandaSmokeActionsCfg = M1PandaSmokeActionsCfg()
