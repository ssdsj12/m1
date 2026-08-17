"""Isolated effort-control environment for the deterministic M1 + Panda WBC Teacher."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.utils import configclass

from go2_pvcnn.assets.m1_panda import (
    M1_PANDA_WBC_CFG,
    M1_PANDA_WBC_CONTROLLED_JOINT_NAMES,
)
from go2_pvcnn.tasks.m1_panda_smoke_env_cfg import (
    M1PandaSmokeEnvCfg,
    M1PandaSmokeSceneCfg,
)


@configclass
class M1PandaWbcTeacherSceneCfg(M1PandaSmokeSceneCfg):
    """Single combined articulation with wheel/base contact sensing."""

    robot = M1_PANDA_WBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class M1PandaWbcTeacherActionsCfg:
    """One ordered effort action for 12 legs, four wheels, and seven arm joints."""

    joint_effort = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


@configclass
class M1PandaWbcTeacherEnvCfg(M1PandaSmokeEnvCfg):
    """C0 stationary whole-body Teacher environment; no RSL-RL runner contract."""

    scene: M1PandaWbcTeacherSceneCfg = M1PandaWbcTeacherSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    actions: M1PandaWbcTeacherActionsCfg = M1PandaWbcTeacherActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.scene.contact_forces.update_period = self.sim.dt
