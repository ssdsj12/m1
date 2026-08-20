"""Combined M1 + Panda coordinated mission environment configuration."""

from __future__ import annotations

from isaaclab.utils import configclass

from go2_pvcnn.assets.m1_panda import M1_PANDA_CFG
from go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg import M1PandaWbcRollTeacherEnvCfg


@configclass
class M1PandaCoordinatedEnvCfg(M1PandaWbcRollTeacherEnvCfg):
    """Combined 25-DOF articulation with a 16+7 coordinated action boundary."""

    # Inherited from M1PandaSmokeObservationsCfg: mount_wrench_b is the
    # unchanged six-value [Fx, Fy, Fz, Tx, Ty, Tz] base-frame signal.

    mission_target_base_pose: tuple[float, float, float] = (0.5, 0.0, 0.0)
    mission_ee_target_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0)
    mission_folded_arm_target: tuple[float, ...] = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0)
    combined_action_dim: int = 23
    mount_wrench_dim: int = 6

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M1_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 30.0


__all__ = ["M1PandaCoordinatedEnvCfg"]
