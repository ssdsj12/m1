"""Isaac Lab scene/config boundary for the deployable Student S1 task."""

from __future__ import annotations

from isaaclab.utils import configclass

from go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg import (
    M1PandaWbcRollTeacherEnvCfg,
)


@configclass
class M1PandaStudentS1EnvCfg(M1PandaWbcRollTeacherEnvCfg):
    """Flat-ground C1a asset scene with a distinct Student task identity."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 30.0
        self.decimation = 1
        self.sim.dt = 0.005
        self.sim.render_interval = 4


__all__ = ["M1PandaStudentS1EnvCfg"]
