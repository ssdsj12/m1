"""Isolated effort-control environment for the C1a rolling WBC Teacher."""

from __future__ import annotations

from isaaclab.utils import configclass

from go2_pvcnn.tasks.m1_panda_wbc_teacher_env_cfg import (
    M1PandaWbcTeacherEnvCfg,
)


@configclass
class M1PandaWbcRollTeacherEnvCfg(M1PandaWbcTeacherEnvCfg):
    """Single M1 + Panda articulation with margin for a 20-second mission."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
