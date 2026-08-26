"""Stationary private-effort scene for Arm-MPC plus 8D residual control."""

from __future__ import annotations

from isaaclab.utils import configclass

from go2_pvcnn.tasks.m1_panda_wbc_teacher_env_cfg import M1PandaWbcTeacherEnvCfg


@configclass
class M1PandaArmMpcResidualEnvCfg(M1PandaWbcTeacherEnvCfg):
    """No-disturbance Phase 5/6 scene; wrapper owns the public RL contract."""

    private_action_dim = 23
    public_action_dim = 8
    observation_dim = 103

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 1
        self.sim.render_interval = 4
        self.episode_length_s = 20.0


__all__ = ["M1PandaArmMpcResidualEnvCfg"]
