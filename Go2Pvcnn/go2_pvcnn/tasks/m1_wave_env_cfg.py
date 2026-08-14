"""Flat-ground transition from locked rolling to wheel-assisted leg control."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from go2_pvcnn.tasks.m1_roll_env_cfg import M1RollEnvCfg, M1RollRewardsCfg
import go2_pvcnn.mdp as mdp


@configclass
class M1WaveFlatRewardsCfg(M1RollRewardsCfg):
    """Keep long-horizon wheel motion from collapsing into a static equilibrium."""

    forward_velocity = RewTerm(func=mdp.forward_velocity, weight=8.0, params={"max_velocity": 0.25})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class M1WaveFlatEnvCfg(M1RollEnvCfg):
    """Preserve stable wheel motion while gradually releasing leg residuals."""

    roll_equal_wheel_actions: bool = False
    wave_fixed_forward_wheels: bool = True
    wave_leg_action_limit: float = 0.10
    wave_reference_actions: bool = True
    wave_reference_raw_amplitude: float = 0.10
    wave_reference_knee_ratio: float = 1.5
    wave_reference_frequency: float = 0.5
    wave_left_right_symmetric: bool = True
    wave_lock_abduction: bool = True
    wave_front_wheel_action: float = 0.40
    wave_rear_wheel_action: float = 0.40
    wave_wheel_residual_scale: float = 0.05
    rewards: M1WaveFlatRewardsCfg = M1WaveFlatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.03, 0.05)
        self.terminations.bad_orientation.params["limit_angle"] = 0.35
