"""Rolling priority configuration over the accepted whole-body QP."""

from __future__ import annotations

from dataclasses import dataclass

from .standing_wbc import (
    StandingWbcCfg,
    StandingWbcInput,
    StandingWbcProblem,
    StandingWbcResult,
    build_standing_wbc_problem,
    solve_standing_wbc,
)


@dataclass(frozen=True)
class RollingWbcCfg:
    """Balance-first weights for C1a base and wheel acceleration tracking."""

    balance_weight: float = 1.0e6
    base_velocity_weight: float = 1.0e5
    leg_posture_weight: float = 2.0e4
    arm_tracking_weight: float = 1.0e4
    wheel_tracking_weight: float = 5.0e4
    force_equalization_weight: float = 10.0
    tangential_force_weight: float = 10.0
    regularization: float = 1.0e-6
    qp_tolerance: float = 1.0e-5

    def standing_cfg(self) -> StandingWbcCfg:
        """Translate semantic rolling names to the shared QP weight contract."""

        return StandingWbcCfg(
            balance_weight=self.balance_weight,
            base_pose_weight=self.base_velocity_weight,
            leg_posture_weight=self.leg_posture_weight,
            arm_tracking_weight=self.arm_tracking_weight,
            wheel_stop_weight=self.wheel_tracking_weight,
            force_equalization_weight=self.force_equalization_weight,
            tangential_force_weight=self.tangential_force_weight,
            regularization=self.regularization,
            qp_tolerance=self.qp_tolerance,
        )


def build_rolling_wbc_problem(
    state: StandingWbcInput,
    cfg: RollingWbcCfg | None = None,
) -> StandingWbcProblem:
    """Build C1a with hard dynamics/contact and rolling tracking weights."""

    cfg = cfg or RollingWbcCfg()
    return build_standing_wbc_problem(state, cfg.standing_cfg())


def solve_rolling_wbc(
    state: StandingWbcInput,
    cfg: RollingWbcCfg | None = None,
) -> StandingWbcResult:
    """Solve one C1a rolling whole-body problem."""

    cfg = cfg or RollingWbcCfg()
    return solve_standing_wbc(state, cfg.standing_cfg())
