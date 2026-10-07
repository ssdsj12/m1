"""Bounded dual-palm trajectories derived from a DeformX centerline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real

import torch

from .contracts import BimanualSnapshot
from .deformx_adapter import DualDloStepResult
from .dlo_wire_goal import DloCenterlineState, WireLineContactGoal, build_wire_station_contact_goal
from .object_mpc import ObjectMpcSolution


@dataclass(frozen=True)
class DloWireTrajectoryPlan:
    """Wire contact geometry paired with an MPC-compatible palm trajectory."""

    object_solution: ObjectMpcSolution
    contact_goal: WireLineContactGoal


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _palm_target(contact_targets, approach_normal, standoff_m: float) -> torch.Tensor:
    positions = torch.tensor(
        [target.position for target in contact_targets], dtype=torch.float64
    )
    inward = torch.tensor(approach_normal, dtype=torch.float64)
    inward /= torch.linalg.vector_norm(inward).clamp_min(1.0e-12)
    return positions.mean(dim=0) - standoff_m * inward


def _bounded_position_horizon(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    horizon_steps: int,
    max_step_m: float,
) -> torch.Tensor:
    delta = target - current
    distance = torch.linalg.vector_norm(delta)
    if float(distance) <= 1.0e-12:
        return current.repeat(horizon_steps, 1)
    direction = delta / distance
    travel = torch.clamp(
        max_step_m * torch.arange(1, horizon_steps + 1, dtype=torch.float64),
        max=float(distance),
    )
    return current + travel[:, None] * direction


class DloWireTrajectoryPlanner:
    """Turn the latest Cosserat centerline into safe dual-palm approach paths.

    This adapter changes only the palm references consumed by the two Arm
    MPCs.  The object trajectory, wrench allocation, feasibility diagnostics,
    and WBC acceptance boundary remain owned by their existing controllers.
    """

    def __init__(
        self,
        *,
        dt: float = 0.04,
        palm_speed_m_s: float = 0.04,
        palm_standoff_m: float = 0.08,
        contact_clearance_m: float = 0.003,
    ) -> None:
        self.dt = _positive_finite("dt", dt)
        self.palm_speed_m_s = _positive_finite("palm_speed_m_s", palm_speed_m_s)
        self.palm_standoff_m = _positive_finite("palm_standoff_m", palm_standoff_m)
        self.contact_clearance_m = _positive_finite(
            "contact_clearance_m", contact_clearance_m
        )

    def plan(
        self,
        snapshot: BimanualSnapshot,
        object_solution: ObjectMpcSolution,
        dlo_step: DualDloStepResult,
    ) -> DloWireTrajectoryPlan:
        if not isinstance(dlo_step, DualDloStepResult):
            raise TypeError("dlo_step must be DualDloStepResult")
        return self.plan_state(snapshot, object_solution, dlo_step.state)

    def plan_state(
        self,
        snapshot: BimanualSnapshot,
        object_solution: ObjectMpcSolution,
        state: DloCenterlineState,
    ) -> DloWireTrajectoryPlan:
        """Plan from scene geometry before bind or accepted Cosserat geometry after."""
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if not isinstance(object_solution, ObjectMpcSolution):
            raise TypeError("object_solution must be ObjectMpcSolution")
        goal = build_wire_station_contact_goal(
            state,
            clearance_m=self.contact_clearance_m,
        )
        horizon = object_solution.left_palm_pose.shape[0]
        max_step = self.dt * self.palm_speed_m_s
        left_position = _bounded_position_horizon(
            snapshot.left_arm.palm_pose_b[:3],
            _palm_target(goal.left_contact_targets, goal.left_palm_approach_normal, self.palm_standoff_m),
            horizon_steps=horizon,
            max_step_m=max_step,
        )
        right_position = _bounded_position_horizon(
            snapshot.right_arm.palm_pose_b[:3],
            _palm_target(goal.right_contact_targets, goal.right_palm_approach_normal, self.palm_standoff_m),
            horizon_steps=horizon,
            max_step_m=max_step,
        )
        left_pose = torch.cat(
            (
                left_position,
                snapshot.left_arm.palm_pose_b[3:].repeat(horizon, 1),
            ),
            dim=1,
        )
        right_pose = torch.cat(
            (
                right_position,
                snapshot.right_arm.palm_pose_b[3:].repeat(horizon, 1),
            ),
            dim=1,
        )
        return DloWireTrajectoryPlan(
            object_solution=replace(
                object_solution,
                left_palm_pose=left_pose,
                right_palm_pose=right_pose,
            ),
            contact_goal=goal,
        )


__all__ = ["DloWireTrajectoryPlan", "DloWireTrajectoryPlanner"]
