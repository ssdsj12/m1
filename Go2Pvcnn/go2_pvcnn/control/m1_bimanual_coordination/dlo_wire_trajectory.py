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
from .reduced_wire_runtime import ReducedWireUpdate


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
        track_pinch_center: bool = False,
    ) -> None:
        self.dt = _positive_finite("dt", dt)
        self.palm_speed_m_s = _positive_finite("palm_speed_m_s", palm_speed_m_s)
        self.palm_standoff_m = _positive_finite("palm_standoff_m", palm_standoff_m)
        self.contact_clearance_m = _positive_finite(
            "contact_clearance_m", contact_clearance_m
        )
        if not isinstance(track_pinch_center, bool):
            raise TypeError("track_pinch_center must be bool")
        self.track_pinch_center = track_pinch_center

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

        def palm_target(side: str) -> torch.Tensor:
            targets = getattr(goal, f"{side}_contact_targets")
            if self.track_pinch_center:
                # Orientation is held below, so translate the observed pinch
                # center onto its station instead of assuming a palm-local
                # fingertip offset aligned with the wire's normal.
                palm = getattr(snapshot, f"{side}_arm").palm_pose_b[:3]
                pinch = getattr(snapshot, f"{side}_hand").fingertip_positions_b[:2].mean(0)
                station = torch.tensor([target.position for target in targets], dtype=torch.float64).mean(0)
                return palm + station - pinch
            return _palm_target(targets, getattr(goal, f"{side}_palm_approach_normal"), self.palm_standoff_m)

        left_position = _bounded_position_horizon(
            snapshot.left_arm.palm_pose_b[:3],
            palm_target("left"),
            horizon_steps=horizon,
            max_step_m=max_step,
        )
        right_position = _bounded_position_horizon(
            snapshot.right_arm.palm_pose_b[:3],
            palm_target("right"),
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

    def plan_prediction(
        self,
        snapshot: BimanualSnapshot,
        object_solution: ObjectMpcSolution,
        update: ReducedWireUpdate,
    ) -> DloWireTrajectoryPlan:
        """Track every predicted endpoint, or hold measured palms on rejection.

        Existing geometry-only entry points retain their contracts. This opt-in
        path also supplies the synchronized wire lift reference to object MPC.
        """
        if not isinstance(update, ReducedWireUpdate):
            raise TypeError("update must be ReducedWireUpdate")
        horizon = object_solution.left_palm_pose.shape[0]
        if update.endpoint_targets.shape != (horizon, 2, 3) or update.lift_velocity.shape != (horizon,):
            raise ValueError("wire prediction must match object MPC horizon")
        geometry = DloCenterlineState(update.state.positions, update.state.radius_m)
        goal = build_wire_station_contact_goal(geometry, clearance_m=self.contact_clearance_m)
        paths = {}
        for index, side in enumerate(("left", "right")):
            pose = getattr(snapshot, f"{side}_arm").palm_pose_b
            if update.failure_reason is not None:
                paths[f"{side}_palm_pose"] = pose.repeat(horizon, 1)
                continue
            targets = update.endpoint_targets[:, index].to(pose)
            if self.track_pinch_center:
                pinch = getattr(snapshot, f"{side}_hand").fingertip_positions_b[:2].mean(0)
                targets = targets + pose[:3] - pinch
            else:
                normal = pose.new_tensor(getattr(goal, f"{side}_palm_approach_normal"))
                targets = targets - self.palm_standoff_m * normal
            current = pose[:3]
            positions = []
            for target in targets:
                delta = target - current
                distance = torch.linalg.vector_norm(delta).clamp_min(1.0e-12)
                current = current + delta * torch.clamp(self.dt * self.palm_speed_m_s / distance, max=1.)
                positions.append(current)
            paths[f"{side}_palm_pose"] = torch.cat(
                (torch.stack(positions), pose[3:].repeat(horizon, 1)), dim=1)
        twist = object_solution.box_twist.clone()
        twist[:, 2] = (0. if update.failure_reason is not None else
                       update.lift_velocity.to(twist).clamp(0., self.palm_speed_m_s))
        for side in ("left", "right"):
            initial_height = getattr(snapshot, f"{side}_arm").palm_pose_b[2:3]
            heights = paths[f"{side}_palm_pose"][:, 2]
            available_lift = torch.diff(torch.cat((initial_height, heights))) / self.dt
            twist[:, 2] = torch.minimum(twist[:, 2], available_lift.to(twist).clamp_min(0.))
        box_pose = object_solution.box_pose.clone()
        box_pose[:, 2] = snapshot.box.pose_b[2] + self.dt * torch.cumsum(twist[:, 2], dim=0)
        return DloWireTrajectoryPlan(
            object_solution=replace(object_solution, **paths, box_twist=twist, box_pose=box_pose),
            contact_goal=goal,
        )


__all__ = ["DloWireTrajectoryPlan", "DloWireTrajectoryPlanner"]
