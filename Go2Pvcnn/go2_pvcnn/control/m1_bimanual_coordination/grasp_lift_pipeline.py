"""Small closed-loop boundary for a catalog grasp/lift mission.

The pipeline is intentionally an adapter around :class:`BimanualRuntime`.
Planning remains in the existing object/arm/O6/WBC layers; this module only
exposes one step result suitable for a headless probe or a task wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import BimanualCommand, BimanualPhase, BimanualSnapshot
from .grasp_goal import BimanualGraspGoal
from .runtime import BimanualRuntime


_SAFE_PHASES = frozenset(
    {
        BimanualPhase.HOLD_SAFE,
        BimanualPhase.LOWER_SAFE,
        BimanualPhase.SAFE_RELEASE,
        BimanualPhase.TERMINATED,
    }
)
@dataclass(frozen=True)
class BimanualGraspLiftStep:
    """One accepted or rejected control cycle at the pipeline boundary."""

    command: BimanualCommand
    phase: BimanualPhase
    success: bool
    fallback: bool
    fallback_reason: str | None
    grasp_goal: BimanualGraspGoal | None
    left_hand_target: Any | None
    right_hand_target: Any | None


class BimanualGraspLiftPipeline:
    """Drive one catalog goal through approach, clamp, and lift phases.

    ``goal=None`` is deliberately supported so callers that still use the
    historical ``Box`` target can adopt the same boundary without changing
    their runtime configuration.
    """

    def __init__(
        self,
        goal: BimanualGraspGoal | None = None,
        *,
        runtime: BimanualRuntime | None = None,
    ) -> None:
        if goal is not None and not isinstance(goal, BimanualGraspGoal):
            raise TypeError("goal must be BimanualGraspGoal or None")
        self.runtime = (
            BimanualRuntime(grasp_goal=goal) if runtime is None else runtime
        )
        if not callable(getattr(self.runtime, "compute", None)):
            raise TypeError("runtime must expose compute(snapshot)")

        mission = getattr(self.runtime, "mission", None)
        runtime_goals = []
        for candidate in (
            getattr(mission, "grasp_goal", None),
            getattr(self.runtime, "_grasp_goal", None),
            getattr(self.runtime, "latest_grasp_goal", None),
        ):
            if candidate is not None:
                if not isinstance(candidate, BimanualGraspGoal):
                    raise TypeError("runtime grasp_goal must be BimanualGraspGoal or None")
                if all(candidate != existing for existing in runtime_goals):
                    runtime_goals.append(candidate)
        if len(runtime_goals) > 1:
            raise ValueError("runtime exposes conflicting grasp_goal values")
        existing_goal = runtime_goals[0] if runtime_goals else None
        if goal is not None and existing_goal is not None and goal != existing_goal:
            raise ValueError("conflicting grasp_goal between pipeline and runtime")
        self.goal = goal if goal is not None else existing_goal
        if self.goal is not None:
            binder = getattr(self.runtime, "bind_grasp_goal", None)
            if callable(binder):
                binder(self.goal)
            elif goal is not None and existing_goal is None and mission is not None:
                setter = getattr(mission, "set_grasp_goal", None)
                if callable(setter):
                    setter(goal)

    @property
    def phase(self) -> BimanualPhase:
        phase = getattr(self.runtime, "phase", None)
        if phase is None:
            phase = getattr(getattr(self.runtime, "mission", None), "phase", None)
        if not isinstance(phase, BimanualPhase):
            raise TypeError("runtime must expose a BimanualPhase")
        return phase

    def reset(self) -> None:
        reset = getattr(self.runtime, "reset", None)
        if not callable(reset):
            raise TypeError("runtime must expose reset()")
        reset()

    def step(self, snapshot: BimanualSnapshot) -> BimanualGraspLiftStep:
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        command = self.runtime.compute(snapshot)
        if not isinstance(command, BimanualCommand):
            # Keep test/probe adapters useful while still checking the fields
            # consumed by this boundary.
            if not isinstance(getattr(command, "feasible", None), bool):
                raise TypeError("runtime.compute() must return BimanualCommand-like data")
        phase = self.phase
        reasons = tuple(getattr(command, "fallback_reasons", ()))
        reason = reasons[0] if reasons else None
        mission_state = getattr(self.runtime, "latest_mission_state", None)
        if reason is None and mission_state is not None:
            reason = getattr(mission_state, "fallback_reason", None)
        if reason is None:
            reason = getattr(self.runtime, "fallback_reason", None)
        fallback = (not bool(command.feasible)) or phase in _SAFE_PHASES or reason is not None
        latest = getattr(self.runtime, "latest_solutions", {})
        left = latest.get("left_hand") if isinstance(latest, dict) else None
        right = latest.get("right_hand") if isinstance(latest, dict) else None
        step_goal = self.goal
        if step_goal is None:
            step_goal = getattr(self.runtime, "latest_grasp_goal", None)
        if step_goal is None:
            step_goal = getattr(getattr(self.runtime, "mission", None), "grasp_goal", None)
        return BimanualGraspLiftStep(
            command=command,
            phase=phase,
            success=phase is BimanualPhase.DONE and not fallback,
            fallback=fallback,
            fallback_reason=reason,
            grasp_goal=step_goal,
            left_hand_target=None if left is None else getattr(left, "q_ref", None),
            right_hand_target=None if right is None else getattr(right, "q_ref", None),
        )


__all__ = ["BimanualGraspLiftPipeline", "BimanualGraspLiftStep"]
