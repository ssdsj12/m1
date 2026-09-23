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
_SUCCESS_PHASES = frozenset({BimanualPhase.DONE, BimanualPhase.TERMINATED})


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
        self.goal = goal
        self.runtime = (
            BimanualRuntime(grasp_goal=goal) if runtime is None else runtime
        )
        if not callable(getattr(self.runtime, "compute", None)):
            raise TypeError("runtime must expose compute(snapshot)")
        mission = getattr(self.runtime, "mission", None)
        if goal is not None and mission is not None:
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
        fallback = (not bool(command.feasible)) or phase in _SAFE_PHASES
        latest = getattr(self.runtime, "latest_solutions", {})
        left = latest.get("left_hand") if isinstance(latest, dict) else None
        right = latest.get("right_hand") if isinstance(latest, dict) else None
        return BimanualGraspLiftStep(
            command=command,
            phase=phase,
            success=phase in _SUCCESS_PHASES,
            fallback=fallback,
            fallback_reason=reasons[0] if reasons else None,
            grasp_goal=self.goal,
            left_hand_target=None if left is None else getattr(left, "q_ref", None),
            right_hand_target=None if right is None else getattr(right, "q_ref", None),
        )


__all__ = ["BimanualGraspLiftPipeline", "BimanualGraspLiftStep"]
