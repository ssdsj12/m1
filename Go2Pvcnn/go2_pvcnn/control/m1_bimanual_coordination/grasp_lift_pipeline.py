"""Small closed-loop boundary for a catalog grasp/lift mission.

The pipeline is intentionally an adapter around :class:`BimanualRuntime`.
Planning remains in the existing object/arm/O6/WBC layers; this module only
exposes one step result suitable for a headless probe or a task wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contracts import BimanualCommand, BimanualPhase, BimanualSnapshot
from .constraints import WireGateConfig, evaluate_wire_gate
from .diagnostics import WirePipelineDiagnostics
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
    diagnostics: WirePipelineDiagnostics | None = None

    @property
    def terminal_reason(self) -> str | None:
        """Machine-readable terminal/fallback reason for probe consumers."""

        return self.fallback_reason


@dataclass(frozen=True)
class WirePipelineConfig(WireGateConfig):
    """Pipeline-facing alias retaining the wire gate thresholds."""

    contact_dwell_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.contact_dwell_steps, bool) or not isinstance(self.contact_dwell_steps, int) or self.contact_dwell_steps <= 0:
            raise ValueError("contact_dwell_steps must be a positive integer")


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
        wire_config: WirePipelineConfig | None = None,
    ) -> None:
        if goal is not None and not isinstance(goal, BimanualGraspGoal):
            raise TypeError("goal must be BimanualGraspGoal or None")
        self.runtime = (
            BimanualRuntime(grasp_goal=goal) if runtime is None else runtime
            )
        if wire_config is None:
            self.wire_config = WirePipelineConfig()
        elif isinstance(wire_config, WirePipelineConfig):
            self.wire_config = wire_config
        elif isinstance(wire_config, WireGateConfig):
            self.wire_config = WirePipelineConfig(
                min_contact_force_n=wire_config.min_contact_force_n,
                min_contacts_per_hand=wire_config.min_contacts_per_hand,
                max_slip_m=wire_config.max_slip_m,
            )
        else:
            raise TypeError("wire_config must be WirePipelineConfig")
        self._wire_left_latch = [False] * 5
        self._wire_right_latch = [False] * 5
        self._last_wire_command: BimanualCommand | None = None
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
        self._wire_left_latch = [False] * 5
        self._wire_right_latch = [False] * 5
        self._last_wire_command = None

    @staticmethod
    def _coerce_wire_diagnostics(value: Any) -> WirePipelineDiagnostics | None:
        if value is None:
            return None
        if isinstance(value, WirePipelineDiagnostics):
            return value
        if isinstance(value, dict):
            return WirePipelineDiagnostics(**value)
        # Runtime adapters may expose a diagnostics dataclass with matching
        # fields.  Ignore it unless all required fields are present.
        required = (
            "phase", "bilateral_contact", "left_contact_mask", "right_contact_mask",
            "left_clamp_force_n", "right_clamp_force_n", "force_closure_margin",
            "slip_m", "wire_feasible", "arm_recoverability", "command_accepted",
        )
        if all(hasattr(value, name) for name in required):
            fields = {name: getattr(value, name) for name in required}
            for name in ("terminal_reason", "wire_failure_reason", "left_latched", "right_latched"):
                if hasattr(value, name):
                    fields[name] = getattr(value, name)
            return WirePipelineDiagnostics(**fields)
        return None

    def _wire_observation(self, snapshot: BimanualSnapshot, command: BimanualCommand) -> WirePipelineDiagnostics | None:
        runtime_value = None
        for name in ("wire_diagnostics", "latest_wire_pipeline_diagnostics", "latest_reduced_wire_diagnostics"):
            runtime_value = getattr(self.runtime, name, None)
            if runtime_value is not None:
                break
        observation = self._coerce_wire_diagnostics(runtime_value)
        if observation is None:
            # Native runtime adapters publish reduced prediction/object/arm
            # diagnostics rather than the convenience dataclass.  Normalize
            # that boundary here so all callers use the same gates.
            update = None
            for name in ("latest_reduced_wire_update", "latest_wire_update", "reduced_wire_update"):
                update = getattr(self.runtime, name, None)
                if update is not None:
                    break
            latest = getattr(self.runtime, "latest_solutions", {})
            object_solution = latest.get("object") if isinstance(latest, dict) else None
            object_diag = getattr(object_solution, "diagnostics", None)
            if update is not None or object_diag is not None:
                phase = getattr(self.runtime, "phase", BimanualPhase.APPROACH)
                phase_name = phase.name if isinstance(phase, BimanualPhase) else str(phase).upper()
                left_mask = tuple(bool(v) for v in snapshot.left_hand.contact_mask.tolist())
                right_mask = tuple(bool(v) for v in snapshot.right_hand.contact_mask.tolist())
                left_force = float(snapshot.left_hand.fingertip_forces_b[:, 2].clamp_min(0.0).sum().item())
                right_force = float(snapshot.right_hand.fingertip_forces_b[:, 2].clamp_min(0.0).sum().item())
                prediction = getattr(update, "prediction", None)
                feasible = bool(getattr(update, "failure_reason", None) is None)
                slip = 0.0
                if prediction is not None and getattr(prediction, "endpoint_slip_m", None) is not None:
                    try:
                        slip = float(prediction.endpoint_slip_m.max())
                    except (TypeError, ValueError, AttributeError):
                        feasible = False
                if object_diag is not None:
                    feasible = feasible and bool(getattr(object_diag, "wire_friction_feasible", True) is not False)
                    feasible = feasible and bool(getattr(object_diag, "wire_lift_feasible", True) is not False)
                    if getattr(object_diag, "wire_slip_max_m", None) is not None:
                        slip = max(slip, float(object_diag.wire_slip_max_m))
                arm_solution = latest.get("arm") if isinstance(latest, dict) else None
                arm_ok = bool(getattr(arm_solution, "both_feasible", True))
                reason = getattr(update, "failure_reason", None)
                observation = WirePipelineDiagnostics(
                    phase=phase_name,
                    bilateral_contact=bool(left_mask and right_mask and any(left_mask) and any(right_mask)),
                    left_contact_mask=left_mask,
                    right_contact_mask=right_mask,
                    left_clamp_force_n=left_force,
                    right_clamp_force_n=right_force,
                    force_closure_margin=float(getattr(object_diag, "force_closure_margin", 1.0) or 0.0),
                    slip_m=slip,
                    wire_feasible=feasible,
                    arm_recoverability=arm_ok,
                    command_accepted=bool(command.feasible),
                    wire_failure_reason=reason,
                )
        if observation is None:
            return None
        left = [old or new for old, new in zip(self._wire_left_latch, observation.left_contact_mask)]
        right = [old or new for old, new in zip(self._wire_right_latch, observation.right_contact_mask)]
        # A latch is monotonic only through GRASP; before that, contact is an
        # observation and after a lift loss the command is rejected atomically.
        if observation.phase in {"GRASP", "LIFT", "HOLD"}:
            self._wire_left_latch, self._wire_right_latch = left, right
        return replace(
            observation,
            left_contact_mask=tuple(observation.left_contact_mask),
            right_contact_mask=tuple(observation.right_contact_mask),
            left_latched=sum(self._wire_left_latch) >= self.wire_config.min_contacts_per_hand,
            right_latched=sum(self._wire_right_latch) >= self.wire_config.min_contacts_per_hand,
            command_accepted=bool(command.feasible and observation.command_accepted),
        )

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
        wire_diagnostics = self._wire_observation(snapshot, command)
        wire_reason = None
        if wire_diagnostics is not None:
            gate = evaluate_wire_gate(wire_diagnostics, self.wire_config)
            wire_reason = gate.reason
            if not gate.accepted:
                wire_diagnostics = replace(wire_diagnostics, terminal_reason=wire_reason)
                # Do not publish a newly rejected effort.  Keep the last safe
                # command, while preserving the runtime's immutable command
                # type and timestamp contract.
                effort = (
                    self._last_wire_command.effort
                    if self._last_wire_command is not None
                    else command.effort
                )
                if isinstance(command, BimanualCommand):
                    command = replace(command, effort=effort, feasible=False,
                                      fallback_reasons=(wire_reason,))
                else:
                    command = BimanualCommand(
                        timestamp_ns=snapshot.timestamp_ns,
                        effort=effort,
                        feasible=False,
                        fallback_reasons=(wire_reason,),
                    )
            else:
                self._last_wire_command = command
            try:
                wire_phase = BimanualPhase[wire_diagnostics.phase]
                if wire_phase in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD,
                                  BimanualPhase.GRASP, BimanualPhase.LIFT, BimanualPhase.HOLD}:
                    phase = wire_phase
            except KeyError:
                pass
        reasons = tuple(getattr(command, "fallback_reasons", ()))
        reason = wire_reason or (reasons[0] if reasons else None)
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
            diagnostics=wire_diagnostics,
        )


__all__ = ["BimanualGraspLiftPipeline", "BimanualGraspLiftStep"]
