"""PPO-facing environment for the dual Panda/O6 reduced-wire controller.

The environment deliberately keeps the policy boundary small: PPO emits a
15-dimensional normalized latent and the existing bimanual wrapper remains
responsible for snapshot construction, DeformX/reduced-Cosserat stepping,
MPC safety projection, contact gating, and atomic fallback.  This module does
not duplicate any of those controllers; it only adapts their diagnostics to
the PPO transition contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math
from typing import Any

import torch

try:  # Gymnasium is present in IsaacLab runs but keep CPU contracts importable.
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - minimal CPU-only installations
    spaces = None

from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_contracts import (
    WIRE_PPO_ACTION_DIM,
    WireLatentCommand,
    WirePpoObservationSpec,
    decode_wire_latent,
)
from go2_pvcnn.tasks.m1_wire_ppo_curriculum import (
    WirePpoPhase,
    WirePpoRewardCfg,
    WirePpoStage,
    WirePpoState,
    approval_episode,
    compute_wire_ppo_reward,
    infer_wire_ppo_phase,
)


def _flat(value: Any) -> list[float]:
    """Convert a tensor/dataclass/numeric container into finite scalars."""

    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return []
        return [float(item) for item in value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)]
    if is_dataclass(value):
        result: list[float] = []
        for field_value in asdict(value).values():
            result.extend(_flat(field_value))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flat(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flat(item))
        return result
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _finite_clip(values: list[float], size: int) -> torch.Tensor:
    output = torch.zeros(size, dtype=torch.float32)
    if values:
        values = [float(item) if math.isfinite(float(item)) else 0.0 for item in values]
        count = min(size, len(values))
        output[:count] = torch.tensor(values[:count], dtype=torch.float32)
    # Observations are normalized at the policy boundary.  The controller
    # diagnostics are metric-valued, so clip only here and never in control.
    return output.clamp_(-1.0, 1.0)


def _diagnostics(wrapper: object) -> dict[str, Any]:
    value = getattr(wrapper, "wire_acceptance_diagnostics", None)
    if value is None:
        return {}
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        try:
            value = dict(vars(value))
        except TypeError:
            return {}
    return deepcopy(value)


def _forces(diag: dict[str, Any], side: str) -> tuple[float, ...]:
    raw = diag.get(f"{side}_fingertip_forces_b", ())
    result: list[float] = []
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    for force in raw if isinstance(raw, (tuple, list)) else ():
        if isinstance(force, (tuple, list)):
            result.append(float(torch.linalg.vector_norm(torch.as_tensor(force, dtype=torch.float64))))
        elif isinstance(force, (int, float)):
            result.append(abs(float(force)))
    return tuple(item for item in result if math.isfinite(item) and item >= 0.0)


def _contact_count(diag: dict[str, Any], side: str) -> int:
    value = diag.get(f"{side}_valid_contacts", None)
    if value is None:
        mask = diag.get(f"{side}_contact_mask", ())
        if isinstance(mask, torch.Tensor):
            value = int(mask.to(dtype=torch.bool).sum().item())
        elif isinstance(mask, (tuple, list)):
            value = sum(bool(item) for item in mask)
        else:
            value = 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _scalar(diag: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(diag.get(name, default) or default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


class M1WirePpoEnv:
    """Single-lane Gym-like adapter around ``M1DualPandaO6BimanualWrapper``.

    ``wrapper`` may be the Isaac wrapper or a deterministic test double that
    exposes ``reset`` and ``step_latent``.  This keeps CPU contract tests free
    of Isaac imports while the production path uses the same wrapper and
    therefore the same contact/safety gates.
    """

    def __init__(
        self,
        wrapper: object,
        *,
        stage: WirePpoStage | str = WirePpoStage.APPROVAL,
        reward_cfg: WirePpoRewardCfg | None = None,
        max_episode_steps: int = 4000,
        device: torch.device | str = "cpu",
    ) -> None:
        if isinstance(stage, str):
            try:
                stage = WirePpoStage(stage.lower())
            except ValueError as exc:
                raise ValueError("stage must be 'approval' or 'action'") from exc
        if not isinstance(stage, WirePpoStage):
            raise TypeError("stage must be WirePpoStage or str")
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        self.wrapper = wrapper
        self._stage = stage
        self.reward_cfg = reward_cfg or WirePpoRewardCfg()
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.observation_spec = WirePpoObservationSpec()
        self.action_dim = WIRE_PPO_ACTION_DIM
        self.action_space = (
            spaces.Box(-1.0, 1.0, (WIRE_PPO_ACTION_DIM,), dtype="float32")
            if spaces is not None else (WIRE_PPO_ACTION_DIM, (-1.0, 1.0))
        )
        self.observation_space = (
            spaces.Box(-1.0, 1.0, (self.observation_spec.observation_dim,), dtype="float32")
            if spaces is not None else (self.observation_spec.observation_dim, (-1.0, 1.0))
        )
        self._previous_state: WirePpoState | None = None
        self._previous_action = torch.zeros(WIRE_PPO_ACTION_DIM, dtype=torch.float32)
        self._observation = torch.zeros(self.observation_spec.observation_dim, dtype=torch.float32)
        self._episode_step = 0
        self._initial_wire_height = 0.0
        self._diagnostic_history: list[dict[str, Any]] = []
        self._reset_done = False

    @property
    def stage(self) -> WirePpoStage:
        return self._stage

    @property
    def observation_dim(self) -> int:
        return self.observation_spec.observation_dim

    def _wire_state(self, snapshot: object, diag: dict[str, Any]) -> WirePpoState:
        left_forces = _forces(diag, "left")
        right_forces = _forces(diag, "right")
        height = _scalar(diag, "wire_center_height_m", self._initial_wire_height)
        lift = max(0.0, height - self._initial_wire_height)
        approach = max(
            0.0,
            _scalar(diag, "left_palm_error_m", _scalar(diag, "left_pinch_error_m")),
            _scalar(diag, "right_palm_error_m", _scalar(diag, "right_pinch_error_m")),
        )
        feasibility = diag.get("feasibility", {})
        safety_failure = bool(
            diag.get("safety_failure", False)
            or diag.get("rejection_reason") in {"safety_rejected", "terminated", "non_finite_contact_observation"}
            or (isinstance(feasibility, dict) and feasibility.get("safety") is False)
        )
        return WirePpoState(
            approach_error=approach,
            contacts_left=_contact_count(diag, "left"),
            contacts_right=_contact_count(diag, "right"),
            contact_forces_left=left_forces,
            contact_forces_right=right_forces,
            lift_height=lift,
            hold_time=max(0.0, _scalar(diag, "hold_elapsed_s", _scalar(diag, "hold_time"))),
            slip=max(_scalar(diag, "left_slip_m"), _scalar(diag, "right_slip_m")),
            safety_failure=safety_failure,
            wire_centerline_error=max(0.0, _scalar(diag, "wire_centerline_error")),
            tangent_error=max(0.0, _scalar(diag, "tangent_error")),
            deformation_error=max(0.0, _scalar(diag, "deformation_error")),
            synchronization_error=max(0.0, _scalar(diag, "synchronization_error")),
            force_balance_error=max(0.0, _scalar(diag, "force_balance_error")),
            latent_delta_norm=float(torch.linalg.vector_norm(self._previous_action).item()),
        )

    def _build_observation(self, snapshot: object, diag: dict[str, Any]) -> torch.Tensor:
        # Keep group boundaries explicit to make observation changes reviewable.
        wire_values = _flat(diag)
        wire_values.extend(_flat(getattr(self.wrapper, "runtime", None) and getattr(self.wrapper.runtime, "latest_wire_diagnostics", None)))
        wire_values.extend(_flat(getattr(snapshot, "box", None)))
        wire = _finite_clip(wire_values, self.observation_spec.wire_state_dim)

        bimanual_values = []
        for side in ("left_arm", "right_arm", "left_hand", "right_hand"):
            bimanual_values.extend(_flat(getattr(snapshot, side, None)))
        bimanual = _finite_clip(bimanual_values, self.observation_spec.bimanual_state_dim)

        arm = _finite_clip(_flat(getattr(snapshot, "left_arm", None)) + _flat(getattr(snapshot, "right_arm", None)), self.observation_spec.arm_state_dim)
        phase_name = str(diag.get("mission_phase", diag.get("acceptance_phase", "approach"))).lower()
        phase_values = [1.0 if phase_name == phase else 0.0 for phase in ("approach", "contact", "grasp", "lift", "hold", "failed")]
        phase_values += [float(_contact_count(diag, side)) / 5.0 for side in ("left", "right")]
        phase_values += [_scalar(diag, "lift_elapsed_s"), _scalar(diag, "hold_elapsed_s")]
        phase_values += [1.0 if bool(diag.get("success", False)) else 0.0, 1.0 if bool(diag.get("request_reclamp", False)) else 0.0]
        safety = diag.get("feasibility", {})
        phase_values += [1.0 if not (isinstance(safety, dict) and safety.get(name) is False) else 0.0 for name in ("safety", "object", "arm", "left_hand", "right_hand", "wbc")]
        phase = _finite_clip(phase_values, self.observation_spec.safety_phase_dim)
        result = torch.cat((wire, bimanual, arm, phase, self._previous_action), dim=0)
        self.observation_spec.validate(result)
        return result.to(device=self.device)

    def _info(self, state: WirePpoState, diag: dict[str, Any], *, terminated: bool, truncated: bool) -> dict[str, Any]:
        phase = infer_wire_ppo_phase(state, self.reward_cfg)
        info = {
            "stage": self.stage.value,
            "phase": phase.value,
            "episode_step": self._episode_step,
            "success": bool(approval_episode(state, self.reward_cfg) or diag.get("success", False)),
            "failure_reason": diag.get("rejection_reason") or diag.get("contact_failure_reason"),
            "terminated": terminated,
            "truncated": truncated,
            "contacts_left": state.contacts_left,
            "contacts_right": state.contacts_right,
            "lift_height": state.lift_height,
            "hold_time": state.hold_time,
            "slip": state.slip,
            "diagnostics": deepcopy(diag),
        }
        self._diagnostic_history.append(deepcopy(info))
        return info

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")
        reset = getattr(self.wrapper, "reset", None)
        if not callable(reset):
            raise TypeError("wrapper must expose reset()")
        try:
            snapshot = reset(seed=0 if seed is None else int(seed))
        except TypeError:
            snapshot = reset() if seed is None else reset(int(seed))
        diag = _diagnostics(self.wrapper)
        self._initial_wire_height = _scalar(diag, "wire_center_height_m")
        self._previous_action.zero_()
        self._episode_step = 0
        self._diagnostic_history.clear()
        self._previous_state = self._wire_state(snapshot, diag)
        self._observation = self._build_observation(snapshot, diag)
        self._reset_done = True
        info = self._info(self._previous_state, diag, terminated=False, truncated=False)
        return self._observation.clone(), info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        if not self._reset_done or self._previous_state is None:
            raise RuntimeError("reset() must be called before step()")
        # Decode is intentionally the first operation: invalid actions cannot
        # reach the wrapper or mutate any physical/controller state.
        command: WireLatentCommand = decode_wire_latent(action, device=self.device, dtype=torch.float32)
        step_latent = getattr(self.wrapper, "step_latent", None)
        if callable(step_latent):
            step_latent(command.as_vector)
        else:
            step = getattr(self.wrapper, "step", None)
            if not callable(step):
                raise TypeError("wrapper must expose step_latent() or step()")
            try:
                step(command.as_vector)
            except TypeError:
                # Existing Isaac wrapper has a no-argument step; it still runs
                # the approved safety/MPC path, while retaining the latent in
                # this adapter's transition and diagnostics.
                step()
        snapshot = getattr(self.wrapper, "last_snapshot", None)
        adapter = getattr(self.wrapper, "adapter", None)
        live_snapshot = getattr(adapter, "snapshot", None)
        if callable(live_snapshot):
            snapshot = live_snapshot()
        diag = _diagnostics(self.wrapper)
        current = self._wire_state(snapshot, diag)
        breakdown = compute_wire_ppo_reward(self._previous_state, current, self.stage, self.reward_cfg)
        self._episode_step += 1
        success = bool(diag.get("success", False) or approval_episode(current, self.reward_cfg))
        terminal_phase = str(diag.get("acceptance_phase", "")).lower() in {"terminated", "failed"}
        terminated = bool(success or terminal_phase)
        truncated = bool(not terminated and self._episode_step >= self.max_episode_steps)
        self._previous_action = command.as_vector.detach().to(device="cpu", dtype=torch.float32).clone()
        self._previous_state = current
        self._observation = self._build_observation(snapshot, diag)
        info = self._info(current, diag, terminated=terminated, truncated=truncated)
        info["reward_terms"] = breakdown.action_terms() if self.stage is WirePpoStage.ACTION else breakdown.approval_terms()
        info["episode_diagnostics"] = deepcopy(self._diagnostic_history)
        return self._observation.clone(), float(breakdown.total), terminated, truncated, info

    def close(self) -> None:
        close = getattr(self.wrapper, "close", None)
        if callable(close):
            close()


__all__ = ["M1WirePpoEnv"]
