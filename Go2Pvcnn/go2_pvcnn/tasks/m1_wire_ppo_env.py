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
from dataclasses import asdict, is_dataclass, replace
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


def _tensor(value: Any) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if not torch.isfinite(result).all():
        raise ValueError("wire PPO measurements must be finite")
    return result


def _resample(points: Any, count: int = 5) -> torch.Tensor:
    points = _tensor(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("wire centerline must have shape (N>=2, 3)")
    return torch.nn.functional.interpolate(points.T[None], size=count, mode="linear", align_corners=True)[0].T


def _tangents(points: torch.Tensor) -> torch.Tensor:
    segments = points[1:] - points[:-1]
    segments = segments / torch.linalg.vector_norm(segments, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat((segments, segments[-1:]))


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
    raw = _tensor(diag.get(f"{side}_fingertip_forces_b", torch.zeros(5, 3)))
    mask = torch.as_tensor(diag.get(f"{side}_contact_mask", [False] * 5), dtype=torch.bool)
    if raw.shape != (5, 3) or mask.shape != (5,):
        raise ValueError("contact forces/mask must have shapes (5,3)/(5,)")
    return tuple(torch.linalg.vector_norm(raw[mask], dim=1).tolist())


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
    value = float(diag.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


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
        target_centerline: torch.Tensor | None = None,
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
        self._configured_target = None if target_centerline is None else _resample(target_centerline)
        self._target_centerline = self._configured_target
        # Fixed v1 layout: wire=5 nodes position/tangent/velocity/target position/
        # target tangent (75), segment stretch/curvature summary (5).
        # Each hand=pose6, twist6, q6, qd6, fingertips15, force summaries3 (42).
        # Each arm=q7, qd7 (14). Phase=6 one-hot, contact masks10, slips2,
        # minimum joint-limit margins2. Previous action=15.
        self.observation_layout = {
            "wire": slice(0, 80), "left_hand": slice(80, 122),
            "right_hand": slice(122, 164), "left_arm": slice(164, 178),
            "right_arm": slice(178, 192), "phase": slice(192, 212),
            "previous_action": slice(212, 227),
        }

    @property
    def stage(self) -> WirePpoStage:
        return self._stage

    @property
    def observation_dim(self) -> int:
        return self.observation_spec.observation_dim

    def _geometry(self, snapshot):
        runtime = getattr(self.wrapper, "runtime", None)
        update = getattr(runtime, "latest_reduced_wire_update", None)
        state = getattr(update, "state", None)
        if state is None:
            state = getattr(getattr(runtime, "latest_dlo_step", None), "state", None)
        if state is None:
            provider = getattr(self.wrapper, "_wire_geometry_provider", None)
            if callable(provider):
                state = provider(snapshot).centerline
        if state is None:
            # The selected object's measured pose provides a reset fallback;
            # no acceptance diagnostic has to exist before the first step.
            center = _tensor(snapshot.box.pose_b[:3])
            points = torch.stack((center + center.new_tensor([0., -.1, 0.]),
                                  center + center.new_tensor([0., .1, 0.])))
            velocities = _tensor(snapshot.box.twist_b[:3]).repeat(2, 1)
        else:
            points = _tensor(state.positions)
            velocities = _tensor(getattr(state, "endpoint_velocities", torch.zeros(2, 3)))
        return _resample(points), _resample(velocities)

    def _wire_state(self, snapshot: object, diag: dict[str, Any]) -> WirePpoState:
        left_forces = _forces(diag, "left")
        right_forces = _forces(diag, "right")
        points, velocities = self._geometry(snapshot)
        height = float(points[:, 2].mean())
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
        target = self._target_centerline if self._target_centerline is not None else points
        segments, target_segments = points[1:] - points[:-1], target[1:] - target[:-1]
        return WirePpoState(
            approach_error=approach,
            contacts_left=len(left_forces),
            contacts_right=len(right_forces),
            contact_forces_left=left_forces,
            contact_forces_right=right_forces,
            lift_height=lift,
            hold_time=max(0.0, _scalar(diag, "hold_elapsed_s", _scalar(diag, "hold_time"))),
            slip=max(_scalar(diag, "left_slip_m"), _scalar(diag, "right_slip_m")),
            safety_failure=safety_failure,
            wire_centerline_error=float(torch.linalg.vector_norm(points - target, dim=1).mean()),
            tangent_error=float(torch.linalg.vector_norm(_tangents(points) - _tangents(target), dim=1).mean()),
            deformation_error=float(torch.linalg.vector_norm(segments - target_segments, dim=1).mean()),
            synchronization_error=float(torch.linalg.vector_norm(velocities[0] - velocities[-1])),
            force_balance_error=abs(sum(left_forces) - sum(right_forces)),
        )

    def _build_observation(self, snapshot: object, diag: dict[str, Any]) -> torch.Tensor:
        points, velocities = self._geometry(snapshot)
        target = self._target_centerline if self._target_centerline is not None else points
        tangents = _tangents(points)
        curvature = torch.linalg.vector_norm(tangents[1:] - tangents[:-1], dim=1)
        length = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1).sum().reshape(1)
        wire = torch.cat((points.flatten() / 2., tangents.flatten(), velocities.flatten(),
                          target.flatten() / 2., _tangents(target).flatten(), curvature, length))
        hands, arms, masks, margins = [], [], [], []
        for side in ("left", "right"):
            arm, hand = getattr(snapshot, f"{side}_arm"), getattr(snapshot, f"{side}_hand")
            forces = torch.linalg.vector_norm(_tensor(hand.fingertip_forces_b), dim=1)
            hands.append(torch.cat((_tensor(arm.palm_pose_b) / 3.2, _tensor(arm.palm_twist_b),
                                    _tensor(hand.q) / 1.6, _tensor(hand.qd) / 4.,
                                    _tensor(hand.fingertip_positions_b).flatten() / 2.,
                                    torch.stack((forces[0], forces[1], forces[2:].sum())) / 10.)))
            arms.append(torch.cat((_tensor(arm.q) / 3.2, _tensor(arm.qd) / 3.)))
            masks.extend(_tensor(hand.contact_mask).tolist())
            lower = _tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973])
            upper = _tensor([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973])
            margins.append(float(torch.minimum(arm.q - lower, upper - arm.q).min()) / 3.2)
        phase_name = infer_wire_ppo_phase(self._wire_state(snapshot, diag), self.reward_cfg).value
        phase_values = [float(phase_name == phase) for phase in ("approach", "contact", "grasp", "lift", "hold", "failed")]
        phase = _tensor(phase_values + masks + [_scalar(diag, f"{s}_slip_m") / .02 for s in ("left", "right")] + margins)
        result = torch.cat((wire, *hands, *arms, phase, self._previous_action)).to(torch.float32).clamp(-1., 1.)
        self.observation_spec.validate(result)
        return result.to(device=self.device)

    def _info(self, state: WirePpoState, diag: dict[str, Any], *, terminated: bool, truncated: bool) -> dict[str, Any]:
        phase = infer_wire_ppo_phase(state, self.reward_cfg)
        info = {
            "stage": self.stage.value,
            "phase": phase.value,
            "episode_step": self._episode_step,
            "success": bool(approval_episode(state, self.reward_cfg)),
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
        points, _ = self._geometry(snapshot)
        self._initial_wire_height = float(points[:, 2].mean())
        self._target_centerline = (self._configured_target.clone() if self._configured_target is not None
                                   else points + points.new_tensor([0., 0., self.reward_cfg.minimum_lift_m]))
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
        if not callable(step_latent):
            raise TypeError("wrapper must expose step_latent(); actions must not be discarded")
        step_latent(command.as_vector)
        snapshot = getattr(self.wrapper, "last_snapshot", None)
        adapter = getattr(self.wrapper, "adapter", None)
        live_snapshot = getattr(adapter, "snapshot", None)
        if callable(live_snapshot):
            snapshot = live_snapshot()
        diag = _diagnostics(self.wrapper)
        current = replace(self._wire_state(snapshot, diag), latent_delta_norm=float(
            torch.linalg.vector_norm(command.as_vector.cpu() - self._previous_action)))
        breakdown = compute_wire_ppo_reward(self._previous_state, current, self.stage, self.reward_cfg)
        self._episode_step += 1
        success = bool(approval_episode(current, self.reward_cfg))
        terminal_phase = str(diag.get("acceptance_phase", "")).lower() in {"terminated", "failed"}
        terminated = bool(success or terminal_phase or current.safety_failure)
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
