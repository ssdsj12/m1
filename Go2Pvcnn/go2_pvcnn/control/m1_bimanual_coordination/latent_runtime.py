"""25 Hz teacher/encoder and 200 Hz latent body-control orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch

from .contracts import BimanualCommand, BimanualSnapshot, FullDynamicsState
from .full_action_teacher import TeacherInput, TeacherSolution
from .latent_contracts import (
    LatentArtifactMetadata,
    LatentNormalizer,
    STATE_DIM,
    pack_state_features,
    pack_teacher_task_features,
)
from .latent_model import LatentActionModel
from .safety_projection import SafetyResult


SafetyInputProvider = Callable[
    [BimanualSnapshot, FullDynamicsState, torch.Tensor], object
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LatentRuntime:
    TEACHER_PERIOD = 8
    LATENT_TTL_STEPS = 16

    def __init__(
        self,
        *,
        model,
        normalizer: LatentNormalizer,
        teacher,
        safety,
        effort_limits: torch.Tensor,
        safety_input_provider: SafetyInputProvider,
    ) -> None:
        if not isinstance(normalizer, LatentNormalizer):
            raise TypeError("normalizer must be LatentNormalizer")
        if effort_limits.dtype != torch.float64 or effort_limits.device.type != "cpu" or effort_limits.shape != (43,):
            raise ValueError("effort_limits must be CPU float64 with shape (43,)")
        if not torch.all(effort_limits > 0.0).item():
            raise ValueError("effort_limits must be positive")
        if not callable(getattr(model, "encode", None)) or not callable(
            getattr(model, "body_action", None)
        ):
            raise TypeError("model must expose encode() and body_action()")
        if not callable(getattr(teacher, "plan", None)):
            raise TypeError("teacher must expose plan()")
        if not callable(getattr(safety, "project", None)):
            raise TypeError("safety must expose project()")
        if not callable(safety_input_provider):
            raise TypeError("safety_input_provider must be callable")
        self.model = model
        self.normalizer = normalizer
        self.teacher = teacher
        self.safety = safety
        self.effort_limits = effort_limits.clone()
        self.safety_input_provider = safety_input_provider
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        self.reset()

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        *,
        action_order: tuple[str, ...],
        teacher,
        safety,
        effort_limits: torch.Tensor,
        safety_input_provider: SafetyInputProvider,
    ) -> "LatentRuntime":
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / "latent_action_model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"latent model not found: {model_path}")
        normalization_path = artifact_dir / "normalization.pt"
        metadata_path = artifact_dir / "metadata.json"
        if not normalization_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("latent normalization or metadata artifact is missing")
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_metadata["action_order"] = tuple(raw_metadata["action_order"])
        raw_metadata["feature_order"] = tuple(raw_metadata["feature_order"])
        metadata = LatentArtifactMetadata(**raw_metadata)
        metadata.validate_runtime(
            action_order=action_order,
            state_dim=STATE_DIM,
            normalization_sha256=_sha256(normalization_path),
        )
        normalization = torch.load(
            normalization_path, map_location="cpu", weights_only=True
        )
        normalizer = LatentNormalizer(
            mean=normalization["mean"], scale=normalization["scale"]
        )
        model = LatentActionModel()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        return cls(
            model=model,
            normalizer=normalizer,
            teacher=teacher,
            safety=safety,
            effort_limits=effort_limits,
            safety_input_provider=safety_input_provider,
        )

    def reset(self) -> None:
        self._step = 0
        self._latent: torch.Tensor | None = None
        self._latent_step = -1
        self._last_effort = torch.zeros(43, dtype=torch.float64)
        self._last_safe_effort: torch.Tensor | None = None
        self.last_teacher_solution: TeacherSolution | None = None
        self.last_safety_result: SafetyResult | None = None
        self._counts = {"teacher": 0, "encoder": 0, "body": 0, "safety": 0}
        self.fallback_log: list[dict[str, object]] = []

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def _safe_command(self, timestamp_ns: int, reason: str) -> BimanualCommand:
        effort = (
            torch.zeros(43, dtype=torch.float64)
            if self._last_safe_effort is None
            else self._last_safe_effort.clone()
        )
        effort[12:16] = 0.0
        self.fallback_log.append(
            {
                "layer": "latent_runtime",
                "reason": reason,
                "duration_steps": 1,
                "last_safe_action_source": (
                    "zero" if self._last_safe_effort is None else "safety_projection"
                ),
            }
        )
        return BimanualCommand(
            timestamp_ns=timestamp_ns,
            effort=effort,
            feasible=False,
            fallback_reasons=(reason,),
        )

    def compute(
        self,
        snapshot: BimanualSnapshot,
        dynamics: FullDynamicsState,
        teacher_input: TeacherInput,
    ) -> BimanualCommand:
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if not isinstance(dynamics, FullDynamicsState):
            raise TypeError("dynamics must be FullDynamicsState")
        if not isinstance(teacher_input, TeacherInput):
            raise TypeError("teacher_input must be TeacherInput")
        state = self.normalizer.normalize(pack_state_features(snapshot)).unsqueeze(0)
        if not torch.isfinite(state).all().item():
            command = self._safe_command(snapshot.timestamp_ns, "nonfinite_state")
            self._step += 1
            return command
        if float(torch.max(torch.abs(state)).item()) > 10.0:
            command = self._safe_command(
                snapshot.timestamp_ns, "state_out_of_distribution"
            )
            self._step += 1
            return command
        if self._step % self.TEACHER_PERIOD == 0:
            solution = self.teacher.plan(teacher_input)
            self._counts["teacher"] += 1
            self.last_teacher_solution = solution
            if solution.diagnostics.feasible:
                with torch.no_grad():
                    latent = self.model.encode(
                        state,
                        solution.action_trajectory.to(dtype=torch.float32).unsqueeze(0),
                        pack_teacher_task_features(solution).unsqueeze(0),
                    )
                self._counts["encoder"] += 1
                if (
                    torch.isfinite(latent).all().item()
                    and float(torch.max(torch.abs(latent)).item()) <= 1.0 + 1.0e-6
                ):
                    self._latent = latent.detach().clone()
                    self._latent_step = self._step
        if self._latent is None or self._step - self._latent_step > self.LATENT_TTL_STEPS:
            command = self._safe_command(snapshot.timestamp_ns, "latent_expired")
            self._step += 1
            return command
        phase = torch.tensor(
            [[(self._step % self.TEACHER_PERIOD) / self.TEACHER_PERIOD]],
            dtype=torch.float32,
        )
        last_effort = (
            self._last_effort / self.effort_limits
        ).to(dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            normalized_candidate = self.model.body_action(
                state, self._latent, phase, last_effort
            )
        self._counts["body"] += 1
        if (
            not torch.isfinite(normalized_candidate).all().item()
            or float(torch.max(torch.abs(normalized_candidate)).item()) > 1.0 + 1.0e-6
        ):
            command = self._safe_command(snapshot.timestamp_ns, "invalid_body_action")
            self._step += 1
            return command
        candidate = (
            normalized_candidate.squeeze(0).to(dtype=torch.float64)
            * self.effort_limits
        )
        safety_input = self.safety_input_provider(snapshot, dynamics, candidate)
        safe: SafetyResult = self.safety.project(safety_input)
        self.last_safety_result = safe
        self._counts["safety"] += 1
        self._last_effort = safe.effort.clone()
        if safe.feasible:
            self._last_safe_effort = safe.effort.clone()
        command = BimanualCommand(
            timestamp_ns=snapshot.timestamp_ns,
            effort=safe.effort,
            feasible=safe.feasible,
            fallback_reasons=(
                () if safe.feasible else (safe.fallback_reason or "safety_infeasible",)
            ),
        )
        self._step += 1
        return command


__all__ = ["LatentRuntime", "SafetyInputProvider"]
