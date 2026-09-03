from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import (
    LatentNormalizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.latent_runtime import LatentRuntime
from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import SafetyResult
from tests.test_m1_bimanual_full_action_teacher import _input as _teacher_input
from tests.test_m1_bimanual_object_mpc import _snapshot


class _Teacher:
    def __init__(self) -> None:
        self.calls = 0
        self.always_fail_after_first = False

    def plan(self, sample):
        self.calls += 1
        solution = _teacher_input_solution(sample)
        if self.always_fail_after_first and self.calls > 1:
            return replace(
                solution,
                diagnostics=replace(
                    solution.diagnostics,
                    feasible=False,
                    fallback_reason="teacher_failed",
                ),
            )
        return solution


def _teacher_input_solution(sample):
    from go2_pvcnn.control.m1_bimanual_coordination.full_action_teacher import FullActionTeacher

    return FullActionTeacher().plan(sample)


class _Model:
    def encode(self, state, teacher_action, teacher_task):
        return torch.zeros((state.shape[0], 16), dtype=torch.float32)

    def body_action(self, state, latent, phase, last_effort):
        return 0.1 * torch.ones((state.shape[0], 43), dtype=torch.float32)


class _Safety:
    def __init__(self) -> None:
        self.calls = 0

    def project(self, sample):
        self.calls += 1
        effort = sample["candidate"].clone()
        effort[12:16] = 0.0
        return SafetyResult(
            effort=effort,
            feasible=True,
            active_constraints=("wheel_lock",),
            fallback_reason=None,
            dynamics_residual=0.0,
        )


def _runtime() -> LatentRuntime:
    return LatentRuntime(
        model=_Model(),
        normalizer=LatentNormalizer(
            mean=torch.zeros(111), scale=torch.ones(111)
        ),
        teacher=_Teacher(),
        safety=_Safety(),
        effort_limits=100.0 * torch.ones(43, dtype=torch.float64),
        safety_input_provider=lambda snapshot, dynamics, candidate: {
            "candidate": candidate
        },
    )


def _snapshots(count: int):
    return [replace(_snapshot(), timestamp_ns=index + 1) for index in range(count)]


def test_teacher_and_encoder_run_once_per_eight_body_steps() -> None:
    runtime = _runtime()
    sample = _teacher_input()

    for snapshot in _snapshots(17):
        runtime.compute(snapshot, sample.dynamics, sample)

    assert runtime.counts == {
        "teacher": 3,
        "encoder": 3,
        "body": 17,
        "safety": 17,
    }


def test_last_valid_z_expires_after_sixteen_physics_steps() -> None:
    runtime = _runtime()
    runtime.teacher.always_fail_after_first = True
    sample = _teacher_input()

    commands = [
        runtime.compute(snapshot, sample.dynamics, sample)
        for snapshot in _snapshots(18)
    ]

    assert commands[16].feasible
    assert not commands[17].feasible
    assert "latent_expired" in commands[17].fallback_reasons


def test_missing_model_rejects_latent_runtime() -> None:
    with pytest.raises(FileNotFoundError, match="latent model"):
        LatentRuntime.from_artifact(
            Path("/missing/model"),
            action_order=tuple(f"joint_{index}" for index in range(43)),
            teacher=_Teacher(),
            safety=_Safety(),
            effort_limits=torch.ones(43, dtype=torch.float64),
            safety_input_provider=lambda snapshot, dynamics, candidate: candidate,
        )


def test_out_of_distribution_state_is_rejected_before_body_control() -> None:
    runtime = _runtime()
    snapshot = _snapshots(1)[0]
    snapshot.base_state[0] = 11.0
    sample = _teacher_input()

    command = runtime.compute(snapshot, sample.dynamics, sample)

    assert not command.feasible
    assert command.fallback_reasons == ("state_out_of_distribution",)
    assert runtime.counts["body"] == 0

