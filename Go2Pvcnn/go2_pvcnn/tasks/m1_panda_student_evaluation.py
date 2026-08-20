"""Strict aggregation and hard-gate checks for Student-only S1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


EXPECTED_STUDENT_SEEDS = (42, 43, 44)
MIN_EVALUATION_ENVS = 64
EVALUATION_STEPS = 4000
MIN_COMPLETION_RATE = 0.95
MIN_TEACHER_RELATIVE_SUCCESS = 0.95


def validate_student_only_summary(summary: dict[str, object], *, teacher_success_rate: float) -> None:
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary")
    if summary.get("teacher_execution_count", 0) != 0:
        raise ValueError("teacher_execution_count must be zero in Student-only mode")
    completion = float(summary.get("completion_rate", 0.0))
    if completion < MIN_COMPLETION_RATE:
        raise ValueError("completion_rate is below the Student hard gate")
    success = float(summary.get("success_rate", 0.0))
    if success < float(teacher_success_rate) * MIN_TEACHER_RELATIVE_SUCCESS:
        raise ValueError("Student success is below the Teacher relative gate")
    for field in ("qp_feasible_rate", "finite_rate", "four_wheel_contact_rate"):
        if float(summary.get(field, 0.0)) < 0.999:
            raise ValueError(f"{field} is below the hard gate")
    for field in ("unexpected_resets", "joint_limit_violations", "body_contacts", "safety_terminate_count"):
        if int(summary.get(field, 0)) != 0:
            raise ValueError(f"{field} must be zero")


@dataclass
class StudentEvaluationAccumulator:
    seed: int
    requested_steps: int
    num_envs: int
    completed_episodes: int = 0
    successful_episodes: int = 0
    teacher_execution_count: int = 0
    qp_feasible_steps: int = 0
    finite_steps: int = 0
    four_wheel_contact_steps: int = 0
    total_steps: int = 0
    unexpected_resets: int = 0
    joint_limit_violations: int = 0
    body_contacts: int = 0
    safety_terminate_count: int = 0

    def to_dict(self) -> dict[str, object]:
        episodes = max(self.completed_episodes, 1)
        steps = max(self.total_steps, 1)
        payload = {
            "seed": self.seed,
            "requested_steps": self.requested_steps,
            "num_envs": self.num_envs,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "completion_rate": self.completed_episodes / max(self.num_envs, 1),
            "success_rate": self.successful_episodes / episodes,
            "teacher_execution_count": self.teacher_execution_count,
            "qp_feasible_rate": self.qp_feasible_steps / steps,
            "finite_rate": self.finite_steps / steps,
            "four_wheel_contact_rate": self.four_wheel_contact_steps / steps,
            "unexpected_resets": self.unexpected_resets,
            "joint_limit_violations": self.joint_limit_violations,
            "body_contacts": self.body_contacts,
            "safety_terminate_count": self.safety_terminate_count,
        }
        return payload


__all__ = ["EXPECTED_STUDENT_SEEDS", "MIN_EVALUATION_ENVS", "EVALUATION_STEPS", "StudentEvaluationAccumulator", "validate_student_only_summary"]
