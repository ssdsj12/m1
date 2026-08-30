#!/usr/bin/env python3
"""GPU0 Phase-5 physical gate for stationary M1 plus small Panda EE motion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"


def build_arg_parser(*, include_app_launcher_args: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--trajectory-scale", type=float, default=0.25)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--stats-interval", type=int, default=200)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.task != TASK_ID:
        raise ValueError(f"--task must be {TASK_ID}")
    if args.steps <= 0 or args.num_envs != 1 or args.stats_interval <= 0:
        raise ValueError("steps/stats must be positive and num_envs must equal one")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be a non-empty unique list")
    if not math.isfinite(args.trajectory_scale) or not 0.0 < args.trajectory_scale <= 1.0:
        raise ValueError("--trajectory-scale must be in (0,1]")


@dataclass
class Phase5Summary:
    seed: int
    requested_steps: int
    steps: int = 0
    finite: bool = True
    mpc_feasible_count: int = 0
    qp_feasible_count: int = 0
    min_wheel_contact_count: int = 4
    base_contacts: int = 0
    joint_limit_violations: int = 0
    reset_count: int = 0
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    max_ee_position_error_m: float = 0.0
    max_ee_orientation_error_rad: float = 0.0
    eligible_force_samples: int = 0
    eligible_moment_samples: int = 0
    force_direction_cosine_sum: float = 0.0
    moment_direction_cosine_sum: float = 0.0
    actual_force_direction_cosine_sum: float = 0.0
    actual_moment_direction_cosine_sum: float = 0.0
    bias_plus_force_direction_cosine_sum: float = 0.0
    bias_minus_force_direction_cosine_sum: float = 0.0
    bias_plus_moment_direction_cosine_sum: float = 0.0
    bias_minus_moment_direction_cosine_sum: float = 0.0
    raw_force_direction_cosine_sum: float = 0.0
    raw_moment_direction_cosine_sum: float = 0.0
    raw_eligible_force_samples: int = 0
    raw_eligible_moment_samples: int = 0
    mpc_fallback_count: int = 0
    max_arm_reference_error_rad: float = 0.0
    max_arm_qd_ref_abs_rad_s: float = 0.0
    max_arm_qdd_first_abs_rad_s2: float = 0.0
    max_actual_arm_qdd_abs_rad_s2: float = 0.0
    max_correction_wrench_norm: float = 0.0
    max_root_xy_displacement_m: float = 0.0
    final_arm_q: list[float] = field(default_factory=list)
    final_arm_q_ref: list[float] = field(default_factory=list)
    final_measured_mount_wrench_b: list[float] = field(default_factory=list)
    final_predicted_mount_wrench_b: list[float] = field(default_factory=list)
    final_actual_dynamic_mount_wrench_b: list[float] = field(default_factory=list)
    final_correction_wrench_b: list[float] = field(default_factory=list)
    final_target_pose: list[float] = field(default_factory=list)
    final_current_ee_pose: list[float] = field(default_factory=list)
    final_predicted_ee_pose_first: list[float] = field(default_factory=list)
    final_replan_start_ee_pose: list[float] = field(default_factory=list)
    final_arm_qd_ref: list[float] = field(default_factory=list)
    final_predicted_ee_pose_terminal: list[float] = field(default_factory=list)
    final_root_xy: list[float] = field(default_factory=list)
    initial_root_xy: list[float] = field(default_factory=list)
    best_force_lag_steps: int = 0
    best_moment_lag_steps: int = 0
    best_force_direction_cosine: float = 0.0
    best_moment_direction_cosine: float = 0.0
    exit_reason: str = "not_started"

    @property
    def force_direction_cosine(self) -> float:
        return self.force_direction_cosine_sum / max(self.eligible_force_samples, 1)

    @property
    def moment_direction_cosine(self) -> float:
        return self.moment_direction_cosine_sum / max(self.eligible_moment_samples, 1)

    @property
    def actual_force_direction_cosine(self) -> float:
        return self.actual_force_direction_cosine_sum / max(
            self.eligible_force_samples, 1
        )

    @property
    def actual_moment_direction_cosine(self) -> float:
        return self.actual_moment_direction_cosine_sum / max(
            self.eligible_moment_samples, 1
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "seed": self.seed,
            "requested_steps": self.requested_steps,
            "steps": self.steps,
            "finite": self.finite,
            "mpc_feasible_rate": self.mpc_feasible_count / max(self.steps, 1),
            "qp_feasible_rate": self.qp_feasible_count / max(self.steps, 1),
            "min_wheel_contact_count": self.min_wheel_contact_count,
            "base_contacts": self.base_contacts,
            "joint_limit_violations": self.joint_limit_violations,
            "reset_count": self.reset_count,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "max_ee_position_error_m": self.max_ee_position_error_m,
            "max_ee_orientation_error_rad": self.max_ee_orientation_error_rad,
            "eligible_force_samples": self.eligible_force_samples,
            "eligible_moment_samples": self.eligible_moment_samples,
            "force_direction_cosine": self.force_direction_cosine,
            "moment_direction_cosine": self.moment_direction_cosine,
            "actual_force_direction_cosine": self.actual_force_direction_cosine,
            "actual_moment_direction_cosine": self.actual_moment_direction_cosine,
            "bias_plus_force_direction_cosine": (
                self.bias_plus_force_direction_cosine_sum
                / max(self.eligible_force_samples, 1)
            ),
            "bias_minus_force_direction_cosine": (
                self.bias_minus_force_direction_cosine_sum
                / max(self.eligible_force_samples, 1)
            ),
            "bias_plus_moment_direction_cosine": (
                self.bias_plus_moment_direction_cosine_sum
                / max(self.eligible_moment_samples, 1)
            ),
            "bias_minus_moment_direction_cosine": (
                self.bias_minus_moment_direction_cosine_sum
                / max(self.eligible_moment_samples, 1)
            ),
            "raw_force_direction_cosine": (
                self.raw_force_direction_cosine_sum
                / max(self.raw_eligible_force_samples, 1)
            ),
            "raw_moment_direction_cosine": (
                self.raw_moment_direction_cosine_sum
                / max(self.raw_eligible_moment_samples, 1)
            ),
            "raw_eligible_force_samples": self.raw_eligible_force_samples,
            "raw_eligible_moment_samples": self.raw_eligible_moment_samples,
            "mpc_fallback_count": self.mpc_fallback_count,
            "max_arm_reference_error_rad": self.max_arm_reference_error_rad,
            "max_arm_qd_ref_abs_rad_s": self.max_arm_qd_ref_abs_rad_s,
            "max_arm_qdd_first_abs_rad_s2": self.max_arm_qdd_first_abs_rad_s2,
            "max_actual_arm_qdd_abs_rad_s2": self.max_actual_arm_qdd_abs_rad_s2,
            "max_correction_wrench_norm": self.max_correction_wrench_norm,
            "max_root_xy_displacement_m": self.max_root_xy_displacement_m,
            "final_arm_q": list(self.final_arm_q),
            "final_arm_q_ref": list(self.final_arm_q_ref),
            "final_measured_mount_wrench_b": list(self.final_measured_mount_wrench_b),
            "final_predicted_mount_wrench_b": list(self.final_predicted_mount_wrench_b),
            "final_actual_dynamic_mount_wrench_b": list(
                self.final_actual_dynamic_mount_wrench_b
            ),
            "final_correction_wrench_b": list(self.final_correction_wrench_b),
            "final_target_pose": list(self.final_target_pose),
            "final_current_ee_pose": list(self.final_current_ee_pose),
            "final_predicted_ee_pose_first": list(self.final_predicted_ee_pose_first),
            "final_replan_start_ee_pose": list(self.final_replan_start_ee_pose),
            "final_arm_qd_ref": list(self.final_arm_qd_ref),
            "final_predicted_ee_pose_terminal": list(
                self.final_predicted_ee_pose_terminal
            ),
            "final_root_xy": list(self.final_root_xy),
            "initial_root_xy": list(self.initial_root_xy),
            "best_force_lag_steps": self.best_force_lag_steps,
            "best_moment_lag_steps": self.best_moment_lag_steps,
            "best_force_direction_cosine": self.best_force_direction_cosine,
            "best_moment_direction_cosine": self.best_moment_direction_cosine,
            "exit_reason": self.exit_reason,
        }
        payload["accepted"] = phase5_gates_pass(self)
        return payload


def phase5_gates_pass(summary: Phase5Summary) -> bool:
    return bool(
        summary.steps == summary.requested_steps
        and summary.finite
        and summary.mpc_feasible_count / max(summary.steps, 1) >= 0.99
        and summary.qp_feasible_count == summary.steps
        and summary.min_wheel_contact_count == 4
        and summary.base_contacts == 0
        and summary.joint_limit_violations == 0
        and summary.reset_count == 0
        and summary.max_abs_roll_rad <= math.radians(10.0)
        and summary.max_abs_pitch_rad <= math.radians(10.0)
        and summary.max_ee_position_error_m <= 0.015
        and summary.max_ee_orientation_error_rad <= 0.08
        and summary.eligible_force_samples > 0
        and summary.eligible_moment_samples > 0
        and summary.force_direction_cosine >= 0.8
        and summary.moment_direction_cosine >= 0.8
        and summary.exit_reason == "steps_complete"
    )


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator <= 1.0e-12:
        return 0.0
    return float(torch.dot(first, second).item() / denominator.item())


def _motion_wrench_increment(active, baseline) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matched active-minus-hold increments for sensor and estimator."""

    measured = (
        active["dynamic_measured_mount_wrench_b"][0]
        - baseline["dynamic_measured_mount_wrench_b"][0]
    )
    predicted = (
        active["predicted_mount_wrench_b"][0]
        - baseline["predicted_mount_wrench_b"][0]
    )
    if measured.shape != (6,) or predicted.shape != (6,):
        raise ValueError("baseline wrench snapshots must each contain one 6D wrench")
    if not torch.isfinite(measured).all().item() or not torch.isfinite(
        predicted
    ).all().item():
        raise ValueError("motion wrench increments must contain only finite values")
    return measured, predicted


def _bias_augmented_predictions(active, baseline, predicted):
    """Return both bias-force signs as diagnostics before changing formal semantics."""

    bias_delta = active["base_bias_wrench"][0] - baseline["base_bias_wrench"][0]
    if bias_delta.shape != (6,) or predicted.shape != (6,):
        raise ValueError("base bias and predicted wrench must each be 6D")
    return predicted + bias_delta, predicted - bias_delta


def _raw_motion_wrench_increment(active, baseline):
    raw = (
        active["measured_mount_wrench_b"][0]
        - baseline["measured_mount_wrench_b"][0]
    )
    if raw.shape != (6,) or not torch.isfinite(raw).all().item():
        raise ValueError("raw motion wrench increment must be finite and 6D")
    return raw


BASELINE_SNAPSHOT_FIELDS = (
    "dynamic_measured_mount_wrench_b",
    "predicted_mount_wrench_b",
    "actual_dynamic_mount_wrench_b",
    "base_bias_wrench",
    "measured_mount_wrench_b",
)


def _baseline_snapshot_after_step(runtime, *, terminal, final_step, previous):
    """Read diagnostics unless the final environment step already auto-reset.

    Manager-based environments reset immediately after a terminal step.  At the
    normal episode timeout the wrapper therefore has no completed MPC plan to
    expose.  Repeating the immediately preceding hold snapshot is preferable to
    reading unrelated post-reset state and changes only one endpoint sample.
    """

    if terminal and final_step:
        if previous is None:
            raise RuntimeError("final timeout occurred before a baseline snapshot")
        return {name: previous[name].clone() for name in BASELINE_SNAPSHOT_FIELDS}
    snapshot = runtime.diagnostics_snapshot()
    return {name: snapshot[name].clone() for name in BASELINE_SNAPSHOT_FIELDS}


def _lagged_direction_cosines(
    measured: torch.Tensor, predicted: torch.Tensor, *, max_lag: int = 8
) -> dict[str, float | int]:
    """Diagnose causal sensor delay without relaxing the zero-lag formal gate."""

    if measured.ndim != 2 or measured.shape != predicted.shape or measured.shape[1] != 6:
        raise ValueError("wrench histories must have matching shape (steps, 6)")
    if measured.shape[0] == 0 or max_lag < 0:
        raise ValueError("wrench histories must be non-empty and max_lag non-negative")
    if not torch.isfinite(measured).all().item() or not torch.isfinite(predicted).all().item():
        raise ValueError("wrench histories must be finite")

    best = {
        "best_force_lag_steps": 0,
        "best_moment_lag_steps": 0,
        "best_force_direction_cosine": -1.0,
        "best_moment_direction_cosine": -1.0,
    }
    for lag in range(min(max_lag, measured.shape[0] - 1) + 1):
        current_measured = measured[lag:]
        earlier_predicted = predicted[: measured.shape[0] - lag]
        for name, section, threshold in (
            ("force", slice(0, 3), 2.0),
            ("moment", slice(3, 6), 0.2),
        ):
            norms = torch.linalg.vector_norm(current_measured[:, section], dim=1)
            eligible = norms >= threshold
            if not bool(eligible.any().item()):
                continue
            values = [
                _cosine(prediction, measurement)
                for prediction, measurement in zip(
                    earlier_predicted[eligible, section],
                    current_measured[eligible, section],
                )
            ]
            score = sum(values) / len(values)
            score_key = f"best_{name}_direction_cosine"
            if score > best[score_key]:
                best[score_key] = score
                best[f"best_{name}_lag_steps"] = lag
    for name in ("force", "moment"):
        key = f"best_{name}_direction_cosine"
        if best[key] < -0.5:
            best[key] = 0.0
    return best


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _extend_episode_horizon(cfg, *, steps: int) -> None:
    """Prevent the formal sample window from coinciding with an auto-reset."""

    control_dt = float(cfg.sim.dt) * int(cfg.decimation)
    minimum_horizon = (steps + 1) * control_dt
    cfg.episode_length_s = max(float(cfg.episode_length_s), minimum_horizon)


def _update(summary, runtime, terminated, truncated, baseline_snapshot):
    snapshot = runtime.diagnostics_snapshot()
    state = runtime.states[0]
    solution = runtime._solutions[0]
    measured, predicted = _motion_wrench_increment(snapshot, baseline_snapshot)
    bias_plus, bias_minus = _bias_augmented_predictions(
        snapshot, baseline_snapshot, predicted
    )
    raw_measured = _raw_motion_wrench_increment(snapshot, baseline_snapshot)
    actual_predicted = snapshot["actual_dynamic_mount_wrench_b"][0]
    actual_predicted = (
        actual_predicted
        - baseline_snapshot["actual_dynamic_mount_wrench_b"][0]
    )
    correction = snapshot["correction_wrench_b"][0]
    arm_q = snapshot["arm_q"][0]
    arm_q_ref = snapshot["arm_q_ref"][0]
    arm_qd_ref = snapshot["arm_qd_ref"][0]
    arm_qdd_first = snapshot["arm_qdd_first"][0]
    actual_arm_qdd = snapshot["actual_arm_qdd"][0]
    root_xy = snapshot["root_xy"][0]
    initial_root_xy = snapshot["initial_root_xy"][0]
    error = snapshot["target_pose"][0] - snapshot["current_ee_pose"][0]
    summary.steps += 1
    summary.mpc_feasible_count += int(snapshot["mpc_feasible"][0])
    summary.qp_feasible_count += int(snapshot["qp_feasible"][0])
    summary.mpc_fallback_count += int(snapshot["mpc_fallback"][0])
    summary.max_arm_reference_error_rad = max(
        summary.max_arm_reference_error_rad,
        float(torch.max(torch.abs(arm_q_ref - arm_q)).item()),
    )
    summary.max_arm_qd_ref_abs_rad_s = max(
        summary.max_arm_qd_ref_abs_rad_s,
        float(torch.max(torch.abs(arm_qd_ref)).item()),
    )
    summary.max_arm_qdd_first_abs_rad_s2 = max(
        summary.max_arm_qdd_first_abs_rad_s2,
        float(torch.max(torch.abs(arm_qdd_first)).item()),
    )
    summary.max_actual_arm_qdd_abs_rad_s2 = max(
        summary.max_actual_arm_qdd_abs_rad_s2,
        float(torch.max(torch.abs(actual_arm_qdd)).item()),
    )
    summary.max_correction_wrench_norm = max(
        summary.max_correction_wrench_norm,
        float(torch.linalg.vector_norm(correction).item()),
    )
    summary.max_root_xy_displacement_m = max(
        summary.max_root_xy_displacement_m,
        float(torch.linalg.vector_norm(root_xy - initial_root_xy).item()),
    )
    summary.final_arm_q = arm_q.tolist()
    summary.final_arm_q_ref = arm_q_ref.tolist()
    summary.final_measured_mount_wrench_b = measured.tolist()
    summary.final_predicted_mount_wrench_b = predicted.tolist()
    summary.final_actual_dynamic_mount_wrench_b = actual_predicted.tolist()
    summary.final_correction_wrench_b = correction.tolist()
    summary.final_target_pose = snapshot["target_pose"][0].tolist()
    summary.final_current_ee_pose = snapshot["current_ee_pose"][0].tolist()
    summary.final_predicted_ee_pose_first = snapshot[
        "predicted_ee_pose_first"
    ][0].tolist()
    summary.final_replan_start_ee_pose = snapshot[
        "replan_start_ee_pose"
    ][0].tolist()
    summary.final_arm_qd_ref = arm_qd_ref.tolist()
    summary.final_predicted_ee_pose_terminal = snapshot[
        "predicted_ee_pose_terminal"
    ][0].tolist()
    summary.final_root_xy = root_xy.tolist()
    summary.initial_root_xy = initial_root_xy.tolist()
    summary.min_wheel_contact_count = min(summary.min_wheel_contact_count, state.wheel_contact_count)
    summary.base_contacts += runtime.base_contacts[0]
    summary.joint_limit_violations += runtime.adapters[0].joint_limit_violations()
    summary.reset_count += int(bool(terminated[0] or truncated[0]))
    summary.max_abs_roll_rad = max(summary.max_abs_roll_rad, abs(float(state.roll)))
    summary.max_abs_pitch_rad = max(summary.max_abs_pitch_rad, abs(float(state.pitch)))
    summary.max_ee_position_error_m = max(
        summary.max_ee_position_error_m, float(torch.linalg.vector_norm(error[:3]).item())
    )
    summary.max_ee_orientation_error_rad = max(
        summary.max_ee_orientation_error_rad, float(torch.linalg.vector_norm(error[3:]).item())
    )
    force_norm = float(torch.linalg.vector_norm(measured[:3]).item())
    moment_norm = float(torch.linalg.vector_norm(measured[3:]).item())
    if force_norm >= 2.0:
        summary.eligible_force_samples += 1
        summary.force_direction_cosine_sum += _cosine(predicted[:3], measured[:3])
        summary.actual_force_direction_cosine_sum += _cosine(
            actual_predicted[:3], measured[:3]
        )
        summary.bias_plus_force_direction_cosine_sum += _cosine(
            bias_plus[:3], measured[:3]
        )
        summary.bias_minus_force_direction_cosine_sum += _cosine(
            bias_minus[:3], measured[:3]
        )
    if moment_norm >= 0.2:
        summary.eligible_moment_samples += 1
        summary.moment_direction_cosine_sum += _cosine(predicted[3:], measured[3:])
        summary.actual_moment_direction_cosine_sum += _cosine(
            actual_predicted[3:], measured[3:]
        )
        summary.bias_plus_moment_direction_cosine_sum += _cosine(
            bias_plus[3:], measured[3:]
        )
        summary.bias_minus_moment_direction_cosine_sum += _cosine(
            bias_minus[3:], measured[3:]
        )
    raw_force_norm = float(torch.linalg.vector_norm(raw_measured[:3]).item())
    raw_moment_norm = float(torch.linalg.vector_norm(raw_measured[3:]).item())
    if raw_force_norm >= 2.0:
        summary.raw_eligible_force_samples += 1
        summary.raw_force_direction_cosine_sum += _cosine(
            predicted[:3], raw_measured[:3]
        )
    if raw_moment_norm >= 0.2:
        summary.raw_eligible_moment_samples += 1
        summary.raw_moment_direction_cosine_sum += _cosine(
            predicted[3:], raw_measured[3:]
        )
    tensors = (
        measured,
        predicted,
        actual_predicted,
        snapshot["effort"],
        snapshot["current_ee_pose"],
    )
    summary.finite = summary.finite and all(torch.isfinite(value).all().item() for value in tensors)
    debug = {
        "correction": correction.clone(),
        "root_xy": root_xy.clone(),
        "arm_q": arm_q.clone(),
        "arm_q_ref": arm_q_ref.clone(),
        "target_pose": snapshot["target_pose"][0].clone(),
        "current_ee_pose": snapshot["current_ee_pose"][0].clone(),
        "actual_arm_qdd": actual_arm_qdd.clone(),
        "actual_dynamic_mount_wrench_b": actual_predicted.clone(),
        "base_bias_wrench": snapshot["base_bias_wrench"][0].clone(),
        "planned_predicted_mount_wrench_b": snapshot[
            "planned_predicted_mount_wrench_b"
        ][0].clone(),
        "root_com_offset_b": runtime.adapters[0].robot.data.body_com_pos_b[
            runtime.adapters[0].env_index, runtime.adapters[0].base_body_id
        ].detach().to(device="cpu", dtype=torch.float64).clone(),
        "incoming_joint_wrench_child": snapshot[
            "incoming_joint_wrench_child"
        ][0].clone(),
        "rne_reaction_raw_b": snapshot["rne_reaction_raw_b"][0].clone(),
        "joint_torque_wrench_b": snapshot["joint_torque_wrench_b"][0].clone(),
        "projected_joint_force": snapshot["projected_joint_force"][0].clone(),
        "actuation_force": snapshot["actuation_force"][0].clone(),
        **{
            f"rne_{name}": snapshot["rne_terms_w"][name][0].clone()
            for name in snapshot["rne_terms_w"]
        },
    }
    return measured.clone(), predicted.clone(), debug


def _collect_zero_motion_baseline(
    *, args, seed, env, wrapper_type, simulation_app
) -> list[dict[str, torch.Tensor]]:
    """Run a synchronized fixed-seed hold trial for dynamic-wrench subtraction."""

    torch.manual_seed(seed)
    wrapper = wrapper_type(env, seed=seed, trajectory_scale=0.0)
    wrapper.reset()
    action = torch.zeros((1, 8), device=wrapper.device)
    snapshots = []
    for baseline_step in range(args.steps):
        if not simulation_app.is_running():
            raise RuntimeError("application closed during zero-motion baseline")
        _, _, dones, _ = wrapper.step(action)
        if bool(dones.any().item()) and baseline_step + 1 < args.steps:
            raise RuntimeError(
                "unexpected reset during zero-motion baseline "
                f"at step {baseline_step + 1}"
            )
        snapshots.append(
            _baseline_snapshot_after_step(
                wrapper.runtime,
                terminal=bool(dones.any().item()),
                final_step=baseline_step + 1 == args.steps,
                previous=snapshots[-1] if snapshots else None,
            )
        )
        if (baseline_step + 1) % args.stats_interval == 0:
            print(
                f"[Phase5 baseline] seed={seed} step={baseline_step + 1}",
                flush=True,
            )
    return snapshots


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    summaries = []
    debug_histories = {}
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
            M1PandaArmMpcResidualEnvWrapper,
        )

        for seed in args.seeds:
            summary = Phase5Summary(seed=seed, requested_steps=args.steps)
            env = None
            try:
                torch.manual_seed(seed)
                cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
                cfg.seed = seed
                _extend_episode_horizon(cfg, steps=args.steps)
                env = gym.make(args.task, cfg=cfg).unwrapped
                baseline = _collect_zero_motion_baseline(
                    args=args,
                    seed=seed,
                    env=env,
                    wrapper_type=M1PandaArmMpcResidualEnvWrapper,
                    simulation_app=simulation_app,
                )
                torch.manual_seed(seed)
                wrapper = M1PandaArmMpcResidualEnvWrapper(
                    env, seed=seed, trajectory_scale=args.trajectory_scale
                )
                wrapper.reset()
                action = torch.zeros((1, 8), device=wrapper.device)
                measured_history = []
                predicted_history = []
                state_history = []
                for step in range(args.steps):
                    if not simulation_app.is_running():
                        summary.exit_reason = "application_closed"
                        break
                    _, _, dones, extras = wrapper.step(action)
                    terminal_for_metrics = (
                        extras["terminated"], extras["truncated"]
                    )
                    if step + 1 == args.steps:
                        terminal_for_metrics = (
                            torch.zeros_like(extras["terminated"]),
                            torch.zeros_like(extras["truncated"]),
                        )
                    measured, predicted, debug = _update(
                        summary,
                        wrapper.runtime,
                        *terminal_for_metrics,
                        baseline[step],
                    )
                    debug["baseline_predicted_mount_wrench_b"] = baseline[step][
                        "predicted_mount_wrench_b"
                    ][0].clone()
                    debug["baseline_dynamic_measured_mount_wrench_b"] = baseline[
                        step
                    ]["dynamic_measured_mount_wrench_b"][0].clone()
                    measured_history.append(measured)
                    predicted_history.append(predicted)
                    state_history.append(debug)
                    if bool(dones.any().item()) and step + 1 < args.steps:
                        summary.exit_reason = "unexpected_reset"
                        break
                    if (step + 1) % args.stats_interval == 0:
                        print(
                            f"[Arm MPC probe] seed={seed} step={step + 1} "
                            f"mpc={summary.mpc_feasible_count / summary.steps:.4f} "
                            f"qp={summary.qp_feasible_count / summary.steps:.4f} "
                            f"ee={summary.max_ee_position_error_m:.5f}", flush=True,
                        )
                if summary.exit_reason == "not_started":
                    summary.exit_reason = "steps_complete" if summary.steps == args.steps else "incomplete"
                if measured_history:
                    lag_diagnostics = _lagged_direction_cosines(
                        torch.stack(measured_history),
                        torch.stack(predicted_history),
                    )
                    for name, value in lag_diagnostics.items():
                        setattr(summary, name, value)
                    if args.history_json is not None:
                        debug_histories[str(seed)] = {
                            "measured": torch.stack(measured_history).tolist(),
                            "predicted": torch.stack(predicted_history).tolist(),
                            **{
                                name: torch.stack(
                                    [sample[name] for sample in state_history]
                                ).tolist()
                                for name in state_history[0]
                            },
                        }
            except Exception:
                summary.exit_reason = "error"
                traceback.print_exc()
            finally:
                if env is not None:
                    env.close()
            summaries.append(summary.to_dict())
            print("[Phase5 summary] " + json.dumps(summaries[-1], sort_keys=True), flush=True)
        payload = {"task": args.task, "summaries": summaries, "accepted": all(item["accepted"] for item in summaries)}
        if args.summary_json is not None:
            _atomic_json(args.summary_json, payload)
        if args.history_json is not None:
            _atomic_json(args.history_json, debug_histories)
        print("[Phase5 aggregate] " + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if payload["accepted"] else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
