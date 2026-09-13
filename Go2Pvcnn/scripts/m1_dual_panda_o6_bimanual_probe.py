"""Smoke and formal fixed-condition acceptance for inline bimanual MPC."""

from __future__ import annotations
from dataclasses import asdict

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence


GYM_ID = "Isaac-M1-DualPanda-O6-Bimanual-Lift-v0"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TRIALS_PER_SEED = 10
DEFAULT_FORMAL_STEPS = 2000


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def make_probe_summary(*, configured: bool = False) -> dict[str, object]:
    """Create the per-side prior summary without inventing disabled values."""

    return {
        "prior_configured": bool(configured),
        "prior_enabled": False,
        "prior_disabled_reason": "no_enabled_inference" if configured else "not_configured",
        "prior_component": None,
        "prior_probability": None,
        "prior_precision_min": None,
        "prior_precision_max": None,
        "prior_soft_cost": None,
        "prior_inference_p99_ms": None,
        "prior_baseline_tip_velocity_delta_norm": None,
        "prior_qp_rejected_count": 0,
        "prior_fallback_count": 0,
        "_enabled_inference_ms": [],
        "_components": [],
        "_probabilities": [],
        "_precision_mins": [],
        "_precision_maxs": [],
        "_costs": [],
        "_tip_velocity_deltas": [],
        "_disabled_reasons": {},
    }


def accumulate_probe_prior_diagnostics(
    summary: dict[str, object], diagnostics: object | None
) -> None:
    """Accumulate one newly solved Hand-MPC diagnostic for one physical side."""

    if diagnostics is None:
        return
    configured = bool(getattr(diagnostics, "prior_configured", False))
    if configured:
        summary["prior_configured"] = True
    enabled = bool(getattr(diagnostics, "prior_enabled", False))
    reason = getattr(diagnostics, "prior_fallback_reason", None)
    if configured and isinstance(reason, str) and reason:
        reasons = summary["_disabled_reasons"]
        assert isinstance(reasons, dict)
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        summary["prior_fallback_count"] = int(summary["prior_fallback_count"]) + 1
        if reason == "prior_qp_rejected":
            summary["prior_qp_rejected_count"] = int(summary["prior_qp_rejected_count"]) + 1
    inference_ms = _finite_number(getattr(diagnostics, "prior_inference_ms", None))
    if not enabled or inference_ms is None or inference_ms < 0.0:
        return
    latencies = summary["_enabled_inference_ms"]
    assert isinstance(latencies, list)
    latencies.append(inference_ms)
    for key, field in (
        ("_components", "prior_component"),
        ("_probabilities", "prior_probability"),
        ("_precision_mins", "prior_precision_min"),
        ("_precision_maxs", "prior_precision_max"),
        ("_costs", "prior_cost"),
        ("_tip_velocity_deltas", "regularized_tip_velocity_delta_norm"),
    ):
        value = getattr(diagnostics, field, None)
        if key == "_components" and isinstance(value, int) and not isinstance(value, bool):
            values = summary[key]
            assert isinstance(values, list)
            values.append(value)
        else:
            number = _finite_number(value)
            if number is not None:
                values = summary[key]
                assert isinstance(values, list)
                values.append(number)


def finalize_probe_prior_summary(summary: dict[str, object]) -> dict[str, object]:
    """Return canonical JSON-safe fields; p99 uses enabled samples only."""

    result = {key: value for key, value in summary.items() if not key.startswith("_")}
    latencies = summary["_enabled_inference_ms"]
    assert isinstance(latencies, list)
    if not latencies:
        reasons = summary["_disabled_reasons"]
        assert isinstance(reasons, dict)
        if result["prior_configured"] and reasons:
            result["prior_disabled_reason"] = sorted(
                reasons, key=lambda value: (-int(reasons[value]), value)
            )[0]
        return result
    result["prior_enabled"] = True
    result["prior_disabled_reason"] = None
    ordered = sorted(float(value) for value in latencies)
    result["prior_inference_p99_ms"] = ordered[
        max(0, math.ceil(0.99 * len(ordered)) - 1)
    ]
    components = summary["_components"]
    probabilities = summary["_probabilities"]
    precision_mins = summary["_precision_mins"]
    precision_maxs = summary["_precision_maxs"]
    costs = summary["_costs"]
    deltas = summary["_tip_velocity_deltas"]
    assert all(isinstance(value, list) for value in (
        components, probabilities, precision_mins, precision_maxs, costs, deltas
    ))
    result["prior_component"] = components[-1] if components else None
    result["prior_probability"] = probabilities[-1] if probabilities else None
    result["prior_precision_min"] = min(precision_mins) if precision_mins else None
    result["prior_precision_max"] = max(precision_maxs) if precision_maxs else None
    result["prior_soft_cost"] = sum(costs) / len(costs) if costs else None
    result["prior_baseline_tip_velocity_delta_norm"] = (
        sum(deltas) / len(deltas) if deltas else None
    )
    return result


def trial_passes(row: dict[str, object]) -> bool:
    """Apply every frozen per-trial acceptance gate."""

    return (
        float(row["lift_height_m"]) >= 0.10
        and float(row["hold_duration_s"]) >= 3.0
        and float(row["hold_position_error_m"]) <= 0.02
        and float(row["hold_orientation_error_rad"]) <= 0.10
        and float(row["relative_palm_slip_m"]) <= 0.005
        and float(row["object_mpc_feasible_rate"]) >= 0.98
        and min(float(value) for value in row["arm_mpc_feasible_rates"]) >= 0.99
        and min(float(value) for value in row["hand_mpc_feasible_rates"]) >= 0.99
        and float(row["wbc_qp_feasible_rate"]) == 1.0
        and float(row["max_abs_roll_rad"]) <= math.radians(10.0)
        and float(row["max_abs_pitch_rad"]) <= math.radians(10.0)
        and int(row["hard_failure_count"]) == 0
        and bool(row["released_supported"]) is True
        and bool(row["box_dropped"]) is False
    )


def aggregate_acceptance(
    trials: Sequence[dict[str, object]],
    *,
    seeds: Sequence[int],
    trials_per_seed: int,
) -> dict[str, object]:
    """Require the exact seed/index matrix and 100% per-trial success."""

    expected = {(int(seed), index) for seed in seeds for index in range(trials_per_seed)}
    observed = {(int(row["seed"]), int(row["trial_index"])) for row in trials}
    exact_trials = len(trials) == len(expected) and observed == expected
    passing = sum(trial_passes(row) for row in trials)
    return {
        "accepted": bool(exact_trials and passing == len(expected)),
        "expected_trial_count": len(expected),
        "observed_trial_count": len(trials),
        "passing_trial_count": passing,
        "exact_trial_matrix": exact_trials,
        "seeds": [int(seed) for seed in seeds],
        "trials_per_seed": int(trials_per_seed),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted((root / "go2_pvcnn/control/m1_bimanual_coordination").glob("*.py"))
    sources += [
        root / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py",
        root / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py",
        Path(__file__).resolve(),
        root / "scripts/m1_dual_panda_o6_verify.py",
    ]
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_ref(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_progress(
    path: Path | None, *, executed_physics_steps: int, num_envs: int
) -> None:
    if path is not None:
        _atomic_json(
            path,
            {
                "executed_physics_steps": int(executed_physics_steps),
                "num_envs": int(num_envs),
            },
        )


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write one canonical trial object per line without exposing partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")))
            stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _artifact_manifest(
    report: dict[str, object], *, report_path: Path, jsonl_path: Path
) -> dict[str, object]:
    """Pin the exact formal results and the source/asset identity that produced them."""

    metadata = report["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("report metadata must be an object")
    return {
        "schema_version": 1,
        "status": "passed" if bool(report["passed"]) else "failed",
        "acceptance_mode": report["acceptance_mode"],
        "aggregate": report["aggregate"],
        "pins": {
            "trials_jsonl": {
                "path": str(jsonl_path.resolve()),
                "sha256": _sha256(jsonl_path),
            },
            "aggregate_report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path),
            },
            "asset_sha256": metadata["asset_sha256"],
            "source_sha256": metadata["source_sha256"],
            "git_ref": metadata["git_ref"],
        },
        "runtime": {
            "isaac_version": metadata["isaac_version"],
            "command": metadata["command"],
        },
    }


def _roll_pitch(quaternion_wxyz) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _finite_snapshot(snapshot) -> bool:
    import torch

    tensors = (
        snapshot.base_state,
        snapshot.m1_q,
        snapshot.m1_qd,
        snapshot.platform_q_qd,
        snapshot.left_arm.q,
        snapshot.right_arm.q,
        snapshot.left_hand.q,
        snapshot.right_hand.q,
        snapshot.box.pose_b,
        snapshot.box.twist_b,
    )
    return all(torch.isfinite(value).all().item() for value in tensors)


def _run_trial(
    env,
    wrapper_type,
    *,
    seed: int,
    trial_index: int,
    steps: int,
    mode: str,
    latent_artifact: Path | None,
    fingertip_prior_artifact: Path | None,
    runtime_factory: Callable[[], object],
    progress_path: Path | None = None,
) -> dict[str, object]:
    import torch

    wrapper = wrapper_type(
        env,
        runtime=runtime_factory(),
        mode=mode,
        latent_artifact=latent_artifact,
        fingertip_prior_artifact=fingertip_prior_artifact,
    )
    initial = wrapper.reset(seed=seed)
    _write_progress(
        progress_path,
        executed_physics_steps=1,
        num_envs=int(env.unwrapped.num_envs),
    )
    initial_box_pose = initial.box.pose_b.clone()
    initial_fingertip_palm_offsets_m: dict[str, list[list[float]]] = {}
    initial_fingertip_box_ray_projections_m: dict[str, list[float]] = {}
    for side in ("left", "right"):
        hand = getattr(initial, f"{side}_hand")
        arm = getattr(initial, f"{side}_arm")
        offsets = hand.fingertip_positions_b - arm.palm_pose_b[:3].unsqueeze(0)
        if not bool(torch.isfinite(offsets).all().item()):
            raise RuntimeError(f"non-finite initial {side} fingertip geometry")
        box_ray = initial.box.pose_b[:3] - arm.palm_pose_b[:3]
        box_ray_norm = torch.linalg.vector_norm(box_ray)
        if not bool(torch.isfinite(box_ray_norm).item()) or float(box_ray_norm.item()) <= 0.0:
            raise RuntimeError(f"invalid initial {side} palm-to-box ray")
        initial_fingertip_palm_offsets_m[f"{side}_o6"] = (
            offsets.detach().cpu().tolist()
        )
        initial_fingertip_box_ray_projections_m[f"{side}_o6"] = (
            (offsets @ (box_ray / box_ray_norm)).detach().cpu().tolist()
        )
    initial_velocity_max = max(
        float(torch.max(torch.abs(value)).item())
        for value in (
            initial.base_state[7:13],
            initial.m1_qd,
            initial.platform_q_qd[1:],
            initial.left_arm.qd,
            initial.right_arm.qd,
            initial.left_hand.qd,
            initial.right_hand.qd,
            initial.box.twist_b,
        )
    )
    previous = initial
    phase_steps: dict[str, int] = {}
    fallback_counts = {"object_mpc": 0, "arm_mpc": 0, "left_hand_mpc": 0, "right_hand_mpc": 0, "wbc_qp": 0}
    fallback_reason_counts: dict[str, int] = {}
    feasible = {key: 0 for key in fallback_counts}
    samples = {key: 0 for key in fallback_counts}
    teacher_feasible = 0
    teacher_samples = 0
    teacher_dynamics_residual_max = 0.0
    teacher_contact_residual_max = 0.0
    teacher_action_max_abs = 0.0
    teacher_fallback_reason_counts: dict[str, int] = {}
    command_fallback_reason_counts: dict[str, int] = {}
    safety_feasible = 0
    safety_samples = 0
    hold_errors_pos: list[float] = []
    hold_errors_rot: list[float] = []
    max_lift = 0.0
    max_slip = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    max_forces = {"left_o6": 0.0, "right_o6": 0.0}
    max_o6_body_contact_forces_n = {"left_o6": 0.0, "right_o6": 0.0}
    max_o6_body_contact_links = {"left_o6": None, "right_o6": None}
    first_o6_body_contact_events = {"left_o6": None, "right_o6": None}
    max_o6_body_contact_events = {"left_o6": None, "right_o6": None}
    contact_timing_steps = {
        "left_o6": {"first": None, "last": None, "count": 0},
        "right_o6": {"first": None, "last": None, "count": 0},
    }
    bilateral_contact_steps = 0
    consecutive_bilateral_contact_steps = 0
    max_consecutive_bilateral_contact_steps = 0
    reset_count = 0
    startup_terminal_count = 0
    nonfinite_count = 0
    collision_count = 0
    limit_violation_count = 0
    max_joint_limit_violation_rad = 0.0
    first_limit_violation: dict[str, object] | None = None
    first_arm_mpc_failure: dict[str, object] | None = None
    first_object_mpc_failure: dict[str, object] | None = None
    first_safety_failure: dict[str, object] | None = None
    box_dropped = False
    min_palm_target_errors = [1.0e9, 1.0e9]
    final_palm_target_errors = [1.0e9, 1.0e9]
    min_palm_target_rotation_errors_rad = [1.0e9, 1.0e9]
    final_palm_target_rotation_errors_rad = [1.0e9, 1.0e9]
    min_fingertip_box_distances_m = {"left_o6": 1.0e9, "right_o6": 1.0e9}
    right_palm_orientation_mpc = None
    orientation_boundary_trace = []
    first_right_palm_base_contact_orientation = None
    first_right_selected_fingertip_contact_orientation = None
    prior_diagnostics = {
        "left_o6": make_probe_summary(
            configured=fingertip_prior_artifact is not None
        ),
        "right_o6": make_probe_summary(
            configured=fingertip_prior_artifact is not None
        ),
    }
    prior_solution_ids = {"left_o6": None, "right_o6": None}

    for step_index in range(steps):
        _, _, terminated, truncated, _ = wrapper.step()
        _write_progress(
            progress_path,
            executed_physics_steps=step_index + 2,
            num_envs=int(env.unwrapped.num_envs),
        )
        snapshot = wrapper.last_snapshot
        orientation_diagnostics = wrapper.runtime.object_mpc.right_orientation_mpc.last_diagnostics
        if orientation_diagnostics is not None:
            right_palm_orientation_mpc = asdict(orientation_diagnostics)
            right_palm_orientation_mpc['qp_feasible'] = orientation_diagnostics.feasible
            right_palm_orientation_mpc['qp_iterations'] = orientation_diagnostics.iterations
        command = wrapper.last_command
        safety_result = wrapper.last_safety_result
        if safety_result is not None:
            safety_samples += 1
            safety_feasible += int(safety_result.feasible)
            if not safety_result.feasible and first_safety_failure is None:
                diagnostic_ids = list(wrapper.adapter.active_joint_ids)
                diagnostic_q = wrapper.adapter.robot.data.joint_pos[0, diagnostic_ids]
                diagnostic_qd = wrapper.adapter.robot.data.joint_vel[0, diagnostic_ids]
                diagnostic_limits = wrapper.adapter.robot.data.soft_joint_pos_limits[0, diagnostic_ids]
                diagnostic_qd_max = wrapper.adapter.robot.data.soft_joint_vel_limits[0, diagnostic_ids]
                normalized_qd = torch.abs(diagnostic_qd) / diagnostic_qd_max
                joint_margin = torch.minimum(
                    diagnostic_q - diagnostic_limits[:, 0],
                    diagnostic_limits[:, 1] - diagnostic_q,
                )
                fastest = int(torch.argmax(normalized_qd).item())
                tightest = int(torch.argmin(joint_margin).item())
                first_safety_failure = {
                    "step": step_index,
                    "reason": safety_result.fallback_reason,
                    "max_velocity_fraction": float(normalized_qd[fastest].item()),
                    "fastest_joint": wrapper.adapter.robot.joint_names[diagnostic_ids[fastest]],
                    "minimum_position_margin_rad": float(joint_margin[tightest].item()),
                    "tightest_joint": wrapper.adapter.robot.joint_names[diagnostic_ids[tightest]],
                    "base_state": snapshot.base_state.tolist(),
                }
        for command_reason in command.fallback_reasons:
            command_fallback_reason_counts[command_reason] = (
                command_fallback_reason_counts.get(command_reason, 0) + 1
            )
        teacher_solution = wrapper.last_teacher_solution
        if teacher_solution is not None:
            teacher_samples += 1
            teacher_feasible += int(teacher_solution.diagnostics.feasible)
            teacher_action_max_abs = max(
                teacher_action_max_abs,
                float(torch.max(torch.abs(teacher_solution.action_trajectory)).item()),
            )
            teacher_reason = teacher_solution.diagnostics.fallback_reason
            if teacher_reason:
                teacher_fallback_reason_counts[teacher_reason] = (
                    teacher_fallback_reason_counts.get(teacher_reason, 0) + 1
                )
            teacher_dynamics_residual_max = max(
                teacher_dynamics_residual_max,
                teacher_solution.diagnostics.dynamics_residual_max,
            )
            teacher_contact_residual_max = max(
                teacher_contact_residual_max,
                teacher_solution.diagnostics.contact_residual_max,
            )
        phase = wrapper.runtime.mission.phase.name
        startup_terminal_count += int(
            step_index < 3 and phase in {"DONE", "TERMINATED"}
        )
        phase_steps[phase] = phase_steps.get(phase, 0) + 1
        reset_count += int(torch.count_nonzero(terminated | truncated).item())
        if not _finite_snapshot(snapshot) or not torch.isfinite(command.effort).all().item():
            nonfinite_count += 1
        lift = float(snapshot.box.pose_b[2] - initial_box_pose[2])
        max_lift = max(max_lift, lift)
        box_dropped = box_dropped or lift < -0.03
        roll, pitch = _roll_pitch(snapshot.base_state[3:7])
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))
        for side in ("left", "right"):
            current_offset = getattr(snapshot, f"{side}_arm").palm_pose_b[:3] - snapshot.box.pose_b[:3]
            previous_offset = getattr(previous, f"{side}_arm").palm_pose_b[:3] - previous.box.pose_b[:3]
            max_slip = max(max_slip, float(torch.linalg.vector_norm(current_offset - previous_offset)))
            forces = getattr(snapshot, f"{side}_hand").fingertip_forces_b
            max_forces[f"{side}_o6"] = max(
                max_forces[f"{side}_o6"], float(torch.linalg.vector_norm(forces, dim=1).max())
            )
            fingertip_delta = torch.abs(
                getattr(snapshot, f"{side}_hand").fingertip_positions_b
                - snapshot.box.pose_b[:3]
            ) - torch.tensor([0.10, 0.09, 0.05], dtype=torch.float64)
            fingertip_distance = torch.linalg.vector_norm(
                torch.clamp(fingertip_delta, min=0.0), dim=1
            ).min()
            min_fingertip_box_distances_m[f"{side}_o6"] = min(
                min_fingertip_box_distances_m[f"{side}_o6"],
                float(fingertip_distance.item()),
            )
            summary = wrapper.adapter.contact_summaries[side]
            body_forces = torch.linalg.vector_norm(summary.filtered_forces_w, dim=1)
            body_index = int(torch.argmax(body_forces).item())
            body_force = float(body_forces[body_index].item())
            key = f"{side}_o6"
            event = {
                "step": step_index,
                "phase": phase,
                "link": summary.selected_names[body_index],
                "force_n": body_force,
                "body_contact_forces_n": {
                    name: float(force)
                    for name, force in zip(
                        summary.selected_names,
                        body_forces.detach().cpu().tolist(),
                        strict=True,
                    )
                },
                "selected_fingertip_forces_n": torch.linalg.vector_norm(
                    getattr(snapshot, f"{side}_hand").fingertip_forces_b, dim=1
                )
                .detach()
                .cpu()
                .tolist(),
                "palm_pose_b": getattr(snapshot, f"{side}_arm").palm_pose_b.tolist(),
                "box_pose_b": snapshot.box.pose_b.tolist(),
                "fingertip_positions_b": getattr(
                    snapshot, f"{side}_hand"
                ).fingertip_positions_b.tolist(),
                "hand_q": getattr(snapshot, f"{side}_hand").q.tolist(),
            }
            if body_force > 0.2 and first_o6_body_contact_events[key] is None:
                first_o6_body_contact_events[key] = event
            if side == 'right':
                event['right_palm_orientation_mpc'] = right_palm_orientation_mpc
                base_force = event['body_contact_forces_n'].get('right_hand_base_link', 0.)
                if base_force > .2 and first_right_palm_base_contact_orientation is None:
                    first_right_palm_base_contact_orientation = {
                        'step': step_index, 'force_n': base_force,
                        'orientation': right_palm_orientation_mpc}
                if bool(snapshot.right_hand.contact_mask.any()) and first_right_selected_fingertip_contact_orientation is None:
                    first_right_selected_fingertip_contact_orientation = {
                        'step': step_index, 'orientation': right_palm_orientation_mpc}
            if body_force > max_o6_body_contact_forces_n[key]:
                max_o6_body_contact_forces_n[key] = body_force
                max_o6_body_contact_links[key] = summary.selected_names[body_index]
                max_o6_body_contact_events[key] = event
            if bool(getattr(snapshot, f"{side}_hand").contact_mask.any()):
                timing = contact_timing_steps[key]
                if timing["first"] is None:
                    timing["first"] = step_index
                timing["last"] = step_index
                timing["count"] += 1
        bilateral_now = bool(
            snapshot.left_hand.contact_mask.any()
            and snapshot.right_hand.contact_mask.any()
        )
        bilateral_contact_steps += int(bilateral_now)
        consecutive_bilateral_contact_steps = (
            consecutive_bilateral_contact_steps + 1 if bilateral_now else 0
        )
        max_consecutive_bilateral_contact_steps = max(
            max_consecutive_bilateral_contact_steps,
            consecutive_bilateral_contact_steps,
        )
        if phase == "HOLD":
            target = initial_box_pose.clone()
            target[2] += 0.10
            hold_errors_pos.append(float(torch.linalg.vector_norm(snapshot.box.pose_b[:3] - target[:3])))
            hold_errors_rot.append(float(torch.linalg.vector_norm(snapshot.box.pose_b[3:] - target[3:])))

        latest = wrapper.runtime.latest_solutions
        object_solution = latest["object"]
        if object_solution is not None and step_index <= 352 and step_index % 8 == 0:
            from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import (
                rotvec_to_matrix, matrix_to_rotvec,
            )
            current_rotation = snapshot.right_arm.palm_pose_b[3:]
            targets = object_solution.right_palm_pose[:, 3:]
            previous_rotations = torch.cat((current_rotation[None], targets[:-1]))
            geometric_steps = torch.stack([
                matrix_to_rotvec(rotvec_to_matrix(after) @ rotvec_to_matrix(before).T)
                for before, after in zip(previous_rotations, targets, strict=True)
            ])
            raw_steps = targets - previous_rotations
            orientation_boundary_trace.append({
                'step': step_index,
                'phase': phase,
                'current_rotvec_b': current_rotation.tolist(),
                'first_target_rotvec_b': targets[0].tolist(),
                'raw_first_delta': raw_steps[0].tolist(),
                'geometric_first_delta_b': geometric_steps[0].tolist(),
                'raw_geometric_first_delta_difference_rad': float((raw_steps[0] - geometric_steps[0]).norm()),
                'max_future_geometric_rate_rad_s': float(geometric_steps[1:].norm(dim=1).max() / .04),
                'first_tracking_geometric_rate_rad_s': float(geometric_steps[0].norm() / .04),
                'right_qd_max': float(snapshot.right_arm.qd.abs().max()),
                'orientation_plan': right_palm_orientation_mpc,
            })
        arm_solution = latest["arm"]
        left_hand = latest["left_hand"]
        right_hand = latest["right_hand"]
        for key, solution in (("left_o6", left_hand), ("right_o6", right_hand)):
            if solution is None or prior_solution_ids[key] == id(solution):
                continue
            prior_solution_ids[key] = id(solution)
            accumulate_probe_prior_diagnostics(
                prior_diagnostics[key], solution.diagnostics
            )
        statuses = {
            "object_mpc": object_solution is not None and object_solution.diagnostics.feasible,
            "arm_mpc": arm_solution is not None and arm_solution.both_feasible,
            "left_hand_mpc": left_hand is not None and left_hand.diagnostics.feasible,
            "right_hand_mpc": right_hand is not None and right_hand.diagnostics.feasible,
            "wbc_qp": latest["wbc"] is not None and latest["wbc"].feasible,
        }
        if object_solution is not None and not object_solution.diagnostics.feasible and first_object_mpc_failure is None:
            first_object_mpc_failure = {
                "step": step_index,
                "reason": object_solution.diagnostics.fallback_reason,
                "box_pose_b": snapshot.box.pose_b.tolist(),
                "box_twist_b": snapshot.box.twist_b.tolist(),
            }
        if arm_solution is not None and not arm_solution.both_feasible and first_arm_mpc_failure is None:
            first_arm_mpc_failure = {
                "step": step_index,
                "left_reason": arm_solution.left.diagnostics.fallback_reason,
                "right_reason": arm_solution.right.diagnostics.fallback_reason,
                "left_min_joint_margin": arm_solution.left.diagnostics.min_joint_margin,
                "right_min_joint_margin": arm_solution.right.diagnostics.min_joint_margin,
                "left_qd_max": float(torch.max(torch.abs(snapshot.left_arm.qd)).item()),
                "right_qd_max": float(torch.max(torch.abs(snapshot.right_arm.qd)).item()),
            }
        reasons = (
            None if object_solution is None else object_solution.diagnostics.fallback_reason,
            None if arm_solution is None or arm_solution.both_feasible else (
                arm_solution.left.diagnostics.fallback_reason
                or arm_solution.right.diagnostics.fallback_reason
                or "dual_arm_infeasible"
            ),
            None if left_hand is None else left_hand.diagnostics.fallback_reason,
            None if right_hand is None else right_hand.diagnostics.fallback_reason,
            None if latest["wbc"] is None else latest["wbc"].diagnostics.fallback_reason,
        )
        for reason in reasons:
            if reason:
                fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
        if object_solution is not None:
            final_palm_target_errors = [
                float(torch.linalg.vector_norm(snapshot.left_arm.palm_pose_b[:3] - object_solution.left_palm_pose[-1, :3])),
                float(torch.linalg.vector_norm(snapshot.right_arm.palm_pose_b[:3] - object_solution.right_palm_pose[-1, :3])),
            ]
            min_palm_target_errors = [
                min(old, new) for old, new in zip(min_palm_target_errors, final_palm_target_errors)
            ]
            final_palm_target_rotation_errors_rad = [
                float(torch.linalg.vector_norm(snapshot.left_arm.palm_pose_b[3:] - object_solution.left_palm_pose[-1, 3:])),
                float(torch.linalg.vector_norm(snapshot.right_arm.palm_pose_b[3:] - object_solution.right_palm_pose[-1, 3:])),
            ]
            min_palm_target_rotation_errors_rad = [
                min(old, new)
                for old, new in zip(
                    min_palm_target_rotation_errors_rad,
                    final_palm_target_rotation_errors_rad,
                )
            ]
        for layer, ok in statuses.items():
            samples[layer] += 1
            feasible[layer] += int(ok)
            fallback_counts[layer] += int(not ok)
        collision_count += int("collision" in command.fallback_reasons)
        active_ids = list(wrapper.adapter.active_joint_ids)
        positions = wrapper.adapter.robot.data.joint_pos[0, active_ids]
        limits = wrapper.adapter.robot.data.soft_joint_pos_limits[0, active_ids]
        violations = torch.maximum(
            torch.clamp(limits[:, 0] - positions, min=0.0),
            torch.clamp(positions - limits[:, 1], min=0.0),
        )
        step_max_violation = float(torch.max(violations).item())
        max_joint_limit_violation_rad = max(
            max_joint_limit_violation_rad, step_max_violation
        )
        if step_max_violation > 0.0:
            limit_violation_count += 1
            if first_limit_violation is None:
                local_id = int(torch.argmax(violations).item())
                joint_id = active_ids[local_id]
                first_limit_violation = {
                    "step": step_index,
                    "joint": wrapper.adapter.robot.joint_names[joint_id],
                    "position": float(positions[local_id].item()),
                    "minimum": float(limits[local_id, 0].item()),
                    "maximum": float(limits[local_id, 1].item()),
                    "violation": step_max_violation,
                }
        previous = snapshot

    rates = {key: feasible[key] / max(samples[key], 1) for key in feasible}
    released_supported = bool(wrapper.runtime.mission.phase.name == "DONE" and previous.box.supported)
    hard_failure_count = (
        reset_count
        + startup_terminal_count
        + nonfinite_count
        + collision_count
        + limit_violation_count
        + int(box_dropped)
    )
    box_half_extents = torch.tensor(
        [0.10, 0.09, 0.05],
        dtype=previous.box.pose_b.dtype,
        device=previous.box.pose_b.device,
    )
    final_fingertip_box_offsets_m: dict[str, list[list[float]]] = {}
    final_fingertip_box_distances_m: dict[str, list[float]] = {}
    for side in ("left", "right"):
        key = f"{side}_o6"
        offsets = (
            getattr(previous, f"{side}_hand").fingertip_positions_b
            - previous.box.pose_b[:3]
        )
        outside = torch.clamp(torch.abs(offsets) - box_half_extents, min=0.0)
        final_fingertip_box_offsets_m[key] = offsets.tolist()
        final_fingertip_box_distances_m[key] = torch.linalg.vector_norm(
            outside, dim=1
        ).tolist()
    row: dict[str, object] = {
        "seed": seed,
        "trial_index": trial_index,
        "steps": steps,
        "final_phase": wrapper.runtime.mission.phase.name,
        "initial_box_pose_b": initial_box_pose.tolist(),
        "initial_left_palm_pose_b": initial.left_arm.palm_pose_b.tolist(),
        "initial_right_palm_pose_b": initial.right_arm.palm_pose_b.tolist(),
        "initial_fingertip_palm_offsets_m": initial_fingertip_palm_offsets_m,
        "initial_fingertip_box_ray_projections_m": (
            initial_fingertip_box_ray_projections_m
        ),
        "initial_velocity_max": initial_velocity_max,
        "startup_terminal_count": startup_terminal_count,
        "phase_dwell_times_s": {key: value * 0.005 for key, value in sorted(phase_steps.items())},
        "lift_height_m": max_lift,
        "hold_duration_s": phase_steps.get("HOLD", 0) * 0.005,
        "hold_position_error_m": max(hold_errors_pos, default=1.0e9),
        "hold_orientation_error_rad": max(hold_errors_rot, default=1.0e9),
        "relative_palm_slip_m": max_slip,
        "object_mpc_feasible_rate": rates["object_mpc"],
        "arm_mpc_feasible_rates": [rates["arm_mpc"], rates["arm_mpc"]],
        "hand_mpc_feasible_rates": [rates["left_hand_mpc"], rates["right_hand_mpc"]],
        "wbc_qp_feasible_rate": rates["wbc_qp"],
        "teacher_feasible_rate": teacher_feasible / max(teacher_samples, 1),
        "teacher_dynamics_residual_max": teacher_dynamics_residual_max,
        "teacher_contact_residual_max": teacher_contact_residual_max,
        "teacher_action_max_abs": teacher_action_max_abs,
        "teacher_fallback_reason_counts": teacher_fallback_reason_counts,
        "command_fallback_reason_counts": command_fallback_reason_counts,
        "safety_feasible_rate": safety_feasible / max(safety_samples, 1),
        "fallback_counts": fallback_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "min_palm_target_errors": min_palm_target_errors,
        "final_palm_target_errors": final_palm_target_errors,
        "min_palm_target_rotation_errors_rad": min_palm_target_rotation_errors_rad,
        "final_palm_target_rotation_errors_rad": final_palm_target_rotation_errors_rad,
        "min_fingertip_box_distances_m": min_fingertip_box_distances_m,
        "final_fingertip_box_distances_m": final_fingertip_box_distances_m,
        "final_fingertip_box_offsets_m": final_fingertip_box_offsets_m,
        "final_left_hand_q": previous.left_hand.q.tolist(),
        "final_right_hand_q": previous.right_hand.q.tolist(),
        "arm_dynamics_diagnostics": wrapper.adapter.arm_dynamics_diagnostics,
        "final_base_state": previous.base_state.tolist(),
        "final_box_pose_b": previous.box.pose_b.tolist(),
        "max_contact_forces_n": max_forces,
        "max_o6_body_contact_forces_n": max_o6_body_contact_forces_n,
        "max_o6_body_contact_links": max_o6_body_contact_links,
        "first_o6_body_contact_events": first_o6_body_contact_events,
        "right_palm_orientation_mpc": right_palm_orientation_mpc,
        "orientation_boundary_trace": orientation_boundary_trace,
        "first_right_palm_base_contact_orientation": first_right_palm_base_contact_orientation,
        "first_right_selected_fingertip_contact_orientation": first_right_selected_fingertip_contact_orientation,
        "max_o6_body_contact_events": max_o6_body_contact_events,
        "contact_timing_steps": contact_timing_steps,
        "bilateral_contact_steps": bilateral_contact_steps,
        "max_consecutive_bilateral_contact_steps": max_consecutive_bilateral_contact_steps,
        "final_contact_latched_hand_q": {
            "left_o6": [
                float(value) if math.isfinite(float(value)) else None
                for value in wrapper._left_hand_contact_q
            ],
            "right_o6": [
                float(value) if math.isfinite(float(value)) else None
                for value in wrapper._right_hand_contact_q
            ],
        },
        "fingertip_jacobian_norms": {
            "left_o6": float(
                torch.linalg.vector_norm(
                    previous.left_hand.fingertip_jacobian_b
                ).item()
            ),
            "right_o6": float(
                torch.linalg.vector_norm(
                    previous.right_hand.fingertip_jacobian_b
                ).item()
            ),
        },
        "max_abs_roll_rad": max_roll,
        "max_abs_pitch_rad": max_pitch,
        "collision_count": collision_count,
        "limit_violation_count": limit_violation_count,
        "max_joint_limit_violation_rad": max_joint_limit_violation_rad,
        "first_limit_violation": first_limit_violation,
        "first_arm_mpc_failure": first_arm_mpc_failure,
        "first_object_mpc_failure": first_object_mpc_failure,
        "first_safety_failure": first_safety_failure,
        "reset_count": reset_count,
        "nonfinite_count": nonfinite_count,
        "hard_failure_count": hard_failure_count,
        "box_dropped": box_dropped,
        "released_supported": released_supported,
        "fingertip_prior": {
            key: finalize_probe_prior_summary(value)
            for key, value in prior_diagnostics.items()
        },
    }
    row["passed"] = trial_passes(row)
    wrapper.close(close_env=False)
    return row


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--trials-per-seed", type=int, default=DEFAULT_TRIALS_PER_SEED)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("teacher", "latent"), default="teacher")
    parser.add_argument("--latent-artifact", type=Path, default=None)
    parser.add_argument("--fingertip-prior-artifact", type=Path, default=None)
    parser.add_argument(
        "--tracking-angular-rate-max-rad-s",
        type=_positive_finite_float,
        default=0.35,
        help="SO(3) target tracking rate cap applied by both arm MPCs",
    )
    # Help is deliberately usable on a host without Isaac or an artifact.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        parser.add_argument("--headless", action="store_true")
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    formal = args.seeds is not None
    if formal and any(path is None for path in (args.report, args.jsonl, args.manifest)):
        parser.error("formal mode requires --report, --jsonl, and --manifest")
    if not formal and (args.jsonl is not None or args.manifest is not None):
        parser.error("--jsonl and --manifest are reserved for formal mode with --seeds")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import go2_pvcnn.tasks  # noqa: F401
    from go2_pvcnn.assets.m1_dual_panda_o6 import M1_DUAL_PANDA_O6_USD_PATH
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import M1DualPandaO6BimanualEnvCfg
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import M1DualPandaO6BimanualWrapper
    from go2_pvcnn.control.m1_bimanual_coordination.dual_arm_mpc import DualArmMpcCoordinator
    from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime

    root = Path(__file__).resolve().parents[1]
    seeds = tuple(args.seeds) if formal else (args.seed,)
    trials_per_seed = args.trials_per_seed if formal else 1
    steps = args.steps if args.steps is not None else (DEFAULT_FORMAL_STEPS if formal else 1)
    torch.manual_seed(seeds[0])
    cfg = M1DualPandaO6BimanualEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = seeds[0]
    env = gym.make(GYM_ID, cfg=cfg)
    trials = [
        _run_trial(
            env,
            M1DualPandaO6BimanualWrapper,
            seed=seed,
            trial_index=index,
            steps=steps,
            mode=args.mode,
            latent_artifact=args.latent_artifact,
            fingertip_prior_artifact=args.fingertip_prior_artifact,
            runtime_factory=lambda: BimanualRuntime(
                arm_mpc=DualArmMpcCoordinator(
                    first_target_angular_rate_max_rad_s=(
                        args.tracking_angular_rate_max_rad_s
                    )
                )
            ),
            progress_path=args.progress,
        )
        for seed in seeds
        for index in range(trials_per_seed)
    ]
    aggregate = aggregate_acceptance(trials, seeds=seeds, trials_per_seed=trials_per_seed)
    smoke_passed = all(
        row["nonfinite_count"] == 0
        and row["reset_count"] == 0
        and row["startup_terminal_count"] == 0
        and row["initial_velocity_max"] <= 1.0e-12
        for row in trials
    )
    metadata = {
        "asset_sha256": _sha256(Path(M1_DUAL_PANDA_O6_USD_PATH)),
        "source_sha256": _source_sha256(root),
        "git_ref": _git_ref(root),
        "isaac_version": importlib.metadata.version("isaacsim"),
        "command": " ".join(sys.argv),
        "tracking_angular_rate_max_rad_s": args.tracking_angular_rate_max_rad_s,
        "fingertip_prior_artifact": (
            None
            if args.fingertip_prior_artifact is None
            else str(args.fingertip_prior_artifact)
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "acceptance_mode": "formal" if formal else "smoke",
        "action_dim": int(env.unwrapped.action_manager.total_action_dim),
        "num_envs": int(args.num_envs),
        "reset_physics_steps_per_trial": 1,
        "finite_snapshot": all(row["nonfinite_count"] == 0 for row in trials),
        "unexpected_reset_count": sum(int(row["reset_count"]) for row in trials),
        "trials": trials,
        "aggregate": aggregate,
        "metadata": metadata,
    }
    report["passed"] = bool(aggregate["accepted"] if formal else smoke_passed and report["action_dim"] == 43)
    if args.report is not None:
        _atomic_json(args.report, report)
    if formal:
        assert args.report is not None and args.jsonl is not None and args.manifest is not None
        _atomic_jsonl(args.jsonl, trials)
        _atomic_json(
            args.manifest,
            _artifact_manifest(report, report_path=args.report, jsonl_path=args.jsonl),
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    env.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
