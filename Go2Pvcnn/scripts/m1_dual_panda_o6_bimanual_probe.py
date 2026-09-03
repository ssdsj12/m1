"""Smoke and formal fixed-condition acceptance for inline bimanual MPC."""

from __future__ import annotations

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
from typing import Any, Sequence


GYM_ID = "Isaac-M1-DualPanda-O6-Bimanual-Lift-v0"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TRIALS_PER_SEED = 10
DEFAULT_FORMAL_STEPS = 2000


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
    ]
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_ref(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


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


def _run_trial(env, wrapper_type, *, seed: int, trial_index: int, steps: int) -> dict[str, object]:
    import torch

    wrapper = wrapper_type(env)
    initial = wrapper.reset(seed=seed)
    initial_box_pose = initial.box.pose_b.clone()
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
    hold_errors_pos: list[float] = []
    hold_errors_rot: list[float] = []
    max_lift = 0.0
    max_slip = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    max_forces = {"left_o6": 0.0, "right_o6": 0.0}
    reset_count = 0
    startup_terminal_count = 0
    nonfinite_count = 0
    collision_count = 0
    limit_violation_count = 0
    box_dropped = False
    min_palm_target_errors = [1.0e9, 1.0e9]
    final_palm_target_errors = [1.0e9, 1.0e9]

    for step_index in range(steps):
        _, _, terminated, truncated, _ = wrapper.step()
        snapshot = wrapper.last_snapshot
        command = wrapper.last_command
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
        if phase == "HOLD":
            target = initial_box_pose.clone()
            target[2] += 0.10
            hold_errors_pos.append(float(torch.linalg.vector_norm(snapshot.box.pose_b[:3] - target[:3])))
            hold_errors_rot.append(float(torch.linalg.vector_norm(snapshot.box.pose_b[3:] - target[3:])))

        latest = wrapper.runtime.latest_solutions
        object_solution = latest["object"]
        arm_solution = latest["arm"]
        left_hand = latest["left_hand"]
        right_hand = latest["right_hand"]
        statuses = {
            "object_mpc": object_solution is not None and object_solution.diagnostics.feasible,
            "arm_mpc": arm_solution is not None and arm_solution.both_feasible,
            "left_hand_mpc": left_hand is not None and left_hand.diagnostics.feasible,
            "right_hand_mpc": right_hand is not None and right_hand.diagnostics.feasible,
            "wbc_qp": command.feasible,
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
                float(torch.linalg.vector_norm(snapshot.left_arm.palm_pose_b - object_solution.left_palm_pose[0])),
                float(torch.linalg.vector_norm(snapshot.right_arm.palm_pose_b - object_solution.right_palm_pose[0])),
            ]
            min_palm_target_errors = [
                min(old, new) for old, new in zip(min_palm_target_errors, final_palm_target_errors)
            ]
        for layer, ok in statuses.items():
            samples[layer] += 1
            feasible[layer] += int(ok)
            fallback_counts[layer] += int(not ok)
        collision_count += int("collision" in command.fallback_reasons)
        active_ids = list(wrapper.adapter.active_joint_ids)
        positions = wrapper.adapter.robot.data.joint_pos[0, active_ids]
        limits = wrapper.adapter.robot.data.soft_joint_pos_limits[0, active_ids]
        limit_violation_count += int(torch.any((positions < limits[:, 0]) | (positions > limits[:, 1])).item())
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
    row: dict[str, object] = {
        "seed": seed,
        "trial_index": trial_index,
        "steps": steps,
        "final_phase": wrapper.runtime.mission.phase.name,
        "initial_box_pose_b": initial_box_pose.tolist(),
        "initial_left_palm_pose_b": initial.left_arm.palm_pose_b.tolist(),
        "initial_right_palm_pose_b": initial.right_arm.palm_pose_b.tolist(),
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
        "fallback_counts": fallback_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "min_palm_target_errors": min_palm_target_errors,
        "final_palm_target_errors": final_palm_target_errors,
        "arm_dynamics_diagnostics": wrapper.adapter.arm_dynamics_diagnostics,
        "final_base_state": previous.base_state.tolist(),
        "final_box_pose_b": previous.box.pose_b.tolist(),
        "max_contact_forces_n": max_forces,
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
        "reset_count": reset_count,
        "nonfinite_count": nonfinite_count,
        "hard_failure_count": hard_failure_count,
        "box_dropped": box_dropped,
        "released_supported": released_supported,
    }
    row["passed"] = trial_passes(row)
    return row


def _parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--trials-per-seed", type=int, default=DEFAULT_TRIALS_PER_SEED)
    parser.add_argument("--report", type=Path)
    # AppLauncher supplies the standard --headless flag.
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = _parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import go2_pvcnn.tasks  # noqa: F401
    from go2_pvcnn.assets.m1_dual_panda_o6 import M1_DUAL_PANDA_O6_USD_PATH
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import M1DualPandaO6BimanualEnvCfg
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import M1DualPandaO6BimanualWrapper

    root = Path(__file__).resolve().parents[1]
    formal = args.seeds is not None
    seeds = tuple(args.seeds) if formal else (args.seed,)
    trials_per_seed = args.trials_per_seed if formal else 1
    steps = args.steps if args.steps is not None else (DEFAULT_FORMAL_STEPS if formal else 1)
    if args.num_envs != 1:
        parser.error("formal snapshot/MPC execution currently requires --num-envs 1")
    torch.manual_seed(seeds[0])
    cfg = M1DualPandaO6BimanualEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = seeds[0]
    env = gym.make(GYM_ID, cfg=cfg)
    trials = [
        _run_trial(env, M1DualPandaO6BimanualWrapper, seed=seed, trial_index=index, steps=steps)
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
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "formal" if formal else "smoke",
        "action_dim": int(env.unwrapped.action_manager.total_action_dim),
        "finite_snapshot": all(row["nonfinite_count"] == 0 for row in trials),
        "unexpected_reset_count": sum(int(row["reset_count"]) for row in trials),
        "trials": trials,
        "aggregate": aggregate,
        "metadata": metadata,
    }
    report["passed"] = bool(aggregate["accepted"] if formal else smoke_passed and report["action_dim"] == 43)
    if args.report is not None:
        _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    env.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
