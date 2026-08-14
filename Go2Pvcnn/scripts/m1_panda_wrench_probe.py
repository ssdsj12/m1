#!/usr/bin/env python3
"""Run a deterministic six-axis known-load probe on the combined M1/Panda asset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from go2_pvcnn.tasks.m1_panda_teacher import (
    base_wrench_to_body_local as _base_wrench_to_body_local,
    clear_external_wrench as _clear_external_wrench,
)

TASK_ID = "Isaac-M1-Panda-Smoke-v0"
SETTLE_STEPS = 100
BASELINE_STEPS = 50
TRANSITION_STEPS = 10
SAMPLE_STEPS = 50
EXPECTED_REACTION_SIGN = -1

CASES = {
    "force_x": ([20.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "force_y": ([0.0, 20.0, 0.0], [0.0, 0.0, 0.0]),
    "force_z": ([0.0, 0.0, 20.0], [0.0, 0.0, 0.0]),
    "torque_x": ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
    "torque_y": ([0.0, 0.0, 0.0], [0.0, 5.0, 0.0]),
    "torque_z": ([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]),
}


def _evaluate_channel(
    samples: torch.Tensor,
    baseline: torch.Tensor,
    *,
    channel: int,
    applied_magnitude: float,
    expected_sign: int,
) -> dict[str, object]:
    """Evaluate the excited channel using every sample and the strict 20% routing gate."""
    corrected = samples - baseline.unsqueeze(0)
    excited = corrected[:, channel]
    signed = excited * float(expected_sign)
    finite = bool(torch.isfinite(samples).all().item() and torch.isfinite(baseline).all().item())
    sign_count = int((signed > 0.0).sum().item())
    sign_fraction = sign_count / int(samples.shape[0])
    stable_sign = sign_fraction >= 0.90
    corrected_mean = corrected.mean(dim=0)
    mean_expected_sign = bool((corrected_mean[channel] * float(expected_sign) > 0.0).item())
    magnitude_ratio = float(abs(corrected_mean[channel].item()) / applied_magnitude)
    return {
        "measured_mean": samples.mean(dim=0).tolist(),
        "baseline_subtracted": corrected_mean.tolist(),
        "expected_sign": expected_sign,
        "mean_expected_sign": mean_expected_sign,
        "stable_sign": stable_sign,
        "sign_count": sign_count,
        "sign_fraction": sign_fraction,
        "magnitude_ratio": magnitude_ratio,
        "finite": finite,
        "pass": finite and mean_expected_sign and stable_sign and magnitude_ratio > 0.2,
    }


def _exact_body_id(robot, body_name: str) -> int:
    ids, names = robot.find_bodies(body_name, preserve_order=True)
    if len(ids) != 1 or names != [body_name]:
        raise RuntimeError(f"Expected exactly one body named {body_name!r}, got ids={ids!r}, names={names!r}")
    return int(ids[0])


def _check_finite(label: str, values: torch.Tensor) -> None:
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError(f"Non-finite data in {label}: {values}")


def _check_no_reset(
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    label: str,
    details: dict[str, object] | None = None,
) -> None:
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError(f"Unexpected environment reset during {label}: {details=}")


def _prepare_independent_window(*, label: str, clear, reset, validate, collect_clear):
    """Reset and re-equilibrate one independent settle/baseline window."""
    clear()
    reset_result = reset()
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise RuntimeError(f"Environment reset failed for {label}: {reset_result!r}")
    validate()
    settle_samples = collect_clear(SETTLE_STEPS, f"{label} settle")
    baseline_samples = collect_clear(BASELINE_STEPS, f"{label} baseline")
    baseline = baseline_samples.mean(dim=0)
    _check_finite(f"{label} settle", settle_samples)
    _check_finite(f"{label} baseline", baseline)
    return settle_samples, baseline


def _write_jsonl_atomic(output: Path, rows: list[dict[str, object]]) -> None:
    """Write a complete JSONL artifact and atomically publish it in the target directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _run_probe(args) -> list[dict[str, object]]:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym

        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.utils import parse_env_cfg

        from go2_pvcnn.assets.m1_panda import M1_PANDA_DOF_COUNT
        from go2_pvcnn.mdp import m1_panda_mount_wrench_b

        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
        env_cfg.scene.num_envs = 1
        env_cfg.seed = 0
        # Independent settle/baseline windows make the probe roughly 1,400 control steps.
        env_cfg.episode_length_s = 60.0
        env = gym.make(TASK_ID, cfg=env_cfg)
        base_env = env.unwrapped
        if base_env.num_envs != 1:
            raise RuntimeError(f"Expected one environment, got {base_env.num_envs}")
        robot = base_env.scene["robot"]
        if robot.num_joints != M1_PANDA_DOF_COUNT:
            raise RuntimeError(f"Expected {M1_PANDA_DOF_COUNT} DOF, got {robot.num_joints}")

        hand_id = _exact_body_id(robot, "panda_hand")
        mount_id = _exact_body_id(robot, "panda_link0")
        base_id = _exact_body_id(robot, "BASE_LINK")
        if len({hand_id, mount_id, base_id}) != 3:
            raise RuntimeError(f"Required bodies did not resolve to distinct IDs: {hand_id}, {mount_id}, {base_id}")

        action_dim = base_env.action_manager.total_action_dim
        if action_dim != 16:
            raise RuntimeError(f"Expected 16 M1 actions, got {action_dim}")
        zero_actions = torch.zeros((1, action_dim), device=base_env.device)
        wrench_asset_cfg = SceneEntityCfg("robot")
        def step_and_measure(label: str) -> torch.Tensor:
            _, _, terminated, truncated, _ = env.step(zero_actions)
            reset_details = None
            if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                reset_details = {
                    "terminated": torch.as_tensor(terminated).detach().cpu().tolist(),
                    "truncated": torch.as_tensor(truncated).detach().cpu().tolist(),
                }
                for term_name in ("base_contact", "bad_orientation", "time_out"):
                    try:
                        term_value = base_env.termination_manager.get_term(term_name)
                    except Exception as error:
                        reset_details[term_name] = f"unavailable: {error}"
                    else:
                        reset_details[term_name] = torch.as_tensor(term_value).detach().cpu().tolist()
            _check_no_reset(terminated, truncated, label, details=reset_details)
            wrench = m1_panda_mount_wrench_b(base_env, wrench_asset_cfg)[0].detach().clone()
            _check_finite(label, wrench)
            return wrench

        def collect_clear(steps: int, label: str) -> torch.Tensor:
            _clear_external_wrench(robot)
            return torch.stack([step_and_measure(label) for _ in range(steps)])

        def collect_loaded(force_values: list[float], torque_values: list[float], label: str) -> torch.Tensor:
            force_b = torch.tensor(force_values, device=robot.device).reshape(1, 3)
            torque_b = torch.tensor(torque_values, device=robot.device).reshape(1, 3)
            samples = []
            for step in range(TRANSITION_STEPS + SAMPLE_STEPS):
                # Recompute from live poses every step: panda_hand axes need not align with BASE_LINK.
                force_h, torque_h = _base_wrench_to_body_local(
                    force_b,
                    torque_b,
                    robot.data.body_quat_w[:, base_id],
                    robot.data.body_quat_w[:, hand_id],
                )
                robot.set_external_force_and_torque(
                    force_h.unsqueeze(1), torque_h.unsqueeze(1), body_ids=[hand_id]
                )
                measured = step_and_measure(label)
                if step >= TRANSITION_STEPS:
                    samples.append(measured)
            return torch.stack(samples)

        def validate_robot_after_reset() -> None:
            if robot.num_joints != M1_PANDA_DOF_COUNT:
                raise RuntimeError(f"Expected {M1_PANDA_DOF_COUNT} DOF after reset, got {robot.num_joints}")
            current_ids = (
                _exact_body_id(robot, "panda_hand"),
                _exact_body_id(robot, "panda_link0"),
                _exact_body_id(robot, "BASE_LINK"),
            )
            if current_ids != (hand_id, mount_id, base_id) or len(set(current_ids)) != 3:
                raise RuntimeError(f"Body IDs changed or collided after reset: {current_ids}")

        settle_samples, baseline = _prepare_independent_window(
            label="settle",
            clear=lambda: _clear_external_wrench(robot),
            reset=env.reset,
            validate=validate_robot_after_reset,
            collect_clear=collect_clear,
        )
        rows: list[dict[str, object]] = [
            {
                "case": "settle",
                "steps": SETTLE_STEPS,
                "baseline_steps": BASELINE_STEPS,
                "measured_mean": settle_samples.mean(dim=0).tolist(),
                "baseline_mean": baseline.tolist(),
                "dof_count": int(robot.num_joints),
                "body_ids": {"panda_hand": hand_id, "panda_link0": mount_id, "BASE_LINK": base_id},
                "finite": True,
                "unexpected_reset": False,
                "pass": True,
            }
        ]

        for case_index, (case_name, (force_values, torque_values)) in enumerate(CASES.items()):
            _, baseline = _prepare_independent_window(
                label=case_name,
                clear=lambda: _clear_external_wrench(robot),
                reset=env.reset,
                validate=validate_robot_after_reset,
                collect_clear=collect_clear,
            )
            samples = collect_loaded(force_values, torque_values, case_name)
            channel = case_index
            applied = force_values + torque_values
            applied_magnitude = abs(float(applied[channel]))
            evaluation = _evaluate_channel(
                samples,
                baseline,
                channel=channel,
                applied_magnitude=applied_magnitude,
                expected_sign=EXPECTED_REACTION_SIGN,
            )
            row = {
                "case": case_name,
                "steps": SAMPLE_STEPS,
                "transition_steps": TRANSITION_STEPS,
                "sample_steps": SAMPLE_STEPS,
                "baseline_steps": BASELINE_STEPS,
                "applied_frame": "BASE_LINK",
                "applied": applied,
                "applied_force_b": force_values,
                "applied_torque_b": torque_values,
                "excited_channel": channel,
                "finite": True,
                "unexpected_reset": False,
                **evaluation,
            }
            rows.append(row)
            if not row["pass"]:
                raise RuntimeError(f"Channel check failed for {case_name}: {row}")

        _clear_external_wrench(robot)
        if len(rows) != 7:
            raise RuntimeError(f"Expected seven output rows, got {len(rows)}")
        _write_jsonl_atomic(args.output, rows)
        print(json.dumps({"output": str(args.output), "rows": len(rows), "pass": True}), flush=True)
        return rows
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit shutdown can mask an ordinary return code on this installation.
        os._exit(1)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> int:
    args = build_arg_parser().parse_args()
    _run_probe(args)
    return 0


if __name__ == "__main__":
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
