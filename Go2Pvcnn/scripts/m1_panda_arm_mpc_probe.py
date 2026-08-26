#!/usr/bin/env python3
"""GPU0 Phase-5 physical gate for stationary M1 plus small Panda EE motion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import torch


TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"


def build_arg_parser(*, include_app_launcher_args: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--trajectory-scale", type=float, default=0.25)
    parser.add_argument("--summary-json", type=Path, default=None)
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
    mpc_fallback_count: int = 0
    exit_reason: str = "not_started"

    @property
    def force_direction_cosine(self) -> float:
        return self.force_direction_cosine_sum / max(self.eligible_force_samples, 1)

    @property
    def moment_direction_cosine(self) -> float:
        return self.moment_direction_cosine_sum / max(self.eligible_moment_samples, 1)

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
            "mpc_fallback_count": self.mpc_fallback_count,
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


def _update(summary, runtime, terminated, truncated):
    snapshot = runtime.diagnostics_snapshot()
    state = runtime.states[0]
    solution = runtime._solutions[0]
    measured = snapshot["measured_mount_wrench_b"][0]
    predicted = snapshot["predicted_mount_wrench_b"][0]
    error = snapshot["target_pose"][0] - state.ee_pose
    summary.steps += 1
    summary.mpc_feasible_count += int(snapshot["mpc_feasible"][0])
    summary.qp_feasible_count += int(snapshot["qp_feasible"][0])
    summary.mpc_fallback_count += int(snapshot["mpc_fallback"][0])
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
    if moment_norm >= 0.2:
        summary.eligible_moment_samples += 1
        summary.moment_direction_cosine_sum += _cosine(predicted[3:], measured[3:])
    tensors = (measured, predicted, snapshot["effort"], state.ee_pose)
    summary.finite = summary.finite and all(torch.isfinite(value).all().item() for value in tensors)


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    summaries = []
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
                env = gym.make(args.task, cfg=cfg).unwrapped
                wrapper = M1PandaArmMpcResidualEnvWrapper(
                    env, seed=seed, trajectory_scale=args.trajectory_scale
                )
                wrapper.reset()
                action = torch.zeros((1, 8), device=wrapper.device)
                for step in range(args.steps):
                    if not simulation_app.is_running():
                        summary.exit_reason = "application_closed"
                        break
                    _, _, dones, extras = wrapper.step(action)
                    _update(
                        summary,
                        wrapper.runtime,
                        extras["terminated"],
                        extras["truncated"],
                    )
                    if bool(dones.any().item()):
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
        print("[Phase5 aggregate] " + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if payload["accepted"] else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
