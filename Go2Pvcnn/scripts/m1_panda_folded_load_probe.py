#!/usr/bin/env python3
"""Probe the real folded-load articulation with zero policy actions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASK_ID = "Isaac-M1-Panda-Folded-Load-v0"


def evaluate_probe(
    diagnostics: dict[str, float], *, finite_state: bool, physics_steps: int
) -> dict[str, object]:
    required = (
        "inactive_action_max",
        "fold_error_max",
        "effort_utilization_max",
        "joint_limit_proximity_min",
        "mount_wrench_norm_max",
    )
    finite_diagnostics = all(
        key in diagnostics and math.isfinite(float(diagnostics[key]))
        for key in required
    )
    checks = {
        "physics_step_executed": int(physics_steps) >= 1,
        "finite_state": bool(finite_state) and finite_diagnostics,
        "inactive_action_exact_zero": diagnostics.get("inactive_action_max") == 0.0,
        "fold_error_within_limit": diagnostics.get("fold_error_max", math.inf) <= 0.35,
        "effort_within_asset_limit": diagnostics.get("effort_utilization_max", math.inf) <= 1.0,
        "joint_limit_margin": diagnostics.get("joint_limit_proximity_min", -math.inf) > 0.01,
        "mount_response_present": diagnostics.get("mount_wrench_norm_max", 0.0) > 1.0e-6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "physics_steps": int(physics_steps),
        "diagnostics": {key: float(value) for key, value in diagnostics.items()},
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
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
        return path
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def build_arg_parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "logs/m1_panda_folded_load/probe-gpu0.json",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs != 8:
        raise ValueError("folded-load physical probe requires num_envs=8")
    if args.steps < 1:
        raise ValueError("probe steps must be positive")
    report_path = args.report.expanduser().resolve()
    report: dict[str, object] = {
        "task": TASK_ID,
        "num_envs": args.num_envs,
        "device": args.device,
        "requested_steps": args.steps,
        "external_wrench_enabled": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    app = None
    env = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(args).app
        import gymnasium as gym
        import torch
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from go2_pvcnn.tasks.m1_panda_folded_load_env_cfg import configure_folded_load_stage
        from go2_pvcnn.tasks.m1_panda_folded_load_wrapper import M1PandaFoldedLoadEnvWrapper

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=8)
        cfg.seed = 42
        configure_folded_load_stage(cfg, "L0-C0")
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        wrapper = M1PandaFoldedLoadEnvWrapper(env, stage="L0-C0", seed=42)
        observations, _ = wrapper.reset()
        finite_state = bool(torch.isfinite(observations).all())
        actions = torch.zeros((8, 23), device=wrapper.device)
        aggregate = wrapper.get_training_diagnostics()
        physics_steps = 0
        for _ in range(args.steps):
            observations, rewards, dones, _ = wrapper.step(actions)
            physics_steps += 1
            finite_state &= bool(
                torch.isfinite(observations).all()
                and torch.isfinite(rewards).all()
                and torch.isfinite(dones).all()
                and actions[:, 16:23].eq(0.0).all()
            )
            current = wrapper.get_training_diagnostics()
            for key, value in current.items():
                if key == "joint_limit_proximity_min":
                    aggregate[key] = min(aggregate[key], value)
                else:
                    aggregate[key] = max(aggregate[key], value)
        report.update(
            evaluate_probe(
                aggregate,
                finite_state=finite_state,
                physics_steps=physics_steps,
            )
        )
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(report_path, report)
        print(json.dumps({"report": str(report_path), **report}, indent=2))
        return 0 if report["passed"] else 2
    except BaseException as error:
        report.update(
            {
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(report_path, report)
        print(json.dumps({"report": str(report_path), **report}, indent=2))
        return 2
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
