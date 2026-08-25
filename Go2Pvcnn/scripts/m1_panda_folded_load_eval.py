#!/usr/bin/env python3
"""Evaluate one folded-load candidate with a fixed balanced command table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import (
    balanced_eval_commands,
    stage_spec,
)
from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import (
    AtomicStageArtifacts,
    EpisodeRecord,
    sha256_file,
)


TASK_ID = "Isaac-M1-Panda-Folded-Load-v0"
EVALUATION_SEEDS = (42, 43, 44)
DIRECTIONAL_TRACKING = {
    "forward": ("vx_error_sq_sum", "vx_rmse", 0.04),
    "reverse": ("vx_error_sq_sum", "vx_rmse", 0.04),
    "left": ("wz_error_sq_sum", "wz_rmse", 0.12),
    "right": ("wz_error_sq_sum", "wz_rmse", 0.12),
}


def _bucket(records: list[EpisodeRecord], name: str) -> list[EpisodeRecord]:
    if name == "forward":
        return [record for record in records if record.command[0] > 0.0]
    if name == "reverse":
        return [record for record in records if record.command[0] < 0.0]
    if name == "left":
        return [record for record in records if record.command[2] > 0.0]
    if name == "right":
        return [record for record in records if record.command[2] < 0.0]
    if name == "stationary":
        return [record for record in records if record.command[0] == 0.0 and record.command[2] == 0.0]
    raise KeyError(name)


def _finite_non_negative(value: float, label: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0.0 or (positive and value == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _total_steps(records: list[EpisodeRecord]) -> float:
    return sum(
        _finite_non_negative(record.steps, "episode steps", positive=True)
        for record in records
    )


def _rmse(records: list[EpisodeRecord], field: str) -> float:
    steps = _total_steps(records)
    squared_error = sum(
        _finite_non_negative(getattr(record, field), field) for record in records
    )
    return math.sqrt(squared_error / steps)


def _event_rate(records: list[EpisodeRecord], field: str) -> float:
    rate = sum(bool(getattr(record, field)) for record in records) / len(records)
    if not math.isfinite(rate):
        raise ValueError(f"{field} rate must be finite")
    return rate


def _directional_report(records: list[EpisodeRecord], name: str) -> dict[str, object]:
    field, tracking_metric, tracking_limit = DIRECTIONAL_TRACKING[name]
    values = _bucket(records, name)
    if not values:
        return {
            "episode_count": 0,
            "tracking_metric": tracking_metric,
            "tracking_rmse": None,
            "tracking_limit": tracking_limit,
            "base_contact_rate": None,
            "bad_orientation_rate": None,
            "passed": False,
        }
    tracking_rmse = _rmse(values, field)
    contact_rate = _event_rate(values, "base_contact")
    orientation_rate = _event_rate(values, "bad_orientation")
    return {
        "episode_count": len(values),
        "tracking_metric": tracking_metric,
        "tracking_rmse": tracking_rmse,
        "tracking_limit": tracking_limit,
        "base_contact_rate": contact_rate,
        "bad_orientation_rate": orientation_rate,
        "passed": bool(
            len(values) >= 8
            and contact_rate <= 0.02
            and orientation_rate <= 0.02
            and tracking_rmse <= tracking_limit
        ),
    }


def evaluate_records(stage: str, seed: int, episodes) -> dict[str, object]:
    """Apply training-equivalent physical gates to one fixed 64-env episode."""
    stage_spec(stage)
    if seed not in EVALUATION_SEEDS:
        raise ValueError("evaluation seed must be 42, 43, or 44")
    records = list(episodes)
    env_ids = [record.env_id for record in records]
    if len(records) != 64 or sorted(env_ids) != list(range(64)):
        raise ValueError("evaluation requires exactly one episode from each of 64 environments")
    count = len(records)
    timeout = _event_rate(records, "time_out")
    contact = _event_rate(records, "base_contact")
    orientation = _event_rate(records, "bad_orientation")
    vx_rmse = _rmse(records, "vx_error_sq_sum")
    wz_rmse = _rmse(records, "wz_error_sq_sum")
    buckets = {name: _bucket(records, name) for name in ("forward", "reverse", "left", "right")}
    directional_metrics = {
        name: _directional_report(records, name) for name in DIRECTIONAL_TRACKING
    }
    directional_pass = all(item["passed"] for item in directional_metrics.values())
    stationary = _bucket(records, "stationary")
    stationary_steps = _total_steps(stationary)
    stationary_vx = sum(
        _finite_non_negative(record.stationary_abs_vx_sum, "stationary_abs_vx_sum")
        for record in stationary
    ) / stationary_steps
    stationary_wz = sum(
        _finite_non_negative(record.stationary_abs_wz_sum, "stationary_abs_wz_sum")
        for record in stationary
    ) / stationary_steps
    passed = bool(
        timeout >= 0.95
        and contact <= 0.02
        and orientation <= 0.02
        and vx_rmse <= 0.04
        and wz_rmse <= 0.12
        and directional_pass
        and stationary_vx <= 0.03
        and stationary_wz <= 0.08
    )
    return {
        "stage": stage,
        "seed": seed,
        "passed": passed,
        "completed_episodes": count,
        "time_out_rate": timeout,
        "base_contact_rate": contact,
        "bad_orientation_rate": orientation,
        "vx_rmse": vx_rmse,
        "wz_rmse": wz_rmse,
        "stationary_abs_vx": stationary_vx,
        "stationary_abs_wz": stationary_wz,
        "bucket_counts": tuple(sorted((name, len(values)) for name, values in buckets.items())),
        "directional_metrics": directional_metrics,
        "directional_pass": directional_pass,
    }


def build_arg_parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=EVALUATION_SEEDS)
    parser.add_argument("--num_envs", type=int, default=64)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _finalize_report_set(
    artifacts: AtomicStageArtifacts,
    checkpoint: Path,
    reports: list[Path],
    *,
    diagnostic_only: bool,
) -> dict[str, object]:
    if diagnostic_only:
        return artifacts.finalize_diagnostics(checkpoint, reports)
    return artifacts.finalize_evaluations(checkpoint, reports)


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs != 64:
        raise ValueError("fixed evaluation requires num_envs=64")
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != args.stage or manifest.get("training_eligible") is not True:
        raise ValueError("evaluation requires the matching eligible training manifest")
    checkpoint = (args.checkpoint or run_dir / "model_best.pt").expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if manifest.get("best_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("evaluation checkpoint SHA does not match training manifest")

    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import go2_pvcnn.tasks  # noqa: F401
        from agent import get_m1_panda_folded_load_train_cfg
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from go2_pvcnn.tasks.m1_panda_folded_load_env_cfg import configure_folded_load_stage
        from go2_pvcnn.tasks.m1_panda_folded_load_wrapper import M1PandaFoldedLoadEnvWrapper

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=64)
        cfg.seed = args.seed
        configure_folded_load_stage(cfg, args.stage)
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        wrapper = M1PandaFoldedLoadEnvWrapper(env, stage=args.stage, seed=args.seed)
        wrapper.reset()
        commands = balanced_eval_commands(64, stage_spec(args.stage), device=wrapper.device)
        wrapper.set_evaluation_commands(commands)
        observations, _ = wrapper.get_observations()
        runner = OnPolicyRunner(wrapper, get_m1_panda_folded_load_train_cfg(), log_dir=None, device=args.device)
        runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=args.device)
        records: dict[int, EpisodeRecord] = {}
        for _ in range(int(wrapper.max_episode_length) + 1):
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, _, _ = wrapper.step(actions)
            for record in wrapper.drain_completed_episode_records():
                records.setdefault(record.env_id, record)
            if len(records) == 64:
                break
        report = evaluate_records(args.stage, args.seed, records.values())
        report.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        artifacts = AtomicStageArtifacts(run_dir)
        artifacts.write_evaluation(args.seed, report)
        reports = [run_dir / f"evaluation_seed_{seed}.json" for seed in EVALUATION_SEEDS]
        if all(path.is_file() for path in reports):
            diagnostic_only = manifest.get("diagnostic_only") is True
            decision = _finalize_report_set(
                artifacts,
                checkpoint,
                reports,
                diagnostic_only=diagnostic_only,
            )
            manifest.update(decision)
            if diagnostic_only:
                manifest["status"] = "diagnostic_complete"
            else:
                manifest["status"] = "accepted" if decision["accepted"] else "rejected"
            from m1_panda_folded_load_train import atomic_write_json
            atomic_write_json(manifest_path, manifest)
        return 0 if report["passed"] else 2
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
