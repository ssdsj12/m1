#!/usr/bin/env python3
"""Train exactly one guarded folded-load locomotion curriculum stage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import NamedTuple


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import stage_spec
from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import (
    FoldedLoadTrainingGuard,
    sha256_file,
)


TASK_ID = "Isaac-M1-Panda-Folded-Load-v0"


class ParentLineage(NamedTuple):
    stage: str
    manifest: Path
    manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str


def atomic_write_json(path: Path, payload: dict[str, object]) -> Path:
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


def prepare_empty_run_dir(path: str | os.PathLike[str]) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def training_completion_exit_code(*, eligible: bool) -> int:
    """A finite completed run is successful even when it is only a smoke run."""
    if not isinstance(eligible, bool):
        raise TypeError("eligible must be boolean")
    return 0


def validate_parent(
    stage: str, parent_manifest: str | os.PathLike[str] | None
) -> ParentLineage | None:
    contract = stage_spec(stage)
    if contract.parent is None:
        if parent_manifest is not None:
            raise ValueError("L0-C0 must start fresh without a parent or resume checkpoint")
        return None
    if parent_manifest is None:
        raise ValueError(f"{stage} requires an accepted immediate-parent manifest")
    manifest_path = Path(parent_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("accepted") is not True:
        raise ValueError("parent manifest must contain accepted=true")
    if document.get("stage") != contract.parent:
        raise ValueError(
            f"{stage} immediate parent must be {contract.parent}, "
            f"got {document.get('stage')!r}"
        )
    raw_checkpoint = document.get("final_checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("accepted parent must name final_checkpoint")
    checkpoint = Path(raw_checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = document.get("final_checkpoint_sha256")
    actual_sha = sha256_file(checkpoint)
    if expected_sha != actual_sha:
        raise ValueError("parent final checkpoint SHA does not match manifest")
    return ParentLineage(
        stage=contract.parent,
        manifest=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        checkpoint=checkpoint,
        checkpoint_sha256=actual_sha,
    )


def build_arg_parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--parent_manifest", type=Path)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=600)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _atomic_save(runner, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        runner.save(str(temporary))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class StageTrainingController:
    def __init__(self, run_dir: Path, wrapper, stage: str):
        self.run_dir = run_dir
        self.wrapper = wrapper
        self.guard = FoldedLoadTrainingGuard(stage_spec(stage))
        self.best_checkpoint = run_dir / "model_best.pt"

    def on_iteration(self, runner, summary):
        diagnostics = dict(summary.environment_metrics)
        scalar_values = (
            summary.learning_rate,
            summary.kl_mean,
            summary.kl_max,
            summary.grad_norm,
            summary.active_action_std_min,
            summary.active_action_std_max,
            *diagnostics.values(),
        )
        finite = all(math.isfinite(float(value)) for value in scalar_values)
        fold_failure = bool(
            diagnostics.get("fold_error_max", 0.0) > 0.35
            or diagnostics.get("effort_utilization_max", 0.0) > 1.0
            or diagnostics.get("joint_limit_proximity_min", math.inf) <= 0.01
        )
        decision = self.guard.update(
            summary.iteration,
            self.wrapper.drain_completed_episode_records(),
            finite=finite,
            inactive_action_max=diagnostics.get("inactive_action_max", 0.0),
            fold_hard_failure=fold_failure,
        )
        if decision.save_best:
            _atomic_save(runner, self.best_checkpoint)
            atomic_write_json(
                self.run_dir / "model_best.json",
                {
                    "checkpoint": str(self.best_checkpoint),
                    "checkpoint_sha256": sha256_file(self.best_checkpoint),
                    "snapshot": asdict(decision.snapshot),
                },
            )
        return decision.reason if decision.stop else None


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0 < args.max_iterations <= 600:
        raise ValueError("max_iterations must be in [1, 600]")
    contract = stage_spec(args.stage)
    parent = validate_parent(args.stage, args.parent_manifest)
    run_dir = prepare_empty_run_dir(args.run_dir)
    asset = (ROOT / "assets/m1_panda/m1_panda.usd").resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)
    from agent import get_m1_panda_folded_load_train_cfg

    train_cfg = get_m1_panda_folded_load_train_cfg()
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "starting",
        "accepted": False,
        "task": TASK_ID,
        "stage": contract.name,
        "parent_stage": None if parent is None else parent.stage,
        "parent_manifest": None if parent is None else str(parent.manifest),
        "parent_manifest_sha256": None if parent is None else parent.manifest_sha256,
        "parent_checkpoint": None if parent is None else str(parent.checkpoint),
        "parent_checkpoint_sha256": None if parent is None else parent.checkpoint_sha256,
        "asset_path": str(asset),
        "asset_sha256": sha256_file(asset),
        "active_action_mask": [1] * 16 + [0] * 7,
        "external_wrench_enabled": False,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_iterations": args.max_iterations,
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_cfg": train_cfg,
    }
    atomic_write_json(manifest_path, manifest)

    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from go2_pvcnn.tasks.m1_panda_folded_load_env_cfg import (
            configure_folded_load_stage,
        )
        from go2_pvcnn.tasks.m1_panda_folded_load_wrapper import (
            M1PandaFoldedLoadEnvWrapper,
        )

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        configure_folded_load_stage(cfg, args.stage)
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        wrapper = M1PandaFoldedLoadEnvWrapper(env, stage=args.stage, seed=args.seed)
        wrapper.reset()
        observations, _ = wrapper.get_observations()
        if tuple(observations.shape) != (args.num_envs, 103):
            raise RuntimeError("folded-load policy observations must be [num_envs, 103]")
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(run_dir), device=args.device)
        if parent is not None:
            runner.load(str(parent.checkpoint), load_optimizer=False, keep_std=True)
            runner.alg.actor_critic.clip_std(max=0.01)
            runner.current_learning_iteration = 0
        controller = StageTrainingController(run_dir, wrapper, args.stage)
        manifest.update({"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(manifest_path, manifest)
        result = runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=False,
            iteration_callback=lambda summary: controller.on_iteration(runner, summary),
        )
        eligible = controller.guard.eligible_best is not None and controller.best_checkpoint.is_file()
        manifest.update(
            {
                "status": "eligible_pending_evaluation" if eligible else "rejected",
                "training_eligible": eligible,
                "completed_iterations": result.completed_iterations,
                "stop_reason": result.stop_reason or "requested_iterations_complete",
                "best_checkpoint": str(controller.best_checkpoint) if eligible else None,
                "best_checkpoint_sha256": sha256_file(controller.best_checkpoint) if eligible else None,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        return training_completion_exit_code(eligible=eligible)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
