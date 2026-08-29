#!/usr/bin/env python3
"""Train the fresh 8D Arm-MPC residual policy behind staged safety gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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


TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"
STAGE_LIMITS = {"zero": 10, "short": 100, "long": 3000}


@dataclass(frozen=True)
class PromotionLineage:
    manifest: Path
    manifest_sha256: str
    short_manifest: Path
    short_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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


def resolve_max_iterations(stage: str, requested: int | None) -> int:
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown stage {stage!r}")
    limit = STAGE_LIMITS[stage]
    value = limit if requested is None else requested
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
        raise ValueError(f"{stage} max_iterations must be in [1, {limit}]")
    if stage == "short" and value != limit:
        raise ValueError("short max_iterations must be exactly 100")
    return value


def _resolve_manifest_path(raw_path: object, *, parent: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_promotion_manifest(
    manifest_path: str | os.PathLike[str] | None,
    *,
    asset_path: Path,
    config_path: Path,
    reward_path: Path,
) -> PromotionLineage:
    if manifest_path is None:
        raise ValueError("long stage requires an accepted promotion manifest")
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("accepted") is not True:
        raise ValueError("promotion manifest must contain accepted=true")
    if document.get("asset_sha256") != sha256_file(asset_path):
        raise ValueError("promotion asset SHA does not match current asset")
    if document.get("config_sha256") != sha256_file(config_path):
        raise ValueError("promotion config SHA does not match current config")
    if document.get("reward_sha256") != sha256_file(reward_path):
        raise ValueError("promotion reward SHA does not match current reward")
    short_manifest = _resolve_manifest_path(
        document.get("short_manifest"), parent=manifest.parent, label="short_manifest"
    )
    short_manifest_sha = sha256_file(short_manifest)
    if document.get("short_manifest_sha256") != short_manifest_sha:
        raise ValueError("short manifest SHA does not match promotion manifest")
    short_document = json.loads(short_manifest.read_text(encoding="utf-8"))
    if (
        short_document.get("stage") != "short"
        or short_document.get("status") != "safe_complete"
        or short_document.get("promotion_required") is not True
        or short_document.get("accepted") is not False
    ):
        raise ValueError("promotion parent must be a safe-complete short run")
    for label, path in (
        ("asset", asset_path),
        ("config", config_path),
        ("reward", reward_path),
    ):
        if short_document.get(f"{label}_sha256") != sha256_file(path):
            raise ValueError(f"short-run {label} SHA does not match current {label}")
    checkpoint = _resolve_manifest_path(
        document.get("best_checkpoint"),
        parent=manifest.parent,
        label="best_checkpoint",
    )
    checkpoint_sha = sha256_file(checkpoint)
    if document.get("best_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("promotion checkpoint SHA does not match manifest")
    candidates = short_document.get("candidate_checkpoints")
    if not isinstance(candidates, list) or not any(
        isinstance(value, dict)
        and value.get("checkpoint_sha256") == checkpoint_sha
        for value in candidates
    ):
        raise ValueError("promoted checkpoint is not a recorded short candidate")
    return PromotionLineage(
        manifest=manifest,
        manifest_sha256=sha256_file(manifest),
        short_manifest=short_manifest,
        short_manifest_sha256=short_manifest_sha,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )


def build_arg_parser(*, include_app_launcher_args: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_LIMITS), default="short")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--promotion_manifest", type=Path)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _atomic_save(runner, path: Path) -> None:
    descriptor, raw = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        runner.save(str(temporary))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, target: Path) -> None:
    import shutil

    descriptor, raw = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def initialize_fresh_residual_policy(runner) -> None:
    """Start with an exact zero mean while retaining trainable exploration."""
    import torch

    actor = runner.alg.actor_critic
    output = actor.actor_head[-2]
    if not isinstance(output, torch.nn.Linear) or output.out_features != 8:
        raise RuntimeError("residual actor must end in an 8-output Linear before Tanh")
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    if not actor.noise_parameter.requires_grad:
        raise RuntimeError("residual action standard deviation must remain trainable")


class ResidualTrainingSafetyController:
    CANDIDATE_UPDATES = (0, 25, 50, 75, 100)

    def __init__(self, run_dir: Path):
        from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import (
            ResidualTrainingSafetyGuard,
        )

        self.run_dir = run_dir
        self.safety_guard = ResidualTrainingSafetyGuard()
        self.candidate_checkpoints: dict[int, Path] = {}

    def prime(self, runner) -> None:
        """Save the exact pre-rollout, pre-update zero policy."""
        checkpoint = self.run_dir / "candidate_u000.pt"
        _atomic_save(runner, checkpoint)
        self.candidate_checkpoints[0] = checkpoint

    @staticmethod
    def _metrics(summary):
        from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import (
            ResidualEvalMetrics,
        )

        values = dict(summary.environment_metrics)
        required = (
            "hard_failure_count",
            "mpc_feasible_rate",
            "qp_feasible_rate",
            "four_contact_rate",
            "roll_pitch_rms",
            "base_height_rms",
            "ee_position_error",
            "ee_orientation_error",
            "wrench_error",
            "slip",
            "intervention_ratio",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise RuntimeError(f"missing residual training diagnostics: {missing}")
        saturation = tuple(float(values[f"saturation_fraction_{i}"]) for i in range(8))
        return ResidualEvalMetrics(
            hard_failure_count=int(round(values["hard_failure_count"])),
            mpc_feasible_rate=float(values["mpc_feasible_rate"]),
            qp_feasible_rate=float(values["qp_feasible_rate"]),
            four_contact_rate=float(values["four_contact_rate"]),
            roll_pitch_rms=float(values["roll_pitch_rms"]),
            base_height_rms=float(values["base_height_rms"]),
            ee_position_error=float(values["ee_position_error"]),
            ee_orientation_error=float(values["ee_orientation_error"]),
            wrench_error=float(values["wrench_error"]),
            slip=float(values["slip"]),
            intervention_ratio=float(values["intervention_ratio"]),
            saturation_fraction=saturation,
        )

    def on_iteration(self, runner, summary):
        scalars = (
            summary.learning_rate,
            summary.kl_mean,
            summary.kl_max,
            summary.grad_norm,
            summary.active_action_std_min,
            summary.active_action_std_max,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            return "nonfinite_optimizer_diagnostic"
        metrics = self._metrics(summary)
        stop_reason = self.safety_guard.observe(metrics)
        completed_updates = int(summary.iteration) + 1
        if stop_reason is None and completed_updates in self.CANDIDATE_UPDATES:
            checkpoint = self.run_dir / f"candidate_u{completed_updates:03d}.pt"
            _atomic_save(runner, checkpoint)
            self.candidate_checkpoints[completed_updates] = checkpoint
        return stop_reason


def _default_run_dir(stage: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (ROOT / "logs/m1_panda_arm_mpc_residual" / f"{stage}_s{seed}_{stamp}").resolve()


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    iterations = resolve_max_iterations(args.stage, args.max_iterations)
    asset_path = (ROOT / "assets/m1_panda/m1_panda.usd").resolve()
    config_path = (ROOT / "agent/m1_panda_arm_mpc_residual_train_cfg.py").resolve()
    reward_path = (
        ROOT / "go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py"
    ).resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    lineage = (
        validate_promotion_manifest(
            args.promotion_manifest,
            asset_path=asset_path,
            config_path=config_path,
            reward_path=reward_path,
        )
        if args.stage == "long"
        else None
    )
    if args.stage != "long" and args.promotion_manifest is not None:
        raise ValueError("promotion_manifest is valid only for the long stage")
    run_dir = (
        _default_run_dir(args.stage, args.seed)
        if args.run_dir is None
        else args.run_dir.expanduser().resolve()
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from agent import get_m1_panda_arm_mpc_residual_train_cfg

    train_cfg = get_m1_panda_arm_mpc_residual_train_cfg()
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "starting",
        "accepted": False,
        "task": TASK_ID,
        "stage": args.stage,
        "device": args.device,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_iterations": iterations,
        "fresh_8d_policy": args.stage != "long",
        "legacy_23d_checkpoint_loaded": False,
        "force_zero_residual": args.stage == "zero",
        "asset_path": str(asset_path),
        "asset_sha256": sha256_file(asset_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "reward_path": str(reward_path),
        "reward_sha256": sha256_file(reward_path),
        "promotion_manifest": None if lineage is None else str(lineage.manifest),
        "promotion_manifest_sha256": None if lineage is None else lineage.manifest_sha256,
        "short_manifest": None if lineage is None else str(lineage.short_manifest),
        "short_manifest_sha256": None if lineage is None else lineage.short_manifest_sha256,
        "parent_checkpoint": None if lineage is None else str(lineage.checkpoint),
        "parent_checkpoint_sha256": None if lineage is None else lineage.checkpoint_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
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
        from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
            M1PandaArmMpcResidualEnvWrapper,
        )

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        wrapper = M1PandaArmMpcResidualEnvWrapper(
            env,
            seed=args.seed,
            trajectory_scale=0.0 if args.stage == "zero" else 1.0,
            force_zero_residual=args.stage == "zero",
        )
        wrapper.reset()
        observations, _ = wrapper.get_observations()
        if tuple(observations.shape) != (args.num_envs, 103):
            raise RuntimeError("residual observations must have shape [num_envs, 103]")
        if wrapper.num_actions != 8:
            raise RuntimeError("residual policy action dimension must be 8")
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(run_dir), device=args.device)
        if lineage is None:
            initialize_fresh_residual_policy(runner)
        else:
            # Only an accepted, hash-matched 8D short checkpoint may seed long mode;
            # legacy 23D checkpoints never enter this path.
            runner.load(str(lineage.checkpoint), load_optimizer=False, keep_std=True)
            runner.current_learning_iteration = 0
        controller = ResidualTrainingSafetyController(run_dir)
        controller.prime(runner)
        manifest.update(
            {
                "status": "running",
                "observation_dim": 103,
                "action_dim": 8,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        result = runner.learn(
            num_learning_iterations=iterations,
            init_at_random_ep_len=False,
            iteration_callback=lambda summary: controller.on_iteration(runner, summary),
        )
        safe_complete = result.stop_reason is None
        candidate_records = [
            {
                "completed_updates": updates,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            for updates, checkpoint in sorted(controller.candidate_checkpoints.items())
        ]
        if args.stage == "short" and safe_complete and set(
            controller.candidate_checkpoints
        ) != set(controller.CANDIDATE_UPDATES):
            raise RuntimeError("safe short run did not produce all five candidates")
        manifest.update(
            {
                "status": "safe_complete" if safe_complete else "safety_stopped",
                "accepted": False,
                "promotion_required": True,
                "completed_iterations": result.completed_iterations,
                "stop_reason": result.stop_reason or "requested_iterations_complete",
                "candidate_checkpoints": candidate_records,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        return 0
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
