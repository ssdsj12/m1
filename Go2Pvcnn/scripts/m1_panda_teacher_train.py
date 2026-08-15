#!/usr/bin/env python3
"""Train M1 + Panda force-aware Teacher balance stages with RSL-RL PPO."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for import_path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
TASK_IDS = {
    "A0": "Isaac-M1-Panda-Teacher-A0-v0",
    "A1": "Isaac-M1-Panda-Teacher-A1-v0",
}
RECOVERY_LINEAGE_FIELDS = (
    "recovery_source_checkpoint",
    "recovery_source_checkpoint_sha256",
    "recovery_source_iteration",
    "optimizer_reset",
    "recovery_learning_rate",
    "noise_std_mode",
    "minimum_effective_std",
    "initial_curriculum_step",
    "initial_curriculum_scale",
)


def validate_cli_contract(args) -> None:
    """Reject contradictory or non-positive training arguments before startup."""
    fork_checkpoint = getattr(args, "fork_checkpoint", None)
    reset_optimizer = getattr(args, "reset_optimizer", False)
    if fork_checkpoint is not None:
        if args.stage != "A1":
            raise ValueError("--fork-checkpoint is A1-only")
        if args.run_name is None:
            raise ValueError("--fork-checkpoint requires --run_name")
        if args.resume_checkpoint is not None:
            raise ValueError("--fork-checkpoint cannot be combined with --resume-checkpoint")
        if reset_optimizer:
            raise ValueError(
                "--reset-optimizer is implicit and forbidden with --fork-checkpoint"
            )
    if args.stage == "A0" and args.base_checkpoint is not None:
        raise ValueError("A0 does not accept --base-checkpoint")
    if args.stage == "A1" and args.base_checkpoint is None:
        raise ValueError("A1 requires --base-checkpoint")
    positive_fields = (
        "max_iterations",
        "num_envs",
        "save_interval",
        "num_steps_per_env",
        "learning_epochs",
        "num_mini_batches",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive")
    if args.resume_checkpoint is not None and args.run_name is not None:
        raise ValueError("--run_name is forbidden with --resume-checkpoint")


def recovery_initial_curriculum_step(
    source_iteration: int,
    num_steps_per_env: int,
    curriculum_steps: int,
) -> int:
    """Restore scheduler progress from completed RSL-RL rollout iterations."""
    values = {
        "source_iteration": source_iteration,
        "num_steps_per_env": num_steps_per_env,
        "curriculum_steps": curriculum_steps,
    }
    for name, value in values.items():
        minimum = 0 if name == "source_iteration" else 1
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            qualifier = "nonnegative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    return min(source_iteration * num_steps_per_env, curriculum_steps)


def preserve_recovery_resume_state(
    current_manifest: dict[str, object],
    previous_manifest: dict[str, object],
) -> None:
    """Carry immutable recovery lineage and evaluation history across resume."""
    if "recovery_source_checkpoint" not in previous_manifest:
        return
    missing = [
        field for field in RECOVERY_LINEAGE_FIELDS if field not in previous_manifest
    ]
    if missing:
        raise ValueError(f"recovery resume manifest is missing fields: {missing}")
    for field in RECOVERY_LINEAGE_FIELDS:
        current_manifest[field] = previous_manifest[field]
    artifacts = previous_manifest.get("evaluation_artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("recovery evaluation_artifacts must be a list")
    current_manifest.update(
        {
            "evaluation_artifacts": list(artifacts),
            "best_checkpoint": previous_manifest.get("best_checkpoint"),
            "best_metrics": previous_manifest.get("best_metrics"),
            "stop_reason": None,
            "consecutive_survival_regressions": previous_manifest.get(
                "consecutive_survival_regressions", 0
            ),
        }
    )


def apply_recovery_resume_train_cfg(
    train_cfg: dict[str, object], previous_manifest: dict[str, object]
) -> None:
    """Restore recovery hyperparameters before constructing a resumed runner."""
    if "recovery_source_checkpoint" not in previous_manifest and (
        "recovery_learning_rate" not in previous_manifest
    ):
        return
    learning_rate = previous_manifest.get("recovery_learning_rate")
    if not isinstance(learning_rate, (int, float)) or isinstance(
        learning_rate, bool
    ) or learning_rate <= 0.0:
        raise ValueError("recovery_learning_rate must be positive")
    algorithm = train_cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("train_cfg algorithm must be a dictionary")
    algorithm["learning_rate"] = float(learning_rate)


def mark_recovery_block_completed(manifest: dict[str, object]) -> None:
    """Mark either a forked or resumed recovery block ready for evaluation."""
    if "recovery_source_checkpoint" in manifest:
        manifest["stop_reason"] = "block_completed_pending_evaluation"


def build_log_dir(
    log_root: Path, stage: str, run_name: str | None
) -> Path:
    """Create a fresh stage-scoped run directory."""
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(log_root).expanduser().resolve() / stage.lower() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_log_dir(args) -> Path:
    """Create a new run directory or reuse the resume checkpoint directory."""
    if args.resume_checkpoint is not None:
        checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        return checkpoint.parent
    return build_log_dir(args.log_root, args.stage, args.run_name)


def advance_runner_after_resume(runner) -> None:
    """Continue at the iteration after the checkpoint saved by RSL-RL."""
    current = runner.current_learning_iteration
    if not isinstance(current, int) or isinstance(current, bool) or current < 0:
        raise ValueError(f"invalid loaded checkpoint iteration: {current!r}")
    runner.current_learning_iteration = current + 1


def runtime_contract_snapshot(wrapper) -> dict[str, int | float]:
    """Capture the exact live 60/16 contract and a nonzero scheduled wrench."""
    import torch

    observations, _ = wrapper.get_observations()
    if (
        not isinstance(observations, torch.Tensor)
        or observations.ndim != 2
        or observations.shape[1] != 60
        or not bool(torch.isfinite(observations).all())
    ):
        raise RuntimeError("runtime policy observation must be finite with width 60")
    if wrapper.num_actions != 16:
        raise RuntimeError(f"runtime action dimension must be 16, got {wrapper.num_actions}")
    max_abs_wrench = float(wrapper.max_abs_wrench_seen)
    if not torch.isfinite(torch.tensor(max_abs_wrench)):
        raise RuntimeError("runtime scheduled wrench maximum must be finite")
    if max_abs_wrench <= 0.0:
        raise RuntimeError("runtime scheduled wrench must be nonzero after rollout")
    return {
        "policy_observation_dim": int(observations.shape[1]),
        "action_dim": int(wrapper.num_actions),
        "max_abs_wrench_b_seen": max_abs_wrench,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Train M1 + Panda force-aware Teacher balance stages."
    )
    parser.add_argument("--stage", choices=tuple(TASK_IDS), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--log-root", type=Path, default=GO2PVCNN_ROOT / "logs/m1_panda_teacher"
    )
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--fork-checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true", default=False)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()
    simulation_app = None
    env = None
    wrapper = None
    frozen_actor = None
    log_dir = None
    manifest = None
    try:
        validate_cli_contract(args)

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args).app

        import gymnasium as gym

        from agent import get_m1_panda_teacher_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_panda_teacher import stage_disturbance_cfg
        from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
            MANIFEST_FILENAME,
            TEACHER_ACTION_DIM,
            TEACHER_HIDDEN_DIMS,
            TEACHER_OBSERVATION_DIM,
            atomic_write_manifest,
            build_run_manifest,
            checkpoint_iteration,
            file_sha256,
            load_frozen_teacher_actor,
            module_sha256,
            validate_teacher_checkpoint,
        )
        from go2_pvcnn.tasks.m1_panda_teacher_wrapper import (
            M1PandaTeacherEnvWrapper,
        )
        from go2_pvcnn.tasks.m1_residual_action import (
            M1ResidualActionComposerCfg,
        )
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        task_id = TASK_IDS[args.stage]
        train_cfg = get_m1_panda_teacher_train_cfg()
        train_cfg["save_interval"] = args.save_interval
        train_cfg["num_steps_per_env"] = args.num_steps_per_env
        train_cfg["algorithm"]["num_learning_epochs"] = args.learning_epochs
        train_cfg["algorithm"]["num_mini_batches"] = args.num_mini_batches
        disturbance_cfg = stage_disturbance_cfg(args.stage)

        expected_base_hash = (
            file_sha256(args.base_checkpoint) if args.stage == "A1" else None
        )
        selected_checkpoint = args.fork_checkpoint or args.resume_checkpoint
        selected_manifest = None
        source_iteration = None
        initial_curriculum_step = 0
        if selected_checkpoint is not None:
            _, selected_manifest = validate_teacher_checkpoint(
                selected_checkpoint,
                expected_stage=args.stage,
                expected_observation_dim=TEACHER_OBSERVATION_DIM,
                expected_action_dim=TEACHER_ACTION_DIM,
                expected_actor_hidden_dims=TEACHER_HIDDEN_DIMS,
                expected_base_sha256=expected_base_hash,
                require_optimizer=(
                    args.resume_checkpoint is not None and not args.reset_optimizer
                ),
            )
            source_iteration = checkpoint_iteration(selected_checkpoint)
            initial_curriculum_step = recovery_initial_curriculum_step(
                source_iteration,
                args.num_steps_per_env,
                disturbance_cfg.curriculum_steps,
            )
            if args.resume_checkpoint is not None:
                apply_recovery_resume_train_cfg(train_cfg, selected_manifest)
        if args.fork_checkpoint is not None:
            train_cfg["algorithm"]["learning_rate"] = 1.0e-4

        if args.stage == "A1":
            frozen_actor = load_frozen_teacher_actor(
                args.base_checkpoint,
                device=args.device,
                policy_cfg=deepcopy(train_cfg["policy"]),
            )

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed

        env = gym.make(task_id, cfg=env_cfg)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise TypeError(f"{task_id} did not create ManagerBasedRLEnv")
        wrapper = M1PandaTeacherEnvWrapper(
            env.unwrapped,
            stage=args.stage,
            base_actor=frozen_actor,
            seed=args.seed,
            initial_curriculum_step=initial_curriculum_step,
        )

        log_dir = resolve_log_dir(args)
        dump_yaml(str(log_dir / "env_cfg.yaml"), env_cfg.to_dict())
        dump_yaml(str(log_dir / "train_cfg.yaml"), train_cfg)

        manifest = build_run_manifest(
            stage=args.stage,
            task_id=task_id,
            seed=args.seed,
            composer_cfg=M1ResidualActionComposerCfg(),
            disturbance_cfg=disturbance_cfg,
            base_checkpoint=args.base_checkpoint,
            frozen_actor=frozen_actor,
            resume_checkpoint=args.resume_checkpoint,
            recovery_source_checkpoint=args.fork_checkpoint,
            recovery_source_iteration=(
                source_iteration if args.fork_checkpoint is not None else None
            ),
            initial_curriculum_step=(
                initial_curriculum_step if args.fork_checkpoint is not None else None
            ),
            optimizer_reset=True if args.fork_checkpoint is not None else None,
            recovery_learning_rate=(
                1.0e-4 if args.fork_checkpoint is not None else None
            ),
            noise_std_mode=("scalar" if args.fork_checkpoint is not None else None),
            minimum_effective_std=(
                0.001 if args.fork_checkpoint is not None else None
            ),
        )
        if args.resume_checkpoint is not None and selected_manifest is not None:
            preserve_recovery_resume_state(manifest, selected_manifest)
        if args.fork_checkpoint is not None:
            manifest.update(
                {
                    "evaluation_artifacts": [],
                    "best_checkpoint": None,
                    "best_metrics": None,
                    "stop_reason": None,
                }
            )
        atomic_write_manifest(log_dir / MANIFEST_FILENAME, manifest)

        runner = OnPolicyRunner(
            wrapper,
            deepcopy(train_cfg),
            log_dir=str(log_dir),
            device=env_cfg.sim.device,
        )
        if selected_checkpoint is not None:
            load_runner_checkpoint(
                runner,
                selected_checkpoint,
                load_optimizer=(
                    args.resume_checkpoint is not None and not args.reset_optimizer
                ),
                minimum_effective_std=(
                    0.001 if args.fork_checkpoint is not None else None
                ),
            )
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
        runtime_snapshot = runtime_contract_snapshot(wrapper)
        print(f"[Teacher Runtime] {runtime_snapshot}", flush=True)
        wrapper.assert_frozen_actor_unchanged()

        final_iteration = int(runner.current_learning_iteration)
        final_checkpoint = log_dir / f"model_{final_iteration}.pt"
        if not final_checkpoint.is_file():
            raise FileNotFoundError(
                f"runner did not write final checkpoint: {final_checkpoint}"
            )
        manifest["status"] = "completed"
        manifest["final_iteration"] = final_iteration
        manifest["final_checkpoint"] = final_checkpoint.name
        manifest["frozen_actor_final_sha256"] = (
            module_sha256(frozen_actor) if frozen_actor is not None else None
        )
        manifest["runtime_contract"] = runtime_snapshot
        mark_recovery_block_completed(manifest)
        atomic_write_manifest(log_dir / MANIFEST_FILENAME, manifest)
        return 0
    except BaseException as error:
        traceback.print_exc()
        if log_dir is not None:
            try:
                if manifest is None:
                    manifest = {
                        "schema_version": 1,
                        "stage": args.stage,
                        "task_id": TASK_IDS.get(args.stage),
                    }
                manifest["status"] = "failed"
                manifest["error"] = f"{type(error).__name__}: {error}"
                if frozen_actor is not None:
                    from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
                        module_sha256,
                    )

                    manifest["frozen_actor_final_sha256"] = module_sha256(
                        frozen_actor
                    )
                from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
                    MANIFEST_FILENAME,
                    atomic_write_manifest,
                )

                atomic_write_manifest(log_dir / MANIFEST_FILENAME, manifest)
            except BaseException:
                traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def load_runner_checkpoint(
    runner,
    checkpoint_path: str | os.PathLike[str],
    *,
    load_optimizer: bool,
    minimum_effective_std: float | None = None,
) -> None:
    """Load one validated checkpoint and apply recovery-only policy resets."""
    resolved = str(Path(checkpoint_path).expanduser().resolve())
    runner.load(
        resolved,
        load_optimizer=load_optimizer,
        keep_std=True,
    )
    advance_runner_after_resume(runner)
    if minimum_effective_std is not None:
        runner.alg.actor_critic.clip_std(min=minimum_effective_std)


if __name__ == "__main__":
    raise SystemExit(main())
