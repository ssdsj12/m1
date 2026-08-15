#!/usr/bin/env python3
"""Play strict M1 + Panda force-aware Teacher A0/A1 checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import traceback

import torch


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
TERMINATION_NAMES = ("bad_orientation", "base_contact", "time_out")


def validate_cli_contract(args) -> None:
    """Reject contradictory or out-of-range playback arguments."""
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args.stage == "A0" and args.base_checkpoint is not None:
        raise ValueError("A0 does not accept --base-checkpoint")
    if args.stage == "A1" and args.base_checkpoint is None:
        raise ValueError("A1 requires --base-checkpoint")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.stats_interval <= 0:
        raise ValueError("--stats_interval must be positive")
    full_scale = getattr(args, "full_scale_disturbance", False)
    summary_json = getattr(args, "summary_json", None)
    disable_disturbance = getattr(args, "disable_disturbance", False)
    if full_scale and disable_disturbance:
        raise ValueError("full-scale evaluation requires disturbance")
    if full_scale and args.stage != "A1":
        raise ValueError("full-scale evaluation is defined only for A1")
    if full_scale and args.steps <= 0:
        raise ValueError("full-scale evaluation requires positive steps")
    if full_scale and summary_json is None:
        raise ValueError("full-scale evaluation requires --summary-json")


def _termination_terms(env) -> dict[str, torch.Tensor]:
    """Read available per-environment termination terms without inventing zeros."""
    base_env = env.unwrapped
    manager = getattr(base_env, "termination_manager", None)
    if manager is None:
        return {}
    terms: dict[str, torch.Tensor] = {}
    for name in TERMINATION_NAMES:
        try:
            value = manager.get_term(name)
        except Exception:
            value = None
        if value is not None:
            terms[name] = torch.as_tensor(
                value, dtype=torch.bool, device=base_env.device
            ).reshape(-1)
    if "time_out" not in terms and getattr(manager, "time_outs", None) is not None:
        terms["time_out"] = torch.as_tensor(
            manager.time_outs, dtype=torch.bool, device=base_env.device
        ).reshape(-1)
    return terms


def update_reset_counts(
    env, counts: dict[str, int | None]
) -> None:
    """Accumulate only termination terms that the environment actually exposes."""
    terms = _termination_terms(env)
    for name in TERMINATION_NAMES:
        value = terms.get(name)
        if value is None:
            continue
        previous = counts.get(name)
        counts[name] = (0 if previous is None else int(previous)) + int(
            value.sum().item()
        )


def format_play_stats(
    *,
    step: int,
    mean_reward: float,
    done_count: int,
    wrench_b: torch.Tensor,
    max_abs_wrench_seen: float,
    reset_counts: dict[str, int | None],
) -> str:
    """Format one finite playback diagnostic line."""
    if (
        not isinstance(wrench_b, torch.Tensor)
        or wrench_b.ndim != 2
        or wrench_b.shape[1] != 6
        or not bool(torch.isfinite(wrench_b).all())
    ):
        raise ValueError("wrench_b must be a finite tensor with shape [N, 6]")
    if not math.isfinite(mean_reward):
        raise ValueError("mean_reward must be finite")
    if not math.isfinite(max_abs_wrench_seen):
        raise ValueError("max_abs_wrench_seen must be finite")
    axis_max = wrench_b.abs().amax(dim=0).detach().cpu().tolist()
    axis_text = ",".join(f"{value:.3f}" for value in axis_max)
    reset_text = " ".join(
        f"{name}={reset_counts.get(name) if reset_counts.get(name) is not None else 'unavailable'}"
        for name in TERMINATION_NAMES
    )
    return (
        f"[Teacher Play] step={step} mean_reward={mean_reward:.6f} "
        f"done={done_count} wrench_axis_abs_max=[{axis_text}] "
        f"max_abs_wrench_seen={max_abs_wrench_seen:.3f} {reset_text}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Play strict M1 + Panda force-aware Teacher A0/A1 checkpoints."
    )
    parser.add_argument("--stage", choices=tuple(TASK_IDS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        "--base_checkpoint",
        dest="base_checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--num-envs", "--num_envs", dest="num_envs", type=int, default=1
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--stats-interval",
        "--stats_interval",
        dest="stats_interval",
        type=int,
        default=100,
    )
    parser.add_argument("--disable-disturbance", action="store_true")
    parser.add_argument("--full-scale-disturbance", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()
    simulation_app = None
    env = None
    wrapper = None
    try:
        validate_cli_contract(args)

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args).app

        import gymnasium as gym

        from agent import get_m1_panda_teacher_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import (
            TEACHER_ACTION_DIM,
            TEACHER_HIDDEN_DIMS,
            TEACHER_OBSERVATION_DIM,
            file_sha256,
            load_frozen_teacher_actor,
            validate_teacher_checkpoint,
            atomic_write_manifest,
        )
        from go2_pvcnn.tasks.m1_panda_teacher import stage_disturbance_cfg
        from go2_pvcnn.tasks.m1_panda_teacher_evaluation import (
            TeacherEvaluationAccumulator,
            validate_full_scale_summary,
        )
        from go2_pvcnn.tasks.m1_panda_teacher_wrapper import (
            M1PandaTeacherEnvWrapper,
        )
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        train_cfg = get_m1_panda_teacher_train_cfg()
        expected_base_hash = None
        frozen_actor = None
        if args.stage == "A1":
            expected_base_hash = file_sha256(args.base_checkpoint)
            frozen_actor = load_frozen_teacher_actor(
                args.base_checkpoint,
                device=args.device,
                policy_cfg=deepcopy(train_cfg["policy"]),
            )
        validate_teacher_checkpoint(
            args.checkpoint,
            expected_stage=args.stage,
            expected_observation_dim=TEACHER_OBSERVATION_DIM,
            expected_action_dim=TEACHER_ACTION_DIM,
            expected_actor_hidden_dims=TEACHER_HIDDEN_DIMS,
            expected_base_sha256=expected_base_hash,
            require_optimizer=False,
        )

        task_id = TASK_IDS[args.stage]
        env_cfg = parse_env_cfg(
            task_id, device=args.device, num_envs=args.num_envs
        )
        env_cfg.seed = args.seed
        disturbance_cfg = stage_disturbance_cfg(args.stage)
        initial_curriculum_step = (
            disturbance_cfg.curriculum_steps
            if args.full_scale_disturbance
            else 0
        )
        env = gym.make(task_id, cfg=env_cfg)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise TypeError(f"{task_id} did not create ManagerBasedRLEnv")
        wrapper = M1PandaTeacherEnvWrapper(
            env.unwrapped,
            stage=args.stage,
            base_actor=frozen_actor,
            seed=args.seed,
            disturbance_enabled=not args.disable_disturbance,
            initial_curriculum_step=initial_curriculum_step,
        )

        runner = OnPolicyRunner(
            wrapper,
            deepcopy(train_cfg),
            log_dir=None,
            device=env_cfg.sim.device,
        )
        runner.load(
            str(Path(args.checkpoint).expanduser().resolve()),
            load_optimizer=False,
            keep_std=True,
        )
        policy = runner.get_inference_policy(device=env_cfg.sim.device)
        observations, _ = wrapper.get_observations()

        step = 0
        done_count = 0
        reset_counts: dict[str, int | None] = {
            name: None for name in TERMINATION_NAMES
        }
        accumulator = TeacherEvaluationAccumulator(num_envs=args.num_envs)
        while simulation_app.is_running() and (
            args.steps == 0 or step < args.steps
        ):
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, _ = wrapper.step(actions)
            step += 1
            done_count += int(dones.sum().item())
            terms = _termination_terms(env)
            step_termination_counts = {
                name: int(terms[name].sum().item()) if name in terms else 0
                for name in TERMINATION_NAMES
            }
            accumulator.update(
                rewards=rewards,
                termination_counts=step_termination_counts,
            )
            update_reset_counts(env, reset_counts)
            if step % args.stats_interval == 0 or (
                args.steps > 0 and step == args.steps
            ):
                print(
                    format_play_stats(
                        step=step,
                        mean_reward=float(rewards.mean().item()),
                        done_count=done_count,
                        wrench_b=wrapper.current_wrench_b,
                        max_abs_wrench_seen=wrapper.max_abs_wrench_seen,
                        reset_counts=reset_counts,
                    ),
                    flush=True,
                )

        wrapper.assert_frozen_actor_unchanged()
        if args.summary_json is not None:
            summary = accumulator.finalize(
                checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                checkpoint_sha256=file_sha256(args.checkpoint),
                base_checkpoint_sha256=(
                    file_sha256(args.base_checkpoint)
                    if args.base_checkpoint is not None
                    else "not-applicable"
                ),
                seed=args.seed,
                steps=step,
                curriculum_scale=wrapper.curriculum_scale,
                axis_abs_wrench_seen=(
                    wrapper.axis_abs_wrench_seen.detach().cpu().tolist()
                ),
                frozen_actor_sha256=(
                    wrapper.frozen_actor_hash or "not-applicable"
                ),
            )
            if args.full_scale_disturbance:
                validate_full_scale_summary(summary)
            atomic_write_manifest(args.summary_json, summary)
        print(
            "[Teacher Play Final] "
            f"stage={args.stage} disturbance_enabled={wrapper.disturbance_enabled} "
            f"steps={step} observation_dim={observations.shape[1]} "
            f"action_dim={wrapper.num_actions} "
            f"max_abs_wrench_seen={wrapper.max_abs_wrench_seen:.6f} "
            f"frozen_actor_sha256={wrapper.frozen_actor_hash}",
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
