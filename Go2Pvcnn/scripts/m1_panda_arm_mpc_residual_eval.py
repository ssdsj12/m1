#!/usr/bin/env python3
"""Run one isolated zero-pair or candidate residual evaluation worker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
for path in (ROOT, ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.training.m1_panda_arm_mpc_residual_lineage import (
    ResidualSourcePaths,
    sha256_file,
    source_lineage,
)


TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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


def build_arg_parser(*, include_app_launcher_args: bool = True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero-pair", "candidate"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--num_envs", type=int, default=1)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _metrics(document: dict[str, float]):
    from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import ResidualEvalMetrics

    return ResidualEvalMetrics(
        hard_failure_count=int(round(document["hard_failure_count"])),
        mpc_feasible_rate=document["mpc_feasible_rate"],
        qp_feasible_rate=document["qp_feasible_rate"],
        four_contact_rate=document["four_contact_rate"],
        roll_pitch_rms=document["roll_pitch_rms"],
        base_height_rms=document["base_height_rms"],
        ee_position_error=document["ee_position_error"],
        ee_orientation_error=document["ee_orientation_error"],
        wrench_error=document["wrench_error"],
        slip=document["slip"],
        intervention_ratio=document["intervention_ratio"],
        saturation_fraction=tuple(
            document[f"saturation_fraction_{index}"] for index in range(8)
        ),
    )


@torch.inference_mode()
def _rollout(wrapper, *, steps: int, policy=None):
    observations, _ = wrapper.reset()
    for _ in range(steps):
        actions = (
            torch.zeros((wrapper.num_envs, 8), device=wrapper.device)
            if policy is None
            else policy(observations)
        )
        observations, _, _, _ = wrapper.step(actions)
    return _metrics(wrapper.get_training_diagnostics())


def main() -> int:
    args = build_arg_parser().parse_args()
    checkpoint = None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
    if args.mode == "candidate":
        if checkpoint is None:
            raise ValueError("candidate mode requires --checkpoint")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    elif checkpoint is not None:
        raise ValueError("zero-pair mode does not accept --checkpoint")
    if args.num_envs != 1:
        raise ValueError("formal fixed-seed evaluation requires num_envs=1")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    output = (
        Path.cwd() / f"residual_eval_seed{args.seed}.json"
        if args.output_json is None
        else args.output_json.expanduser().resolve()
    )
    source_paths = ResidualSourcePaths(
        ROOT / "assets/m1_panda/m1_panda.usd",
        ROOT / "agent/m1_panda_arm_mpc_residual_train_cfg.py",
        ROOT / "go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py",
        ROOT / "go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py",
    )

    manifest = {
        "schema_version": 2,
        "status": "starting",
        "mode": args.mode,
        "task": TASK_ID,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else sha256_file(checkpoint),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **source_lineage(source_paths),
    }
    atomic_write_json(output, manifest)

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from agent import get_m1_panda_arm_mpc_residual_train_cfg
        from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
            M1PandaArmMpcResidualEnvWrapper,
        )
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
        cfg.seed = args.seed
        cfg.episode_length_s = max(
            float(cfg.episode_length_s), args.steps * cfg.sim.dt * cfg.decimation + 1.0
        )
        env = gym.make(TASK_ID, cfg=cfg).unwrapped
        wrapper = M1PandaArmMpcResidualEnvWrapper(env, seed=args.seed)
        wrapper.reset()
        policy = None
        if args.mode == "candidate":
            runner = OnPolicyRunner(
                wrapper,
                get_m1_panda_arm_mpc_residual_train_cfg(),
                log_dir=None,
                device=args.device,
            )
            runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
            policy = runner.get_inference_policy(device=args.device)
        baseline = _rollout(wrapper, steps=args.steps, policy=None)
        candidate = _rollout(wrapper, steps=args.steps, policy=policy)
        manifest.update(
            {
                "status": "complete",
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(output, manifest)
        return 0
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(output, manifest)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
