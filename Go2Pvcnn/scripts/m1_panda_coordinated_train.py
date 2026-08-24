#!/usr/bin/env python3
"""Train stable coordinated M1 + Panda PPO with automatic best rollback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
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


def build_arg_parser():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=600)
    parser.add_argument("--run_name", default="m1_panda_coordinated_train")
    parser.add_argument("--init-a1-checkpoint", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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


def build_manifest_contract(
    args,
    asset_path: Path,
    init_checkpoint: Path,
    train_cfg: dict,
) -> dict[str, object]:
    """Build the finite schema-2 contract before Isaac Sim is launched."""
    algorithm = train_cfg["algorithm"]
    policy = train_cfg["policy"]
    if algorithm["schedule"] != "adaptive":
        raise ValueError("coordinated PPO schedule must be adaptive")
    if train_cfg["num_steps_per_env"] != 256:
        raise ValueError("coordinated PPO rollout must contain 256 steps")
    if not 0 < int(args.max_iterations) <= 600:
        raise ValueError("max_iterations must be in [1, 600]")
    return {
        "schema_version": 2,
        "status": "starting",
        "task": "Isaac-M1-Panda-Coordinated-v0",
        "run_name": args.run_name,
        "device": args.device,
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "requested_iterations": int(args.max_iterations),
        "fresh_policy": True,
        "initialization_lineage_only": True,
        "zero_action_actor_initialization": True,
        "init_a1_checkpoint": str(init_checkpoint),
        "init_a1_checkpoint_sha256": sha256_file(init_checkpoint),
        "asset_path": str(asset_path),
        "asset_sha256": sha256_file(asset_path),
        "ppo": {
            "num_steps_per_env": int(train_cfg["num_steps_per_env"]),
            "save_interval": int(train_cfg["save_interval"]),
            "gamma": float(algorithm["gamma"]),
            "lambda": float(algorithm["lam"]),
            "schedule": algorithm["schedule"],
            "desired_kl": float(algorithm["desired_kl"]),
            "initial_learning_rate": float(algorithm["learning_rate"]),
            "learning_rate_bounds": [
                float(algorithm["min_learning_rate"]),
                float(algorithm["max_learning_rate"]),
            ],
            "initial_action_std": float(policy["init_noise_std"]),
            "action_std_bounds": [
                float(algorithm["clip_min_std"]),
                float(algorithm["clip_max_std"]),
            ],
        },
        "domain_randomization": {
            "enabled": True,
            "target_body": "panda_hand",
            "force_limit_n": 20.0,
            "torque_limit_nm": 5.0,
            "wrench_hold_s": [0.25, 1.0],
            "wrench_mode_probabilities": [0.50, 0.30, 0.20],
            "wrench_curriculum_scale": [0.10, 1.0],
            "wrench_curriculum_steps": 50_000,
            "root_xy_m": [-0.02, 0.02],
            "root_roll_pitch_rad": [-0.03, 0.03],
            "root_yaw_rad": [-0.05, 0.05],
            "root_linear_velocity_mps": [-0.05, 0.05],
            "root_angular_velocity_rad_s": [-0.10, 0.10],
            "leg_position_offset_rad": [-0.02, 0.02],
            "arm_position_offset_rad": [-0.03, 0.03],
            "controlled_velocity_rad_s": [-0.05, 0.05],
            "friction": [0.8, 1.2],
            "restitution": [0.0, 0.0],
            "friction_buckets": 64,
        },
        "guard": {
            "minimum_completed_episodes": 100,
            "eligible_timeout_rate": 0.90,
            "maximum_base_contact_rate": 0.05,
            "maximum_bad_orientation_rate": 0.05,
            "catastrophe_hard_failure_rate": 0.20,
            "catastrophe_updates": 25,
            "eligible_patience_updates": 50,
            "max_iterations": int(args.max_iterations),
        },
    }


def initialize_fresh_zero_action_policy(runner) -> None:
    """Make the safe implicit-actuator hold the fresh policy's exact baseline."""
    import torch

    output_layer = runner.alg.actor_critic.actor[-1]
    if not isinstance(output_layer, torch.nn.Linear) or output_layer.out_features != 23:
        raise RuntimeError("expected the coordinated actor to end in a 23-output Linear layer")
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    if not runner.alg.actor_critic.noise_parameter.requires_grad:
        raise RuntimeError("coordinated policy std must remain trainable")


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0 < args.max_iterations <= 600:
        raise ValueError("max_iterations must be in [1, 600]")
    init_checkpoint = args.init_a1_checkpoint.expanduser().resolve()
    if not init_checkpoint.is_file():
        raise FileNotFoundError(init_checkpoint)
    asset_path = (ROOT / "assets/m1_panda/m1_panda.usd").resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    run_dir = (ROOT / "logs/m1_panda_coordinated" / args.run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    from agent import get_m1_panda_coordinated_train_cfg

    train_cfg = get_m1_panda_coordinated_train_cfg()
    manifest = build_manifest_contract(args, asset_path, init_checkpoint, train_cfg)
    manifest.update(
        {
            "run_dir": str(run_dir),
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_json(manifest_path, manifest)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner
        from go2_pvcnn.tasks.m1_panda_coordinated_env_cfg import (
            COORDINATED_POLICY_OBSERVATION_DIM,
            configure_coordinated_training_domain_randomization,
        )
        from go2_pvcnn.tasks.m1_panda_coordinated_training_guard import (
            AtomicCheckpointController,
            TrainingGuard,
        )
        from go2_pvcnn.tasks.m1_panda_coordinated_wrapper import M1PandaCoordinatedEnvWrapper

        task = "Isaac-M1-Panda-Coordinated-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        configure_coordinated_training_domain_randomization(cfg, True)
        env = gym.make(task, cfg=cfg).unwrapped
        wrapper = M1PandaCoordinatedEnvWrapper(
            env, training_randomization=True, seed=args.seed
        )
        wrapper.reset()
        observations, _ = wrapper.get_observations()
        observation_dim = int(observations.shape[1])
        if observation_dim != 103 or observation_dim != COORDINATED_POLICY_OBSERVATION_DIM:
            raise RuntimeError(f"coordinated observation_dim != 103: {observation_dim}")
        if wrapper.num_actions != 23:
            raise RuntimeError(f"coordinated wrapper.num_actions != 23: {wrapper.num_actions}")
        manifest.update(
            {
                "status": "running",
                "observation_dim": observation_dim,
                "action_dim": wrapper.num_actions,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print({"run_manifest": str(manifest_path), "observation_dim": observation_dim, "action_dim": wrapper.num_actions})
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(run_dir), device=args.device)
        initialize_fresh_zero_action_policy(runner)
        controller = AtomicCheckpointController(
            run_dir, TrainingGuard(max_iterations=args.max_iterations)
        )
        learn_result = runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
            iteration_callback=partial(controller.on_iteration, runner),
        )
        stop_iteration = runner.current_learning_iteration
        final_fields = controller.finalize(runner, learn_result.stop_reason)
        manifest.update(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_iterations": learn_result.completed_iterations,
                "stop_iteration": stop_iteration,
            }
        )
        manifest.update(final_fields)
        atomic_write_json(manifest_path, manifest)
        print(
            {
                "task": task,
                "observation_dim": observation_dim,
                "action_dim": wrapper.num_actions,
                "iterations": learn_result.completed_iterations,
                "stop_reason": learn_result.stop_reason,
                "accepted": final_fields["accepted"],
                "final_checkpoint": final_fields["final_checkpoint"],
            }
        )
        return 0
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
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
