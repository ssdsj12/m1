#!/usr/bin/env python3
"""Train the combined 23-effort M1 + Panda coordinated PPO prerequisite."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
    parser.add_argument("--max_iterations", type=int, default=100)
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


def initialize_fresh_zero_action_policy(runner) -> None:
    """Make the safe implicit-actuator hold the fresh policy's exact baseline."""
    import torch

    output_layer = runner.alg.actor_critic.actor[-1]
    if not isinstance(output_layer, torch.nn.Linear) or output_layer.out_features != 23:
        raise RuntimeError("expected the coordinated actor to end in a 23-output Linear layer")
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    runner.alg.actor_critic.noise_parameter.requires_grad_(False)


def main() -> int:
    args = build_arg_parser().parse_args()
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
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "starting",
        "task": "Isaac-M1-Panda-Coordinated-v0",
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "device": args.device,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "requested_iterations": args.max_iterations,
        "fresh_policy": True,
        "initialization_lineage_only": True,
        "init_a1_checkpoint": str(init_checkpoint),
        "init_a1_checkpoint_sha256": sha256_file(init_checkpoint),
        "asset_path": str(asset_path),
        "asset_sha256": sha256_file(asset_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
        from agent import get_m1_panda_teacher_train_cfg
        from go2_pvcnn.tasks.m1_panda_coordinated_env_cfg import COORDINATED_POLICY_OBSERVATION_DIM
        from go2_pvcnn.tasks.m1_panda_coordinated_wrapper import M1PandaCoordinatedEnvWrapper

        task = "Isaac-M1-Panda-Coordinated-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(task, cfg=cfg).unwrapped
        wrapper = M1PandaCoordinatedEnvWrapper(env)
        observations, _ = wrapper.get_observations()
        observation_dim = int(observations.shape[1])
        if observation_dim != 103 or observation_dim != COORDINATED_POLICY_OBSERVATION_DIM:
            raise RuntimeError(f"coordinated observation_dim != 103: {observation_dim}")
        if wrapper.num_actions != 23:
            raise RuntimeError(f"coordinated wrapper.num_actions != 23: {wrapper.num_actions}")
        train_cfg = deepcopy(get_m1_panda_teacher_train_cfg())
        train_cfg["policy"]["actor_hidden_dims"] = [256, 128]
        train_cfg["policy"]["critic_hidden_dims"] = [256, 128]
        train_cfg["save_interval"] = 100
        train_cfg["algorithm"]["learning_rate"] = 1.0e-4
        train_cfg["algorithm"]["schedule"] = "fixed"
        manifest.update(
            {
                "status": "running",
                "observation_dim": observation_dim,
                "action_dim": wrapper.num_actions,
                "save_interval": train_cfg["save_interval"],
                "learning_rate": train_cfg["algorithm"]["learning_rate"],
                "learning_rate_schedule": train_cfg["algorithm"]["schedule"],
                "zero_action_actor_initialization": True,
                "frozen_action_std": 0.01,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print({"run_manifest": str(manifest_path), "observation_dim": observation_dim, "action_dim": wrapper.num_actions})
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(run_dir), device=args.device)
        initialize_fresh_zero_action_policy(runner)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime_ns)
        if not checkpoints:
            raise RuntimeError("training completed without a checkpoint")
        final_checkpoint = checkpoints[-1]
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_iterations": args.max_iterations,
                "final_checkpoint": str(final_checkpoint),
                "final_checkpoint_sha256": sha256_file(final_checkpoint),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print({"task": task, "observation_dim": observation_dim, "action_dim": wrapper.num_actions, "init_a1_checkpoint": str(init_checkpoint), "iterations": args.max_iterations, "final_checkpoint": str(final_checkpoint)})
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
