#!/usr/bin/env python3
"""Train the 15D dual-Panda/O6 wire policy in approval or action stage."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from typing import Any


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = ROOT / "rsl_rl"
DEFAULT_CONFIG = ROOT / "config" / "m1_wire_ppo_train.yaml"
DEFAULT_OBJECT_CATALOG = ROOT / "config" / "m1_object_catalog.json"
DEFAULT_OBJECT_ASSETS_ROOT = ROOT / "assets" / "m1_objects"
for import_path in (ROOT, RSL_RL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def configure_runtime_environment() -> None:
    """Pin rollout/training to GPU0 and prevent CPU helper oversubscription."""

    for name, value in RUNTIME_ENVIRONMENT.items():
        os.environ[name] = value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def manifest_body_sha256(document: dict[str, object]) -> str:
    body = deepcopy(document)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_write_manifest(path: Path, body: dict[str, object]) -> dict[str, object]:
    document = deepcopy(body)
    document.pop("manifest_sha256", None)
    document["manifest_sha256"] = manifest_body_sha256(document)
    _atomic_bytes(Path(path), _canonical_json(document))
    return document


def load_train_config(path: Path) -> dict[str, Any]:
    import yaml

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training config does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("training config must be a mapping")
    if document.get("schema_version") != 1:
        raise ValueError("training config schema_version must be 1")
    if document.get("action_dim") != 15:
        raise ValueError("wire PPO action_dim must be exactly 15")
    runner = document.get("runner")
    rewards = document.get("rewards")
    if not isinstance(runner, dict) or not isinstance(rewards, dict):
        raise ValueError("training config requires runner and rewards mappings")
    if set(rewards) != {"approval", "action"}:
        raise ValueError("training config rewards must define approval and action")
    return document


def build_arg_parser(*, include_app_launcher_args: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # AppLauncher probes the parser before the final parse, so requiredness is
    # enforced in validate_cli_contract rather than argparse declarations.
    parser.add_argument("--stage", choices=("approval", "action"), default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object-id", default="wire_000")
    parser.add_argument("--object-catalog", type=Path, default=DEFAULT_OBJECT_CATALOG)
    parser.add_argument("--object-assets-root", type=Path, default=DEFAULT_OBJECT_ASSETS_ROOT)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(device="cuda:0")
    else:
        parser.add_argument("--device", default="cuda:0")
    return parser


def validate_cli_contract(args: argparse.Namespace) -> None:
    if args.stage not in {"approval", "action"}:
        raise ValueError("--stage is required")
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    for name in ("num_envs", "max_iterations"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    # Task 3 intentionally exposes one physical lane. Refuse fake parallelism
    # until that environment contract is vectorized rather than broadcasting.
    if args.num_envs != 1:
        raise ValueError("--num-envs must be 1 for the current single-lane M1WirePpoEnv")
    if args.device != "cuda:0":
        raise ValueError("--device must be cuda:0; wire PPO training is pinned to GPU0")
    if args.stage == "action" and args.resume_checkpoint is None:
        raise ValueError("--stage action requires --resume-checkpoint from Stage A approval")
    if args.stage == "approval" and args.resume_checkpoint is not None:
        raise ValueError("--resume-checkpoint is reserved for --stage action")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid approval manifest: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("approval manifest must be a JSON object")
    declared = document.get("manifest_sha256")
    if not isinstance(declared, str) or declared != manifest_body_sha256(document):
        raise ValueError("approval manifest SHA-256 mismatch")
    return document


def validate_approval_resume(checkpoint: Path) -> dict[str, object]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"approval checkpoint does not exist: {checkpoint}")
    manifest_path = checkpoint.parent / "training.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"approval training manifest does not exist: {manifest_path}")
    document = _load_manifest(manifest_path)
    if document.get("schema_version") != 1 or document.get("stage") != "approval":
        raise ValueError("resume artifact is not a Stage A approval manifest")
    if document.get("status") != "completed" or document.get("action_dim") != 15:
        raise ValueError("Stage A approval manifest is incomplete or incompatible")
    counts = document.get("acceptance_counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("accepted"), int) or counts["accepted"] < 1:
        raise ValueError("Stage A approval manifest has no accepted episode")
    best = document.get("best_checkpoint")
    if not isinstance(best, dict):
        raise ValueError("Stage A approval manifest has no best checkpoint")
    declared_path = best.get("path")
    declared_sha = best.get("sha256")
    if not isinstance(declared_path, str) or not isinstance(declared_sha, str):
        raise ValueError("Stage A best checkpoint record is malformed")
    resolved = (manifest_path.parent / declared_path).resolve()
    if resolved != checkpoint:
        raise ValueError("resume checkpoint does not match Stage A best checkpoint")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != declared_sha:
        raise ValueError("approval checkpoint SHA-256 mismatch")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "approval_seed": document.get("seed"),
    }


def build_training_manifest(
    *,
    stage: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    device: str,
    run_dir: Path,
    config_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "starting",
        "stage": stage,
        "seed": int(seed),
        "device": device,
        "num_envs": int(num_envs),
        "requested_iterations": int(max_iterations),
        "action_dim": 15,
        "source": str(THIS_FILE),
        "source_sha256": sha256_file(THIS_FILE),
        "config": str(Path(config_path).expanduser().resolve()),
        "config_sha256": sha256_file(Path(config_path).expanduser().resolve()),
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "episodes_jsonl": "episodes.jsonl",
        "acceptance_counts": {"accepted": 0, "rejected": 0},
        "terminal_reason_counts": {},
        "best_checkpoint": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _json_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("episode diagnostics must be finite")
        return value
    try:
        item = value.item()  # torch/numpy scalar without importing either here
    except (AttributeError, ValueError, RuntimeError):
        return str(value)
    return _json_scalar(item)


class EpisodeRecorder:
    """Durable successful/failed episode ledger consumed by both stages."""

    _FIELDS = (
        "success", "failure_reason", "reward", "terminated", "truncated",
        "contacts_left", "contacts_right", "lift_height", "hold_time", "slip",
        "phase", "stage",
    )

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to reuse episode ledger: {self.path}")
        _atomic_bytes(self.path, b"")
        self.acceptance_counts = {"accepted": 0, "rejected": 0}
        self.terminal_reason_counts: Counter[str] = Counter()
        self.episode_count = 0

    def record(self, info: dict[str, object]) -> dict[str, object]:
        success = info.get("success") is True
        reason = "accepted" if success else str(info.get("failure_reason") or (
            "truncated" if info.get("truncated") else "terminated"
        ))
        row = {"episode_index": self.episode_count, "terminal_reason": reason}
        for name in self._FIELDS:
            if name in info:
                row[name] = _json_scalar(info[name])
        payload = _canonical_json(row)
        with self.path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        key = "accepted" if success else "rejected"
        self.acceptance_counts[key] += 1
        self.terminal_reason_counts[reason] += 1
        self.episode_count += 1
        return row


def atomic_publish_checkpoint(runner: object, target: Path) -> str:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(name)
        runner.save(str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, target)
        return digest
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class WireRslRlEnv:
    """One-lane adapter from Gymnasium-style ``M1WirePpoEnv`` to RSL-RL."""

    def __init__(self, env: object, recorder: EpisodeRecorder, *, seed: int, torch_module) -> None:
        self.env = env
        self.recorder = recorder
        self.seed = int(seed)
        self._torch = torch_module
        self.num_envs = 1
        self.num_actions = 15
        self.max_episode_length = int(env.max_episode_steps)
        self.device = str(env.device)
        self.cfg = {"stage": env.stage.value, "action_dim": self.num_actions}
        self.episode_length_buf = torch_module.zeros(1, dtype=torch_module.long, device=env.device)
        self._episode_index = 0
        observation, _ = env.reset(seed=self.seed)
        self._observation = observation.reshape(1, -1)

    def get_observations(self):
        return self._observation.clone(), {"observations": {}}

    def get_training_diagnostics(self):
        return {
            "accepted_episodes": float(self.recorder.acceptance_counts["accepted"]),
            "rejected_episodes": float(self.recorder.acceptance_counts["rejected"]),
        }

    def reset(self):
        observation, _ = self.env.reset(seed=self.seed + self._episode_index)
        self._observation = observation.reshape(1, -1)
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(self, actions):
        torch = self._torch
        if tuple(actions.shape) != (1, self.num_actions):
            raise ValueError(f"RSL-RL actions must have shape (1, {self.num_actions})")
        action = actions[0].to(device=self.env.device, dtype=torch.float32).clamp(-1.0, 1.0)
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        infos: dict[str, object] = {"observations": {}}
        if done:
            terminal = dict(info)
            terminal.update({"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
            self.recorder.record(terminal)
            infos["episode"] = {
                "success": torch.tensor([float(info.get("success", False))], device=self.env.device),
                "lift_height": torch.tensor([float(info.get("lift_height", 0.0))], device=self.env.device),
                "hold_time": torch.tensor([float(info.get("hold_time", 0.0))], device=self.env.device),
                "slip": torch.tensor([float(info.get("slip", 0.0))], device=self.env.device),
            }
            self._episode_index += 1
            observation, _ = self.env.reset(seed=self.seed + self._episode_index)
            self.episode_length_buf.zero_()
        else:
            self.episode_length_buf += 1
        self._observation = observation.reshape(1, -1)
        return (
            self._observation.clone(),
            torch.tensor([float(reward)], dtype=torch.float32, device=self.env.device),
            torch.tensor([done], dtype=torch.bool, device=self.env.device),
            infos,
        )

    def close(self) -> None:
        self.env.close()


def _seed_everything(seed: int, torch) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _reward_cfg(config: dict[str, Any], stage: str):
    from go2_pvcnn.tasks.m1_wire_ppo_curriculum import WirePpoRewardCfg

    value = config["rewards"][stage]
    if not isinstance(value, dict):
        raise ValueError(f"rewards.{stage} must be a mapping")
    return WirePpoRewardCfg(**value)


def main() -> int:
    configure_runtime_environment()
    args = build_arg_parser(include_app_launcher_args=True).parse_args()
    validate_cli_contract(args)
    config_path = args.config.expanduser().resolve()
    config = load_train_config(config_path)
    resume_lineage = (
        validate_approval_resume(args.resume_checkpoint)
        if args.stage == "action" else None
    )
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "training.manifest.json"
    manifest = build_training_manifest(
        stage=args.stage, seed=args.seed, num_envs=args.num_envs,
        max_iterations=args.max_iterations, device=args.device,
        run_dir=run_dir, config_path=config_path,
    )
    if resume_lineage is not None:
        manifest["approval_lineage"] = resume_lineage
    manifest = atomic_write_manifest(manifest_path, manifest)
    recorder = EpisodeRecorder(run_dir / "episodes.jsonl")
    app = None
    physical_env = None
    wrapped_env = None
    try:
        import torch
        from isaaclab.app import AppLauncher

        _seed_everything(args.seed, torch)
        app = AppLauncher(args).app
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner
        from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_contracts import WIRE_PPO_ACTION_DIM
        from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import M1DualPandaO6BimanualWrapper
        from go2_pvcnn.tasks.m1_wire_ppo_curriculum import WirePpoStage
        from go2_pvcnn.tasks.m1_wire_ppo_env import M1WirePpoEnv
        from scripts import m1_dual_panda_o6_bimanual_play as play

        if WIRE_PPO_ACTION_DIM != config["action_dim"]:
            raise RuntimeError("runtime wire PPO action dimension does not match config")
        catalog, instances = play._prepare_object_scene(
            object_id=args.object_id,
            object_catalog=args.object_catalog,
            object_assets_root=args.object_assets_root,
        )
        cfg = play.build_play_config(
            seed=args.seed, object_id=args.object_id,
            catalog=catalog, object_instances=instances,
        )
        physical_env = gym.make(config["task_id"], cfg=cfg)
        wrapper = M1DualPandaO6BimanualWrapper(
            physical_env, mode="teacher", object_id=args.object_id
        )
        wrapper.runtime.wire_object_id = args.object_id
        wrapper.runtime.wire_target_prim_path = cfg.target_contact_prim_path.replace(
            "{ENV_REGEX_NS}", "/World/envs/env_0"
        )
        play._configure_reduced_wire_runtime(wrapper)
        ppo_env = M1WirePpoEnv(
            wrapper,
            stage=WirePpoStage(args.stage),
            reward_cfg=_reward_cfg(config, args.stage),
            max_episode_steps=int(config["max_episode_steps"]),
            device=args.device,
        )
        wrapped_env = WireRslRlEnv(ppo_env, recorder, seed=args.seed, torch_module=torch)
        runner_cfg = deepcopy(config["runner"])
        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(run_dir), device=args.device)
        if resume_lineage is not None:
            runner.load(str(args.resume_checkpoint), load_optimizer=True)
        manifest.update({
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "observation_dim": int(ppo_env.observation_dim),
        })
        manifest = atomic_write_manifest(manifest_path, manifest)
        previous_best = 0

        def on_iteration(_summary):
            nonlocal manifest, previous_best
            accepted = recorder.acceptance_counts["accepted"]
            if accepted > previous_best:
                target = run_dir / "best.pt"
                digest = atomic_publish_checkpoint(runner, target)
                manifest["best_checkpoint"] = {"path": target.name, "sha256": digest}
                previous_best = accepted
            manifest["acceptance_counts"] = dict(recorder.acceptance_counts)
            manifest["terminal_reason_counts"] = dict(sorted(recorder.terminal_reason_counts.items()))
            manifest = atomic_write_manifest(manifest_path, manifest)
            return None

        result = runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=False,
            iteration_callback=on_iteration,
        )
        manifest.update({
            "status": "completed",
            "completed_iterations": int(result.completed_iterations),
            "stop_reason": result.stop_reason or "max_iterations",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "acceptance_counts": dict(recorder.acceptance_counts),
            "terminal_reason_counts": dict(sorted(recorder.terminal_reason_counts.items())),
            "episodes_jsonl_sha256": sha256_file(recorder.path),
        })
        atomic_write_manifest(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest.update({
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "acceptance_counts": dict(recorder.acceptance_counts),
            "terminal_reason_counts": dict(sorted(recorder.terminal_reason_counts.items())),
        })
        atomic_write_manifest(manifest_path, manifest)
        raise
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif physical_env is not None:
            physical_env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
