"""Collect fixed-base full-action teacher rows for z16 training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", nargs="+", type=int, required=True)
parser.add_argument("--trials-per-seed", type=int, default=10)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

import gymnasium as gym
import torch

import go2_pvcnn.tasks  # noqa: F401, E402
from go2_pvcnn.assets.m1_dual_panda_o6 import (  # noqa: E402
    M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES,
)
from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import (  # noqa: E402
    ACTION_DIM,
    HORIZON,
    STATE_DIM,
    TASK_FEATURE_DIM,
    pack_state_features,
    pack_teacher_task_features,
)
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import (  # noqa: E402
    M1DualPandaO6BimanualEnvCfg,
)
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import (  # noqa: E402
    M1DualPandaO6BimanualWrapper,
)


def _empty_success() -> dict[str, np.ndarray]:
    return {
        "state": np.empty((0, STATE_DIM), dtype=np.float32),
        "teacher_action": np.empty((0, HORIZON, ACTION_DIM), dtype=np.float32),
        "teacher_task": np.empty((0, HORIZON, TASK_FEATURE_DIM), dtype=np.float32),
        "phase": np.empty((0, 1), dtype=np.float32),
        "seed": np.empty((0,), dtype=np.int64),
        "trial": np.empty((0,), dtype=np.int64),
        "action_order": np.asarray(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES),
    }


def _stack_rows(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not rows:
        return _empty_success()
    return {
        "state": np.stack([row["state"] for row in rows]).astype(np.float32),
        "teacher_action": np.stack([row["teacher_action"] for row in rows]).astype(np.float32),
        "teacher_task": np.stack([row["teacher_task"] for row in rows]).astype(np.float32),
        "phase": np.asarray([[row["phase"]] for row in rows], dtype=np.float32),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "trial": np.asarray([row["trial"] for row in rows], dtype=np.int64),
        "action_order": np.asarray(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES),
    }


def _stack_failures(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    values = _stack_rows(rows)
    values["reason"] = np.asarray([str(row["reason"]) for row in rows])
    return values


def main() -> int:
    if args.trials_per_seed <= 0 or args.steps <= 0:
        parser.error("trials and steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = M1DualPandaO6BimanualEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            for trial in range(args.trials_per_seed):
                wrapper = M1DualPandaO6BimanualWrapper(env)
                wrapper.reset(seed=seed)
                trial_rows: list[dict[str, object]] = []
                for step in range(args.steps):
                    wrapper.step()
                    if step % 8 != 0:
                        continue
                    teacher = wrapper.last_teacher_solution
                    if teacher is None:
                        raise RuntimeError("teacher mode did not publish last_teacher_solution")
                    row: dict[str, object] = {
                        "state": pack_state_features(wrapper.last_snapshot).numpy(),
                        "teacher_action": teacher.action_trajectory.to(dtype=torch.float32).numpy(),
                        "teacher_task": pack_teacher_task_features(teacher).numpy(),
                        "phase": float(wrapper.runtime.mission.phase.value),
                        "seed": seed,
                        "trial": trial,
                        "reason": teacher.diagnostics.fallback_reason or "mission_not_done",
                    }
                    if teacher.diagnostics.feasible:
                        trial_rows.append(row)
                    else:
                        failures.append(row)
                    if wrapper.runtime.mission.phase.name in {"DONE", "TERMINATED"}:
                        break
                if wrapper.runtime.mission.phase.name == "DONE":
                    successes.extend(trial_rows)
                else:
                    failures.extend(trial_rows)
        np.savez_compressed(
            args.output_dir / "teacher_success.npz", **_stack_rows(successes)
        )
        np.savez_compressed(
            args.output_dir / "teacher_failures.npz", **_stack_failures(failures)
        )
    finally:
        env.close()
    print(
        f"successful_rows={len(successes)} failure_rows={len(failures)}",
        flush=True,
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
