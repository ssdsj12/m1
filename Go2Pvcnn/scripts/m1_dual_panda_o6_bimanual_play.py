"""GUI playback of the deterministic M1 dual-Panda O6 MPC mission."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--diagnostics", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import go2_pvcnn.tasks  # noqa: F401, E402
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import M1DualPandaO6BimanualEnvCfg  # noqa: E402
from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import M1DualPandaO6BimanualWrapper  # noqa: E402


def main() -> int:
    cfg = M1DualPandaO6BimanualEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
    env.reset(seed=args.seed)
    wrapper = M1DualPandaO6BimanualWrapper(env)
    previous_phase = wrapper.runtime.mission.phase.name
    for step in range(args.max_steps):
        wrapper.step()
        phase = wrapper.runtime.mission.phase.name
        if args.diagnostics and (phase != previous_phase or phase in {"DONE", "TERMINATED"}):
            latest = wrapper.runtime.latest_solutions
            reasons = {
                key: None if value is None else value.diagnostics.fallback_reason
                for key, value in latest.items()
                if key != "arm"
            }
            print(
                f"step={step + 1} phase={phase} feasible={wrapper.last_command.feasible} "
                f"reasons={reasons} base_state={wrapper.last_snapshot.base_state.tolist()}",
                flush=True,
            )
        previous_phase = phase
        if phase in {"DONE", "TERMINATED"}:
            break
        if not simulation_app.is_running():
            break
    print(f"final_phase={wrapper.runtime.mission.phase.name}", flush=True)
    wrapper.close()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
