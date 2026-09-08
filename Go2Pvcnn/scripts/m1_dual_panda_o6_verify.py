"""Run the frozen four-layer verification ladder for M1 + dual Panda + O6."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"

CPU_TESTS = (
    "tests/test_m1_dual_panda_o6_contracts.py",
    "tests/test_m1_dual_panda_o6_asset_static.py",
    "tests/test_m1_dual_panda_o6_env_static.py",
    "tests/test_m1_dual_panda_o6_verification.py",
    "tests/test_m1_bimanual_frame_kinematics.py",
    "tests/test_m1_bimanual_o6_contact.py",
    "tests/test_m1_bimanual_reduced_dynamics.py",
    "tests/test_m1_bimanual_full_action_teacher.py",
    "tests/test_m1_bimanual_runtime.py",
    "tests/test_m1_bimanual_state_machine.py",
)

PURE_QP_TESTS = (
    "tests/test_m1_bimanual_palm_orientation_mpc.py",
    "tests/test_m1_bimanual_object_mpc.py",
    "tests/test_m1_bimanual_dual_arm_mpc.py",
    "tests/test_m1_bimanual_hand_mpc.py",
    "tests/test_m1_bimanual_whole_body_qp.py",
    "tests/test_m1_bimanual_safety_projection.py",
)


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _pytest(files: tuple[str, ...]) -> None:
    _run([sys.executable, "-m", "pytest", "-q", *files])


def _smoke(output_dir: Path, steps: int) -> None:
    _run(
        [
            sys.executable,
            str(PROBE),
            "--headless",
            "--device",
            "cuda:0",
            "--mode",
            "teacher",
            "--steps",
            str(steps),
            "--seed",
            "7",
            "--report",
            str(output_dir / "gpu0_smoke.json"),
        ],
    )


def _formal(output_dir: Path, steps: int) -> None:
    _run(
        [
            sys.executable,
            str(PROBE),
            "--headless",
            "--device",
            "cuda:0",
            "--mode",
            "teacher",
            "--steps",
            str(steps),
            "--seeds",
            "42",
            "43",
            "44",
            "--trials-per-seed",
            "10",
            "--report",
            str(output_dir / "formal_report.json"),
            "--jsonl",
            str(output_dir / "formal_trials.jsonl"),
            "--manifest",
            str(output_dir / "formal_aggregate.manifest.json"),
        ],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer",
        choices=("cpu", "qp", "smoke", "formal", "all"),
        default="cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tests/artifacts/m1_dual_panda_o6_verification",
    )
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--formal-steps", type=int, default=2000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.smoke_steps <= 0 or args.formal_steps <= 0:
        raise SystemExit("step counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = ("cpu", "qp", "smoke", "formal") if args.layer == "all" else (args.layer,)
    for layer in selected:
        if layer == "cpu":
            _pytest(CPU_TESTS)
        elif layer == "qp":
            _pytest(PURE_QP_TESTS)
        elif layer == "smoke":
            _smoke(output_dir, args.smoke_steps)
        else:
            _formal(output_dir, args.formal_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
