#!/usr/bin/env python3
"""Run the folded-load stages sequentially with strict atomic rollback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, NamedTuple


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import STAGE_ORDER, stage_spec
from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import sha256_file


EVALUATION_SEEDS = (42, 43, 44)
TRAIN_SCRIPT = ROOT / "scripts/m1_panda_folded_load_train.py"
EVAL_SCRIPT = ROOT / "scripts/m1_panda_folded_load_eval.py"


class ExecutionRequest(NamedTuple):
    stage: str
    run_dir: Path
    parent_manifest: Path | None


class CurriculumState(NamedTuple):
    status: str
    completed_stages: tuple[str, ...]
    stopped_stage: str | None
    rollback_stage: str | None
    rollback_checkpoint: str | None
    reason: str | None


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
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
        return path
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _document_path(raw: object, manifest: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def validate_lineage(
    experiment_root: str | os.PathLike[str], *, through_stage: str
) -> Path:
    """Verify every accepted manifest/checkpoint from L0 through one stage."""
    root = Path(experiment_root).expanduser().resolve()
    if through_stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {through_stage!r}")
    previous_manifest = None
    previous_document = None
    for stage in STAGE_ORDER[: STAGE_ORDER.index(through_stage) + 1]:
        manifest = root / stage / "run_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing accepted stage {stage}: {manifest}")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if document.get("stage") != stage:
            raise ValueError(f"manifest stage mismatch for {stage}")
        if document.get("accepted") is not True:
            raise ValueError(f"stage {stage} must contain accepted=true")
        checkpoint = _document_path(
            document.get("final_checkpoint"), manifest, f"{stage} final_checkpoint"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if document.get("final_checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError(f"{stage} final checkpoint SHA mismatch")
        contract = stage_spec(stage)
        if previous_manifest is None:
            if contract.parent is not None:
                raise ValueError(f"lineage cannot begin at non-root stage {stage}")
            for field in (
                "parent_stage", "parent_manifest", "parent_manifest_sha256",
                "parent_checkpoint", "parent_checkpoint_sha256",
            ):
                if document.get(field) is not None:
                    raise ValueError(f"L0-C0 {field} must be null")
        else:
            if contract.parent != previous_document["stage"]:
                raise ValueError(f"{stage} is not the immediate child of accepted lineage")
            if document.get("parent_stage") != contract.parent:
                raise ValueError(f"{stage} parent stage mismatch")
            recorded_manifest = _document_path(
                document.get("parent_manifest"), manifest, f"{stage} parent_manifest"
            )
            if recorded_manifest != previous_manifest:
                raise ValueError(f"{stage} parent manifest path mismatch")
            if document.get("parent_manifest_sha256") != sha256_file(previous_manifest):
                raise ValueError(f"{stage} parent manifest SHA mismatch")
            previous_checkpoint = _document_path(
                previous_document.get("final_checkpoint"),
                previous_manifest,
                f"{contract.parent} final_checkpoint",
            )
            if document.get("parent_checkpoint_sha256") != sha256_file(previous_checkpoint):
                raise ValueError(f"{stage} parent checkpoint SHA mismatch")
            recorded_checkpoint = _document_path(
                document.get("parent_checkpoint"), manifest, f"{stage} parent_checkpoint"
            )
            if recorded_checkpoint != previous_checkpoint:
                raise ValueError(f"{stage} parent checkpoint path mismatch")
        previous_manifest = manifest.resolve()
        previous_document = document
    return previous_manifest


class ProcessStageExecutor:
    """Launch one train process followed by the three fixed evaluations."""

    def __init__(
        self, *, num_envs: int, max_iterations: int, device: str,
        headless: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.max_iterations = int(max_iterations)
        self.device = str(device)
        self.headless = bool(headless)

    @staticmethod
    def _run(command: list[str]) -> bool:
        result = subprocess.run(command, cwd=ROOT, check=False)
        return result.returncode == 0

    def __call__(self, request: ExecutionRequest) -> bool:
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--stage", request.stage,
            "--run_dir", str(request.run_dir),
            "--num_envs", str(self.num_envs),
            "--max_iterations", str(self.max_iterations),
            "--device", self.device,
        ]
        if request.parent_manifest is not None:
            command.extend(("--parent_manifest", str(request.parent_manifest)))
        if self.headless:
            command.append("--headless")
        if not self._run(command):
            return False
        for seed in EVALUATION_SEEDS:
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--stage", request.stage,
                "--run_dir", str(request.run_dir),
                "--seed", str(seed),
                "--num_envs", "64",
                "--device", self.device,
            ]
            if self.headless:
                command.append("--headless")
            if not self._run(command):
                return False
        return True


def _persist(root: Path, state: CurriculumState) -> None:
    _atomic_json(root / "curriculum_state.json", state._asdict())


def _accepted_checkpoint(manifest: Path | None) -> str | None:
    if manifest is None:
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return str(_document_path(document.get("final_checkpoint"), manifest, "final_checkpoint"))


def run_curriculum(
    experiment_root: str | os.PathLike[str],
    *,
    start_stage: str = "L0-C0",
    executor: Callable[[ExecutionRequest], bool],
) -> CurriculumState:
    """Execute accepted stages only; stop at the first failed promotion."""
    if start_stage not in STAGE_ORDER:
        raise ValueError(f"unknown start stage {start_stage!r}")
    root = Path(experiment_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    start_index = STAGE_ORDER.index(start_stage)
    previous_manifest = None
    completed = list(STAGE_ORDER[:start_index])
    if start_index:
        previous_stage = STAGE_ORDER[start_index - 1]
        previous_manifest = validate_lineage(root, through_stage=previous_stage)

    for stage in STAGE_ORDER[start_index:]:
        run_dir = root / stage
        request = ExecutionRequest(stage, run_dir, previous_manifest)
        try:
            launched = bool(executor(request))
            if launched:
                current_manifest = validate_lineage(root, through_stage=stage)
            else:
                current_manifest = None
        except BaseException as error:
            state = CurriculumState(
                status="stopped",
                completed_stages=tuple(completed),
                stopped_stage=stage,
                rollback_stage=None if previous_manifest is None else completed[-1],
                rollback_checkpoint=_accepted_checkpoint(previous_manifest),
                reason=f"{type(error).__name__}: {error}",
            )
            _persist(root, state)
            return state
        if not launched or current_manifest is None:
            state = CurriculumState(
                status="stopped",
                completed_stages=tuple(completed),
                stopped_stage=stage,
                rollback_stage=None if previous_manifest is None else completed[-1],
                rollback_checkpoint=_accepted_checkpoint(previous_manifest),
                reason="stage process or acceptance failed",
            )
            _persist(root, state)
            return state
        completed.append(stage)
        previous_manifest = current_manifest
        _persist(
            root,
            CurriculumState(
                status="running",
                completed_stages=tuple(completed),
                stopped_stage=None,
                rollback_stage=stage,
                rollback_checkpoint=_accepted_checkpoint(previous_manifest),
                reason=None,
            ),
        )

    state = CurriculumState(
        status="accepted",
        completed_stages=tuple(completed),
        stopped_stage=None,
        rollback_stage=completed[-1],
        rollback_checkpoint=_accepted_checkpoint(previous_manifest),
        reason=None,
    )
    _persist(root, state)
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--start_stage", choices=STAGE_ORDER, default="L0-C0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0 < args.max_iterations <= 600:
        raise ValueError("max_iterations must be in [1, 600]")
    executor = ProcessStageExecutor(
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        device=args.device,
        headless=args.headless,
    )
    state = run_curriculum(
        args.experiment_root, start_stage=args.start_stage, executor=executor
    )
    print(json.dumps(state._asdict(), indent=2))
    return 0 if state.status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
