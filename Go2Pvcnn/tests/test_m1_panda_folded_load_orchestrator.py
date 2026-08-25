from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import STAGE_ORDER, stage_spec
from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_folded_load_curriculum.py"


def _load():
    spec = importlib.util.spec_from_file_location("folded_curriculum_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_accepted(root: Path, stage: str, previous: Path | None = None) -> Path:
    run_dir = root / stage
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_final.pt"
    checkpoint.write_bytes(f"accepted-{stage}".encode())
    manifest = run_dir / "run_manifest.json"
    parent_document = None
    if previous is not None:
        parent_document = json.loads(previous.read_text(encoding="utf-8"))
    manifest.write_text(
        json.dumps(
            {
                "stage": stage,
                "accepted": True,
                "parent_stage": None if previous is None else parent_document["stage"],
                "parent_manifest": None if previous is None else str(previous.resolve()),
                "parent_manifest_sha256": None if previous is None else sha256_file(previous),
                "parent_checkpoint": None if previous is None else parent_document["final_checkpoint"],
                "parent_checkpoint_sha256": None if previous is None else parent_document["final_checkpoint_sha256"],
                "final_checkpoint": str(checkpoint.resolve()),
                "final_checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _prefix(root: Path, last_stage: str) -> Path | None:
    previous = None
    for stage in STAGE_ORDER[: STAGE_ORDER.index(last_stage) + 1]:
        previous = _write_accepted(root, stage, previous)
    return previous


class FakeExecutor:
    def __init__(self, *, reject: str | None = None):
        self.reject = reject
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        run_dir = request.run_dir
        manifest = run_dir / "run_manifest.json"
        if request.stage == self.reject:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps({"stage": request.stage, "accepted": False}),
                encoding="utf-8",
            )
            return False
        previous = request.parent_manifest
        _write_accepted(run_dir.parent, request.stage, previous)
        return True


def test_rejected_stage_stops_and_keeps_previous_accepted_checkpoint(tmp_path):
    module = _load()
    previous = _prefix(tmp_path, "L1-C1")
    executor = FakeExecutor(reject="L1-C2")

    state = module.run_curriculum(
        tmp_path, start_stage="L1-C2", executor=executor
    )

    assert state.stopped_stage == "L1-C2"
    assert state.rollback_stage == "L1-C1"
    assert Path(state.rollback_checkpoint) == Path(
        json.loads(previous.read_text(encoding="utf-8"))["final_checkpoint"]
    )
    assert not (tmp_path / "L1-C3").exists()
    persisted = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert persisted["status"] == "stopped"
    assert persisted["rollback_stage"] == "L1-C1"


def test_parent_sha_must_match_previous_final_and_manifest(tmp_path):
    module = _load()
    previous = _prefix(tmp_path, "L1-C1")
    document = json.loads(previous.read_text(encoding="utf-8"))
    document["parent_checkpoint_sha256"] = "0" * 64
    previous.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        module.validate_lineage(tmp_path, through_stage="L1-C1")


def test_successful_execution_advances_in_order_and_passes_parent_manifest(tmp_path):
    module = _load()
    executor = FakeExecutor()

    state = module.run_curriculum(
        tmp_path, start_stage="L0-C0", executor=executor
    )

    assert state.status == "accepted"
    assert state.completed_stages == STAGE_ORDER
    assert [request.stage for request in executor.calls] == list(STAGE_ORDER)
    assert executor.calls[0].parent_manifest is None
    assert executor.calls[1].parent_manifest == tmp_path / "L0-C0/run_manifest.json"
    assert state.rollback_stage == "L2-D3"


def test_non_l0_start_requires_complete_accepted_prefix(tmp_path):
    module = _load()
    _write_accepted(tmp_path, "L0-C0")
    with pytest.raises((FileNotFoundError, ValueError), match="L1-C1"):
        module.run_curriculum(
            tmp_path, start_stage="L1-C2", executor=FakeExecutor()
        )


def test_cli_runs_train_then_three_fixed_evaluations_without_difficulty_fallback():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "m1_panda_folded_load_train.py" in source
    assert "m1_panda_folded_load_eval.py" in source
    assert "for seed in EVALUATION_SEEDS" in source
    assert "--parent_manifest" in source
    assert "--num_envs" in source and "--device" in source
    assert "subprocess.run" in source
    assert "lower_difficulty" not in source
    assert "continue_after_failure" not in source
