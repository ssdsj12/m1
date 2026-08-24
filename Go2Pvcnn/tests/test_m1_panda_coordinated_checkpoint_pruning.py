from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prune_m1_panda_coordinated_checkpoints.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("checkpoint_pruner_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pruner():
    return _load_module()


@pytest.fixture
def fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "coordinated_teacher_long_v4_64x5000_20260823"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "run_name": run.name}) + "\n",
        encoding="utf-8",
    )
    for name in (
        "model_0.pt",
        "model_3500.pt",
        "model_3600.pt",
        "model_4999.pt",
        "model_best.pt",
        "model_final.pt",
    ):
        (run / name).write_bytes(f"fixture:{name}".encode())
    return run


def test_plan_keeps_3500_and_selects_only_numeric_models_above_it(
    pruner, fake_run: Path
) -> None:
    plan = pruner.build_pruning_plan(
        fake_run, keep_through=3500, expected_run_dir=fake_run
    )
    assert [item.path.name for item in plan.delete] == [
        "model_3600.pt",
        "model_4999.pt",
    ]
    assert (fake_run / "model_3500.pt").is_file()


def test_plan_rejects_symlinks_and_paths_outside_exact_run(
    pruner, fake_run: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (fake_run / "model_3600.pt").unlink()
    (fake_run / "model_3600.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        pruner.build_pruning_plan(
            fake_run, keep_through=3500, expected_run_dir=fake_run
        )
    with pytest.raises(ValueError, match="exact long-v4"):
        pruner.build_pruning_plan(
            tmp_path, keep_through=3500, expected_run_dir=fake_run
        )


def test_dry_run_does_not_unlink_and_apply_writes_hashes(
    pruner, fake_run: Path
) -> None:
    plan = pruner.build_pruning_plan(
        fake_run, keep_through=3500, expected_run_dir=fake_run
    )
    result = pruner.execute_pruning(plan, apply=False)
    assert result["status"] == "dry_run"
    assert (fake_run / "model_3600.pt").is_file()

    pruner.execute_pruning(plan, apply=True)
    audit = json.loads((fake_run / "checkpoint_pruning.json").read_text())
    assert audit["status"] == "completed"
    assert [item["name"] for item in audit["deleted"]] == [
        "model_3600.pt",
        "model_4999.pt",
    ]
    assert all(len(item["sha256"]) == 64 for item in audit["deleted"])
    assert len(audit["original_manifest_sha256"]) == 64
    assert len(audit["kept_model_3500_sha256"]) == 64
    assert not (fake_run / "model_3600.pt").exists()


def test_postcondition_rejects_remaining_checkpoint_above_limit(
    pruner, fake_run: Path
) -> None:
    with pytest.raises(RuntimeError, match="remaining"):
        pruner.verify_pruning_postcondition(
            fake_run, 3500, ignored_names={"model_3600.pt"}
        )
