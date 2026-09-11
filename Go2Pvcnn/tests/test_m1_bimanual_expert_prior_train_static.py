from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json
import importlib.util

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_train_fingertip_expert.py"


def _trainer_module():
    spec = importlib.util.spec_from_file_location("task6_trainer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_script_does_not_import_task_or_object_features():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("description", "primitive", "object_id", "palm_target", "box_pose"):
        assert forbidden not in source


def test_training_script_has_hash_verification_and_nonproduction_smoke_gate():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_aggregate_manifest" in source
    assert "synthetic_smoke" in source
    assert "production_deployable" in source
    assert "0.01" in source


def test_resume_directory_retains_selected_member_when_epochs_are_complete(tmp_path: Path):
    initial = tmp_path / "initial"
    resumed = tmp_path / "resumed"
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])}
    command = [sys.executable, str(SCRIPT), "--synthetic-smoke", "--epochs", "1"]

    subprocess.run([*command, "--output-dir", str(initial)], check=True, env=environment)
    aggregate = json.loads((initial / "synthetic-shards" / "aggregate_manifest.json").read_text())
    assert aggregate["verified_inputs"] == {"nonproduction_synthetic": True}
    subprocess.run(
        [*command, "--output-dir", str(resumed), "--resume-checkpoint", str(initial)],
        check=True,
        env=environment,
    )

    assert (resumed / "checkpoints" / "member-00-best.pt").is_file()
    assert "nonproduction_synthetic" in SCRIPT.read_text(encoding="utf-8")


def test_group_overlap_is_rejected_before_shards_are_loaded():
    document = {
        "split_groups": {
            "train": ["sequence-a"],
            "validation": ["sequence-a"],
            "test": ["sequence-b"],
        }
    }

    with pytest.raises(ValueError, match="overlap"):
        _trainer_module()._validate_group_assignments(document)


@pytest.mark.parametrize("field, value", [("seed", 9999), ("checkpoint_sha256", "0" * 64)])
def test_resume_rejects_manifest_member_tampering_before_checkpoint_load(
    tmp_path: Path, field: str, value: object
):
    initial = tmp_path / "initial"
    environment = {**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])}
    subprocess.run(
        [sys.executable, str(SCRIPT), "--synthetic-smoke", "--epochs", "1", "--output-dir", str(initial)],
        check=True,
        env=environment,
    )
    manifest_path = initial / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][1][field] = value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256"):
        _trainer_module()._load_resume(
            initial,
            member_index=0,
            seed=1701,
            hidden=(512, 512, 512),
            aggregate_sha=manifest["dataset_aggregate_sha256"],
        )
