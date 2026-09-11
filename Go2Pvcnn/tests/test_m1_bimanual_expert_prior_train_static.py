from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_train_fingertip_expert.py"


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
