from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"
VERIFY = ROOT / "scripts/m1_dual_panda_o6_verify.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("m1_dual_panda_o6_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_layer_verifier_freezes_test_groups_gpu0_and_exact_formal_matrix():
    source = VERIFY.read_text(encoding="utf-8")

    for token in (
        'choices=("cpu", "qp", "smoke", "formal", "all")',
        "CPU_TESTS",
        "PURE_QP_TESTS",
        '"--device"',
        '"cuda:0"',
        '"--seeds"',
        '"42"',
        '"43"',
        '"44"',
        '"--trials-per-seed"',
        '"10"',
        '"--jsonl"',
        '"--manifest"',
        '"formal_aggregate.manifest.json"',
    ):
        assert token in source
    assert ".learn(" not in source
    assert "scripts/train.py" not in source


def test_probe_exposes_formal_jsonl_and_sha_pinned_manifest():
    source = PROBE.read_text(encoding="utf-8")
    for token in (
        "--jsonl",
        "--manifest",
        "_atomic_jsonl",
        "_artifact_manifest",
        "trials_jsonl",
        "aggregate_report",
    ):
        assert token in source


def test_formal_jsonl_and_manifest_pin_exact_artifact_bytes(tmp_path):
    probe = _load_probe_module()
    rows = [{"seed": 42, "trial_index": 0}, {"seed": 43, "trial_index": 0}]
    report = {
        "passed": True,
        "acceptance_mode": "formal",
        "aggregate": {"accepted": True, "observed_trial_count": 2},
        "metadata": {
            "asset_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "git_ref": "c" * 40,
            "isaac_version": "5.1.0.0",
            "command": "probe --seeds 42 43",
        },
    }
    report_path = tmp_path / "report.json"
    jsonl_path = tmp_path / "trials.jsonl"
    manifest_path = tmp_path / "aggregate.manifest.json"
    probe._atomic_json(report_path, report)
    probe._atomic_jsonl(jsonl_path, rows)
    manifest = probe._artifact_manifest(
        report, report_path=report_path, jsonl_path=jsonl_path
    )
    probe._atomic_json(manifest_path, manifest)

    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == rows
    assert manifest["status"] == "passed"
    assert manifest["pins"]["trials_jsonl"]["sha256"] == probe._sha256(jsonl_path)
    assert manifest["pins"]["aggregate_report"]["sha256"] == probe._sha256(report_path)
