import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_wire_ppo_verify.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("m1_wire_ppo_verify_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_episode(*, stage: str = "approval") -> dict[str, object]:
    return {
        "stage": stage,
        "success": True,
        "terminated": True,
        "truncated": False,
        "safety_failure": False,
        "failure_reason": None,
        "contacts_left": 2,
        "contacts_right": 2,
        "contact_forces_left": [0.21, 0.25],
        "contact_forces_right": [0.3, 0.4],
        "lift_height": 0.05,
        "hold_time": 1.0,
        "slip": 0.02,
        "diagnostics": {
            "feasibility": {"safety": True},
            "safety_failure": False,
            "rejection_reason": None,
            "contact_failure_reason": None,
        },
        "action_target": {
            "reached": stage == "action",
            "wire_centerline_error": 0.03,
            "tangent_error": 0.25,
            "deformation_error": 0.02,
            "synchronization_error": 0.05,
            "force_balance_error": 0.01,
        },
    }


def test_cli_is_gpu0_single_lane_and_requires_checkpoint_stage_and_positive_episodes():
    module = _load_module()
    parser = module.build_arg_parser(include_app_launcher_args=False)
    args = parser.parse_args(
        ["--stage", "approval", "--checkpoint", "/tmp/best.pt", "--episodes", "3"]
    )

    assert args.stage == "approval"
    assert args.checkpoint == Path("/tmp/best.pt")
    assert args.seed == 42
    assert args.episodes == 3
    assert args.device == "cuda:0"
    assert args.num_envs == 1
    module.validate_cli_contract(args)

    args.episodes = 0
    with pytest.raises(ValueError, match="episodes"):
        module.validate_cli_contract(args)


def test_runtime_contract_pins_gpu0_and_every_cpu_pool_to_one_thread():
    module = _load_module()

    assert module.RUNTIME_ENVIRONMENT == {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def test_success_filter_replays_physical_and_action_target_gates():
    module = _load_module()
    accepted = _accepted_episode(stage="action")
    weak_contact = _accepted_episode(stage="action")
    weak_contact["contact_forces_left"] = [0.2, 0.25]
    missed_action = _accepted_episode(stage="action")
    missed_action["action_target"] = {
        **missed_action["action_target"],
        "wire_centerline_error": 0.031,
    }
    unsafe = _accepted_episode(stage="action")
    unsafe["safety_failure"] = True

    selected = module.successful_episode_rows(
        [accepted, weak_contact, missed_action, unsafe], stage="action"
    )

    assert selected == [accepted]
    assert module.successful_episode_rows(
        [_accepted_episode(stage="approval")], stage="approval"
    ) == [_accepted_episode(stage="approval")]


def test_action_checkpoint_without_replayable_approval_lineage_is_rejected(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"policy")
    episodes = tmp_path / "episodes.jsonl"
    row = {
        "episode_index": 0,
        "terminal_reason": "accepted",
        **_accepted_episode(stage="action"),
    }
    episodes.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "stage": "action",
        "action_dim": 15,
        "source": str(module.TRAIN_SCRIPT),
        "source_sha256": _sha(module.TRAIN_SCRIPT),
        "config": str(module.DEFAULT_CONFIG),
        "config_sha256": _sha(module.DEFAULT_CONFIG),
        "episodes_jsonl": episodes.name,
        "episodes_jsonl_sha256": _sha(episodes),
        "acceptance_counts": {"accepted": 1, "rejected": 0},
        "terminal_reason_counts": {"accepted": 1},
        "best_checkpoint": {"path": checkpoint.name, "sha256": _sha(checkpoint)},
    }
    module.write_sha_manifest(tmp_path / "training.manifest.json", manifest)

    with pytest.raises(ValueError, match="approval lineage"):
        module.validate_evaluation_checkpoint(
            checkpoint, stage="action", config_path=module.DEFAULT_CONFIG
        )


def test_checkpoint_manifest_and_episode_sha_tampering_is_rejected(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"policy")
    episodes = tmp_path / "episodes.jsonl"
    row = {
        "episode_index": 0,
        "terminal_reason": "accepted",
        **_accepted_episode(stage="approval"),
    }
    episodes.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    body = {
        "schema_version": 1,
        "status": "completed",
        "stage": "approval",
        "action_dim": 15,
        "source": str(module.TRAIN_SCRIPT),
        "source_sha256": _sha(module.TRAIN_SCRIPT),
        "config": str(module.DEFAULT_CONFIG),
        "config_sha256": _sha(module.DEFAULT_CONFIG),
        "episodes_jsonl": episodes.name,
        "episodes_jsonl_sha256": _sha(episodes),
        "acceptance_counts": {"accepted": 1, "rejected": 0},
        "terminal_reason_counts": {"accepted": 1},
        "best_checkpoint": {"path": checkpoint.name, "sha256": _sha(checkpoint)},
    }
    manifest_path = tmp_path / "training.manifest.json"
    module.write_sha_manifest(manifest_path, body)
    episodes.write_text(episodes.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="episode ledger SHA-256 mismatch"):
        module.validate_evaluation_checkpoint(
            checkpoint, stage="approval", config_path=module.DEFAULT_CONFIG
        )

    episodes.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    document = json.loads(manifest_path.read_text())
    document["seed"] = 43
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        module.validate_evaluation_checkpoint(
            checkpoint, stage="approval", config_path=module.DEFAULT_CONFIG
        )


def test_aggregate_manifest_pins_ledger_checkpoint_and_summary(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"policy")
    rows = [
        module.build_episode_record(
            _accepted_episode(stage="approval"),
            episode_index=0,
            stage="approval",
            checkpoint_sha256=_sha(checkpoint),
        ),
        module.build_episode_record(
            {
                **_accepted_episode(stage="approval"),
                "success": False,
                "failure_reason": "slip",
                "slip": 0.021,
            },
            episode_index=1,
            stage="approval",
            checkpoint_sha256=_sha(checkpoint),
        ),
    ]
    ledger = tmp_path / "episodes.jsonl"
    module.write_episode_ledger(ledger, rows)
    manifest_path = tmp_path / "aggregate.manifest.json"
    manifest = module.publish_aggregate_manifest(
        manifest_path,
        stage="approval",
        seed=42,
        requested_episodes=2,
        checkpoint=checkpoint,
        checkpoint_lineage={"training_manifest_sha256": "a" * 64},
        ledger=ledger,
        rows=rows,
    )

    assert manifest["status"] == "completed"
    assert manifest["accepted_episodes"] == 1
    assert manifest["episode_outcomes"] == {
        "bilateral_contact": 2,
        "lift": 2,
        "hold": 2,
        "action_target": 0,
    }
    assert manifest["checkpoint_sha256"] == _sha(checkpoint)
    assert manifest["episodes_jsonl_sha256"] == _sha(ledger)
    assert module.verify_aggregate_manifest(manifest_path) == manifest

    ledger.write_text(ledger.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="episode ledger SHA-256 mismatch"):
        module.verify_aggregate_manifest(manifest_path)


def test_aggregate_verifier_replays_counts_and_formal_gates(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"policy")
    row = module.build_episode_record(
        _accepted_episode(stage="approval"),
        episode_index=0,
        stage="approval",
        checkpoint_sha256=_sha(checkpoint),
    )
    ledger = tmp_path / "episodes.jsonl"
    module.write_episode_ledger(ledger, [row])
    manifest_path = tmp_path / "aggregate.manifest.json"
    manifest = module.publish_aggregate_manifest(
        manifest_path,
        stage="approval",
        seed=42,
        requested_episodes=1,
        checkpoint=checkpoint,
        checkpoint_lineage={"training_manifest_sha256": "a" * 64},
        ledger=ledger,
        rows=[row],
    )
    manifest["accepted_episodes"] = 0
    manifest["rejected_episodes"] = 1
    manifest["formal_passed"] = False
    module.write_sha_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="aggregate counts"):
        module.verify_aggregate_manifest(manifest_path)


def test_episode_record_reduces_non_json_diagnostics_to_task4_safety_evidence():
    torch = pytest.importorskip("torch")
    module = _load_module()
    info = _accepted_episode(stage="approval")
    info["diagnostics"]["unrelated_tensor"] = torch.ones(3)

    row = module.build_episode_record(
        info,
        episode_index=0,
        stage="approval",
        checkpoint_sha256="a" * 64,
    )

    assert "unrelated_tensor" not in row["diagnostics"]
    json.dumps(row, allow_nan=False)


def test_static_runtime_uses_normalized_inference_policy_and_never_trains():
    source = SCRIPT.read_text(encoding="utf-8")

    for token in (
        "AppLauncher",
        "OnPolicyRunner",
        "M1WirePpoEnv",
        "M1DualPandaO6BimanualWrapper",
        "runner.load(",
        "load_optimizer=False",
        "keep_std=True",
        "runner.get_inference_policy(device=args.device)",
        "torch.inference_mode()",
    ):
        assert token in source
    assert "runner.learn(" not in source
    assert "runner_internal.jsonl" not in source
