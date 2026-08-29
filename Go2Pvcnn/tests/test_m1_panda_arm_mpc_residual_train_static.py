from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agent import get_m1_panda_arm_mpc_residual_train_cfg


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_arm_mpc_residual_train.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_arm_mpc_residual_train_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_residual_ppo_config_freezes_200_hz_stability_contract():
    cfg = get_m1_panda_arm_mpc_residual_train_cfg()

    assert cfg["num_steps_per_env"] == 256
    assert cfg["max_iterations"] == 3000
    assert cfg["algorithm"]["schedule"] == "adaptive"
    assert cfg["algorithm"]["desired_kl"] == pytest.approx(0.01)
    assert cfg["algorithm"]["num_learning_epochs"] == 2
    assert cfg["algorithm"]["learning_rate"] == pytest.approx(1.0e-5)
    assert cfg["algorithm"]["min_learning_rate"] == pytest.approx(1.0e-6)
    assert cfg["algorithm"]["max_learning_rate"] == pytest.approx(1.0e-4)
    assert cfg["algorithm"]["kl_abort_threshold"] == pytest.approx(0.015)
    assert cfg["algorithm"]["max_grad_norm"] == pytest.approx(0.5)
    assert cfg["algorithm"]["clip_min_std"] == pytest.approx(0.005)
    assert cfg["algorithm"]["clip_max_std"] == pytest.approx(0.02)
    assert cfg["policy"]["class_name"].endswith(".ResidualActorCritic")
    assert cfg["policy"]["init_noise_std"] == pytest.approx(0.005)


def test_train_cli_defaults_to_bounded_short_gate():
    module = _load_script()
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args([])

    assert args.stage == "short"
    assert args.device == "cuda:0"
    assert args.num_envs == 8
    assert args.max_iterations is None
    assert args.seed == 42
    assert args.headless is True
    assert module.resolve_max_iterations("zero", None) == 10
    assert module.resolve_max_iterations("short", None) == 100
    assert module.resolve_max_iterations("long", None) == 3000
    with pytest.raises(ValueError, match="short.*100"):
        module.resolve_max_iterations("short", 101)
    with pytest.raises(ValueError, match="short.*100"):
        module.resolve_max_iterations("short", 99)


def test_runner_empty_stop_reason_is_safe_requested_completion():
    module = _load_script()

    assert module.is_safe_completion("") is True
    assert module.is_safe_completion(None) is True
    assert module.is_safe_completion("hard_failure") is False


def test_long_stage_requires_accepted_matching_promotion_manifest(tmp_path):
    module = _load_script()
    asset = tmp_path / "robot.usd"
    config = tmp_path / "train_cfg.py"
    reward = tmp_path / "reward.py"
    checkpoint = tmp_path / "model_best.pt"
    asset.write_bytes(b"asset")
    config.write_bytes(b"config")
    reward.write_bytes(b"reward")
    checkpoint.write_bytes(b"checkpoint")
    short_manifest = tmp_path / "run_manifest.json"
    promotion_manifest = tmp_path / "promotion_manifest.json"

    with pytest.raises(ValueError, match="requires.*manifest"):
        module.validate_promotion_manifest(
            None, asset_path=asset, config_path=config, reward_path=reward
        )

    short_payload = {
        "schema_version": 1,
        "stage": "short",
        "accepted": False,
        "status": "safe_complete",
        "promotion_required": True,
        "asset_sha256": module.sha256_file(asset),
        "config_sha256": module.sha256_file(config),
        "reward_sha256": module.sha256_file(reward),
        "candidate_checkpoints": [
            {
                "completed_updates": 25,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": module.sha256_file(checkpoint),
            }
        ],
    }
    short_manifest.write_text(json.dumps(short_payload), encoding="utf-8")
    promotion_payload = {
        "schema_version": 1,
        "accepted": False,
        "short_manifest": str(short_manifest),
        "short_manifest_sha256": module.sha256_file(short_manifest),
        "asset_sha256": module.sha256_file(asset),
        "config_sha256": module.sha256_file(config),
        "reward_sha256": module.sha256_file(reward),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": module.sha256_file(checkpoint),
    }
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted=true"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
        )

    promotion_payload["accepted"] = True
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    lineage = module.validate_promotion_manifest(
        promotion_manifest,
        asset_path=asset,
        config_path=config,
        reward_path=reward,
    )
    assert lineage.checkpoint == checkpoint.resolve()
    assert lineage.short_manifest == short_manifest.resolve()

    promotion_payload["reward_sha256"] = "0" * 64
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reward SHA"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
        )


def test_train_source_uses_fresh_8d_policy_and_offline_promotion():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Isaac-M1-Panda-ArmMpc-Residual-v0" in source
    assert "M1PandaArmMpcResidualEnvWrapper" in source
    assert "force_zero_residual=args.stage == \"zero\"" in source
    assert "OnPolicyRunner" in source
    assert "ResidualTrainingSafetyGuard" in source
    assert "iteration_callback=" in source
    assert "candidate_u000.pt" in source
    assert '"promotion_required": True' in source
    assert "os.replace" in source
    assert "legacy_23d_checkpoint_loaded" in source
    assert "validate_promotion_manifest(" in source
    assert "load_optimizer=False" in source
    assert "legacy" in source.lower()


def test_candidate_names_use_completed_update_counts(tmp_path):
    module = _load_script()

    class Runner:
        payload = b"rollout-policy"

        def save(self, path):
            Path(path).write_bytes(self.payload)

    metrics = {
        "hard_failure_count": 0.0,
        "mpc_feasible_rate": 1.0,
        "qp_feasible_rate": 1.0,
        "four_contact_rate": 1.0,
        "roll_pitch_rms": 0.01,
        "base_height_rms": 0.01,
        "ee_position_error": 0.01,
        "ee_orientation_error": 0.04,
        "wrench_error": 0.1,
        "slip": 0.0,
        "intervention_ratio": 0.0,
        **{f"saturation_fraction_{index}": 0.0 for index in range(8)},
    }
    summary = SimpleNamespace(
        iteration=0,
        environment_metrics=tuple(metrics.items()),
        learning_rate=1.0e-5,
        kl_mean=0.01,
        kl_max=0.01,
        grad_norm=0.1,
        active_action_std_min=0.005,
        active_action_std_max=0.005,
    )
    runner = Runner()
    controller = module.ResidualTrainingSafetyController(tmp_path)
    controller.prime(runner)
    runner.payload = b"post-update-policy"
    summary.iteration = 24

    assert controller.on_iteration(runner, summary) is None

    assert (tmp_path / "candidate_u000.pt").read_bytes() == b"rollout-policy"
    assert (tmp_path / "candidate_u025.pt").read_bytes() == b"post-update-policy"
    assert not (tmp_path / "model_best.pt").exists()


def test_safety_controller_stops_on_hard_failure(tmp_path):
    module = _load_script()

    class Runner:
        def save(self, path):
            Path(path).write_bytes(b"policy")

    metrics = {
        "hard_failure_count": 1.0,
        "mpc_feasible_rate": 1.0,
        "qp_feasible_rate": 1.0,
        "four_contact_rate": 1.0,
        "roll_pitch_rms": 0.01,
        "base_height_rms": 0.01,
        "ee_position_error": 0.01,
        "ee_orientation_error": 0.04,
        "wrench_error": 0.1,
        "slip": 0.0,
        "intervention_ratio": 0.0,
        **{f"saturation_fraction_{index}": 0.0 for index in range(8)},
    }
    summary = SimpleNamespace(
        iteration=0,
        environment_metrics=tuple(metrics.items()),
        learning_rate=1.0e-5,
        kl_mean=0.01,
        kl_max=0.01,
        grad_norm=0.1,
        active_action_std_min=0.005,
        active_action_std_max=0.005,
    )

    controller = module.ResidualTrainingSafetyController(tmp_path)

    assert controller.on_iteration(Runner(), summary) == "hard_failure"
