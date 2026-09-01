from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

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
    assert cfg["empirical_normalization"] is True
    assert cfg["algorithm"]["schedule"] == "adaptive"
    assert cfg["algorithm"]["desired_kl"] == pytest.approx(0.01)
    assert cfg["algorithm"]["num_learning_epochs"] == 2
    assert cfg["algorithm"]["learning_rate"] == pytest.approx(1.0e-5)
    assert cfg["algorithm"]["min_learning_rate"] == pytest.approx(1.0e-6)
    assert cfg["algorithm"]["max_learning_rate"] == pytest.approx(1.0e-5)
    assert (
        cfg["algorithm"]["max_learning_rate"]
        == cfg["algorithm"]["learning_rate"]
    )
    assert cfg["algorithm"]["kl_abort_threshold"] == pytest.approx(0.015)
    assert cfg["algorithm"]["max_grad_norm"] == pytest.approx(0.5)
    assert cfg["algorithm"]["clip_min_std"] == pytest.approx(0.005)
    assert cfg["algorithm"]["clip_max_std"] == pytest.approx(0.02)
    assert cfg["policy"]["class_name"].endswith(".ResidualActorCritic")
    assert cfg["policy"]["init_noise_std"] == pytest.approx(0.01)


def test_train_cli_defaults_to_bounded_short_gate():
    module = _load_script()
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args([])

    assert args.stage == "short"
    assert args.device == "cuda:0"
    assert args.num_envs == 8
    assert args.max_iterations is None
    assert args.pilot_manifest is None
    assert args.short_manifest is None
    assert args.seed == 42
    assert args.headless is True
    assert module.resolve_max_iterations("zero", None) == 10
    assert module.resolve_max_iterations("pilot", None) == 10
    assert module.resolve_max_iterations("short", None) == 100
    assert module.resolve_max_iterations("bridge", None) == 200
    assert module.resolve_max_iterations("bridge", 200) == 200
    assert module.resolve_max_iterations("long", None) == 3000
    with pytest.raises(ValueError, match="short.*100"):
        module.resolve_max_iterations("short", 101)
    with pytest.raises(ValueError, match="short.*100"):
        module.resolve_max_iterations("short", 99)
    with pytest.raises(ValueError, match="pilot.*10"):
        module.resolve_max_iterations("pilot", 9)
    with pytest.raises(ValueError, match="bridge.*exactly 200"):
        module.resolve_max_iterations("bridge", 199)


def test_bridge_cli_accepts_only_its_short_manifest():
    module = _load_script()
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args(
        ["--stage", "bridge", "--short_manifest", "short.json"]
    )

    assert args.stage == "bridge"
    assert args.short_manifest == Path("short.json")


def test_pilot_controller_records_ten_updates_without_candidates(tmp_path):
    module = _load_script()
    controller = module.PilotTrainingController()
    metrics = {
        "hard_failure_count": 0.0,
        "mpc_feasible_rate": 1.0,
        "qp_feasible_rate": 1.0,
        "four_contact_rate": 1.0,
        **{f"saturation_fraction_{index}": 0.0 for index in range(8)},
    }
    for iteration in range(10):
        summary = SimpleNamespace(
            iteration=iteration,
            learning_rate=1.0e-5,
            value_loss=10.0,
            kl_mean=0.005,
            kl_max=0.01,
            kl_aborted=False,
            completed_mini_batches=8,
            grad_norm=0.5,
            active_action_std_min=0.01,
            active_action_std_max=0.01,
            completed_rewards=(),
            environment_metrics=tuple(metrics.items()),
        )
        assert controller.on_iteration(None, summary) is None

    assert controller.decision().accepted is True
    assert len(controller.records) == 10
    assert list(tmp_path.glob("candidate_u*.pt")) == []


def test_short_requires_accepted_hash_matching_pilot_manifest(tmp_path):
    module = _load_script()
    asset = tmp_path / "robot.usd"
    config = tmp_path / "train_cfg.py"
    reward = tmp_path / "reward.py"
    runtime = tmp_path / "runtime.py"
    for path in (asset, config, reward, runtime):
        path.write_text(path.name, encoding="utf-8")
    paths = module.ResidualSourcePaths(asset, config, reward, runtime)
    pilot = tmp_path / "pilot_manifest.json"
    payload = {
        "schema_version": 2,
        "stage": "pilot",
        "status": "safe_complete",
        "accepted": False,
        "promotion_required": False,
        "pilot_accepted": True,
        "completed_iterations": 10,
        "optimizer_summaries": [{"update": update} for update in range(1, 11)],
        "pilot_decision": {"accepted": True},
        **module.source_lineage(paths),
    }
    pilot.write_text(json.dumps(payload), encoding="utf-8")

    lineage = module.validate_pilot_manifest(pilot, paths)
    assert lineage.manifest == pilot.resolve()

    payload["pilot_schema_sha256"] = "0" * 64
    pilot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot schema SHA"):
        module.validate_pilot_manifest(pilot, paths)

    payload.update(module.source_lineage(paths))
    payload["pilot_accepted"] = False
    pilot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot_accepted=true"):
        module.validate_pilot_manifest(pilot, paths)


def test_runner_empty_stop_reason_is_safe_requested_completion():
    module = _load_script()

    assert module.is_safe_completion("") is True
    assert module.is_safe_completion(None) is True
    assert module.is_safe_completion("hard_failure") is False


def test_long_stage_requires_accepted_matching_promotion_manifest(tmp_path):
    module = _load_script()
    asset, config, reward, runtime = (
        tmp_path / name
        for name in ("robot.usd", "train_cfg.py", "reward.py", "runtime.py")
    )
    for path in (asset, config, reward, runtime):
        path.write_bytes(path.name.encode())
    source = module.source_lineage(
        module.ResidualSourcePaths(asset, config, reward, runtime)
    )
    pilot_manifest = tmp_path / "pilot_manifest.json"
    pilot_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "pilot",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": False,
                "pilot_accepted": True,
                "requested_iterations": 10,
                "completed_iterations": 10,
                "optimizer_summaries": [
                    {"update": update} for update in range(1, 11)
                ],
                "pilot_decision": {"accepted": True},
                **source,
            }
        ),
        encoding="utf-8",
    )
    short_candidates = []
    for updates in (0, 25, 50, 75, 100):
        candidate_path = tmp_path / f"short_u{updates:03d}.pt"
        torch.save(
            {
                "model_state_dict": {"weight": torch.tensor([float(updates)])},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(updates)}}},
                "obs_norm_state_dict": {"_mean": torch.zeros(1, 103)},
                "critic_obs_norm_state_dict": {"_mean": torch.zeros(1, 103)},
                "iter": updates - 1,
            },
            candidate_path,
        )
        short_candidates.append(
            {
                "completed_updates": updates,
                "checkpoint": str(candidate_path),
                "checkpoint_sha256": module.sha256_file(candidate_path),
            }
        )
    short_manifest = tmp_path / "short_manifest.json"
    short_payload = {
        "schema_version": 2,
        "stage": "short",
        "accepted": False,
        "status": "safe_complete",
        "promotion_required": True,
        "requested_iterations": 100,
        "completed_iterations": 100,
        "pilot_manifest": str(pilot_manifest),
        "pilot_manifest_sha256": module.sha256_file(pilot_manifest),
        **source,
        "candidate_checkpoints": short_candidates,
    }
    short_manifest.write_text(json.dumps(short_payload), encoding="utf-8")

    bridge_candidates = []
    for updates in (100, 150, 200, 250, 300):
        candidate_path = tmp_path / f"bridge_u{updates:03d}.pt"
        count = 204800 + (updates - 100) * 2048
        torch.save(
            {
                "model_state_dict": {"weight": torch.tensor([float(updates)])},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(updates)}}},
                "obs_norm_state_dict": {
                    "_mean": torch.zeros(1, 103),
                    "_count": torch.tensor(count, dtype=torch.int64),
                },
                "critic_obs_norm_state_dict": {
                    "_mean": torch.zeros(1, 103),
                    "_count": torch.tensor(count, dtype=torch.int64),
                },
                "iter": updates - 1,
            },
            candidate_path,
        )
        bridge_candidates.append(
            {
                "completed_updates": updates,
                "checkpoint": str(candidate_path),
                "checkpoint_sha256": module.sha256_file(candidate_path),
            }
        )
    bridge_manifest = tmp_path / "bridge_manifest.json"
    bridge_payload = {
        "schema_version": 3,
        "stage": "bridge",
        "status": "safe_complete",
        "accepted": False,
        "promotion_required": True,
        "requested_iterations": 200,
        "completed_iterations": 200,
        "starting_updates": 100,
        "requested_additional_updates": 200,
        "target_total_updates": 300,
        "completed_total_updates": 300,
        "starting_normalizer_count": 204800,
        "short_manifest": str(short_manifest),
        "short_manifest_sha256": module.sha256_file(short_manifest),
        "pilot_manifest": str(pilot_manifest),
        "pilot_manifest_sha256": module.sha256_file(pilot_manifest),
        "parent_checkpoint": short_candidates[-1]["checkpoint"],
        "parent_checkpoint_sha256": short_candidates[-1]["checkpoint_sha256"],
        "migrated_checkpoint": bridge_candidates[0]["checkpoint"],
        "migrated_checkpoint_sha256": bridge_candidates[0]["checkpoint_sha256"],
        **source,
        "candidate_checkpoints": bridge_candidates,
    }
    bridge_manifest.write_text(json.dumps(bridge_payload), encoding="utf-8")
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(Path(bridge_candidates[1]["checkpoint"]).read_bytes())
    promotion_manifest = tmp_path / "promotion_manifest.json"
    promotion_payload = {
        "schema_version": 3,
        "status": "accepted",
        "accepted": True,
        "bridge_manifest": str(bridge_manifest),
        "bridge_manifest_sha256": module.sha256_file(bridge_manifest),
        "short_manifest": str(short_manifest),
        "short_manifest_sha256": module.sha256_file(short_manifest),
        "pilot_manifest": str(pilot_manifest),
        "pilot_manifest_sha256": module.sha256_file(pilot_manifest),
        **source,
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": module.sha256_file(checkpoint),
    }
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires.*manifest"):
        module.validate_promotion_manifest(
            None,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
        )
    lineage = module.validate_promotion_manifest(
        promotion_manifest,
        asset_path=asset,
        config_path=config,
        reward_path=reward,
        runtime_path=runtime,
    )
    assert lineage.checkpoint == checkpoint.resolve()
    assert lineage.bridge_manifest == bridge_manifest.resolve()
    assert lineage.short_manifest == short_manifest.resolve()

    promotion_payload["accepted"] = False
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted=true"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
        )

    promotion_payload["accepted"] = True
    promotion_payload["bridge_manifest_sha256"] = "0" * 64
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bridge manifest SHA"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
        )

    promotion_payload["bridge_manifest_sha256"] = module.sha256_file(bridge_manifest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["obs_norm_state_dict"].pop("_count")
    torch.save(payload, checkpoint)
    promotion_payload["best_checkpoint_sha256"] = module.sha256_file(checkpoint)
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
        )

    checkpoint.write_bytes(Path(bridge_candidates[1]["checkpoint"]).read_bytes())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["optimizer_state_dict"] = {}
    torch.save(payload, checkpoint)
    promotion_payload["best_checkpoint_sha256"] = module.sha256_file(checkpoint)
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
        )

    unrecorded = tmp_path / "unrecorded.pt"
    unrecorded.write_bytes(Path(bridge_candidates[1]["checkpoint"]).read_bytes())
    payload = torch.load(unrecorded, map_location="cpu", weights_only=False)
    payload["model_state_dict"]["weight"] += 1.0
    torch.save(payload, unrecorded)
    promotion_payload["best_checkpoint"] = str(unrecorded)
    promotion_payload["best_checkpoint_sha256"] = module.sha256_file(unrecorded)
    promotion_manifest.write_text(json.dumps(promotion_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded bridge candidate"):
        module.validate_promotion_manifest(
            promotion_manifest,
            asset_path=asset,
            config_path=config,
            reward_path=reward,
            runtime_path=runtime,
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
    assert '"promotion_required": promotion_required' in source
    assert 'args.stage == "bridge" and safe_complete' in source
    assert "os.replace" in source
    assert "legacy_23d_checkpoint_loaded" in source
    assert "validate_promotion_manifest(" in source
    assert "source_lineage(" in source
    assert "validate_source_lineage(" in source
    assert "load_optimizer=True" in source
    assert "runner did not restore promoted normalizer counts" in source
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


def test_bridge_controller_publishes_only_total_update_candidates(tmp_path):
    module = _load_script()

    class Runner:
        saves = 0

        def save(self, path):
            self.saves += 1
            Path(path).write_bytes(f"policy-{self.saves}".encode())

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
    initial = tmp_path / "candidate_u100.pt"
    initial.write_bytes(b"migrated-parent")
    controller = module.ResidualTrainingSafetyController(
        tmp_path,
        starting_updates=100,
        candidate_updates=(100, 150, 200, 250, 300),
    )
    controller.register_initial_candidate(100, initial)
    runner = Runner()
    for total_update in (150, 200, 250, 300):
        summary = SimpleNamespace(
            iteration=total_update - 1,
            environment_metrics=tuple(metrics.items()),
            learning_rate=1.0e-5,
            kl_mean=0.01,
            kl_max=0.01,
            grad_norm=0.1,
            active_action_std_min=0.005,
            active_action_std_max=0.005,
        )
        assert controller.on_iteration(runner, summary) is None

    assert tuple(controller.candidate_checkpoints) == (100, 150, 200, 250, 300)
    assert sorted(path.name for path in tmp_path.glob("candidate_u*.pt")) == [
        "candidate_u100.pt",
        "candidate_u150.pt",
        "candidate_u200.pt",
        "candidate_u250.pt",
        "candidate_u300.pt",
    ]
    assert not (tmp_path / "candidate_u101.pt").exists()
    assert not (tmp_path / "candidate_u299.pt").exists()


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
