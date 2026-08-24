from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from agent import get_m1_panda_coordinated_train_cfg


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_coordinated_train.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_coordinated_train_manifest_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manifest_contract_freezes_ppo_dr_and_guard(tmp_path: Path) -> None:
    module = _load_script()
    asset = tmp_path / "m1_panda.usd"
    checkpoint = tmp_path / "model_10402.pt"
    asset.write_bytes(b"asset")
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        run_name="fresh",
        device="cuda:0",
        num_envs=64,
        seed=42,
        max_iterations=600,
    )

    manifest = module.build_manifest_contract(
        args, asset, checkpoint, get_m1_panda_coordinated_train_cfg()
    )

    assert manifest["schema_version"] == 2
    assert manifest["ppo"]["num_steps_per_env"] == 256
    assert manifest["ppo"]["gamma"] == 0.9995
    assert manifest["ppo"]["lambda"] == 0.995
    assert manifest["ppo"]["schedule"] == "adaptive"
    assert manifest["ppo"]["learning_rate_bounds"] == [1.0e-6, 3.0e-4]
    assert manifest["ppo"]["action_std_bounds"] == [0.005, 0.05]
    assert manifest["domain_randomization"]["target_body"] == "panda_hand"
    assert manifest["domain_randomization"]["force_limit_n"] == 20.0
    assert manifest["domain_randomization"]["root_xy_m"] == [-0.02, 0.02]
    assert manifest["domain_randomization"]["friction"] == [0.8, 1.2]
    assert manifest["guard"]["minimum_completed_episodes"] == 100
    assert manifest["guard"]["eligible_timeout_rate"] == 0.90
    assert manifest["guard"]["max_iterations"] == 600
    assert manifest["fresh_policy"] is True
    assert manifest["initialization_lineage_only"] is True
    assert manifest["asset_sha256"] == module.sha256_file(asset)
    assert manifest["init_a1_checkpoint_sha256"] == module.sha256_file(checkpoint)


def test_manifest_builder_rejects_non_stable_config(tmp_path: Path) -> None:
    module = _load_script()
    asset = tmp_path / "asset.usd"
    checkpoint = tmp_path / "model.pt"
    asset.write_bytes(b"asset")
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        run_name="fresh", device="cuda:0", num_envs=64, seed=42, max_iterations=600
    )
    train_cfg = get_m1_panda_coordinated_train_cfg()
    train_cfg["algorithm"]["schedule"] = "fixed"

    try:
        module.build_manifest_contract(args, asset, checkpoint, train_cfg)
    except ValueError as error:
        assert "adaptive" in str(error)
    else:
        raise AssertionError("fixed schedule must be rejected")
