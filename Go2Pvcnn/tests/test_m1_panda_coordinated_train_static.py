from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_coordinated_train.py"


def test_coordinated_train_entrypoint_declares_joint_action_and_init_contract():
    source = SCRIPT.read_text()
    assert "Isaac-M1-Panda-Coordinated-v0" in source
    assert "OnPolicyRunner" in source
    assert "--init-a1-checkpoint" in source
    assert "--max_iterations" in source
    assert "action_dim" in source
    assert "23" in source


def test_train_writes_atomic_manifest_and_freezes_fresh_run_boundary():
    source = SCRIPT.read_text()
    assert "run_manifest.json" in source
    assert "NamedTemporaryFile" in source
    assert "os.replace" in source
    assert "refusing to reuse non-empty run directory" in source
    for field in (
        '"status"',
        '"observation_dim"',
        '"action_dim"',
        '"asset_sha256"',
        '"init_a1_checkpoint_sha256"',
        '"seed"',
        '"run_name"',
    ):
        assert field in source


def test_train_checks_103_by_23_runtime_contract_and_periodic_save():
    source = SCRIPT.read_text()
    assert "COORDINATED_POLICY_OBSERVATION_DIM" in source
    assert "observation_dim != 103" in source
    assert "wrapper.num_actions != 23" in source
    assert "get_m1_panda_coordinated_train_cfg" in source
    assert 'train_cfg["algorithm"]["schedule"] = "fixed"' not in source
    assert "torch.nn.init.zeros_(output_layer.weight)" in source
    assert "torch.nn.init.zeros_(output_layer.bias)" in source
    assert "requires_grad_(False)" not in source
    assert "coordinated policy std must remain trainable" in source
    assert '"zero_action_actor_initialization": True' in source
    assert '"fresh_policy": True' in source


def test_train_enables_exact_dr_and_automatic_checkpoint_rollback():
    source = SCRIPT.read_text()
    assert "configure_coordinated_training_domain_randomization(cfg, True)" in source
    assert (
        "M1PandaCoordinatedEnvWrapper(\n"
        "            env, training_randomization=True, seed=args.seed\n"
        "        )"
    ) in source
    assert "AtomicCheckpointController" in source
    assert "TrainingGuard(max_iterations=args.max_iterations)" in source
    assert "iteration_callback=" in source
    assert "controller.on_iteration" in source
    assert "controller.finalize(runner, learn_result.stop_reason)" in source
    assert 'default=600' in source
