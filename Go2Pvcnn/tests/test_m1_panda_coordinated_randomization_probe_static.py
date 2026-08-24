from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_coordinated_randomization_probe.py"


def test_probe_launches_isaac_before_runtime_imports_and_enables_training_dr():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("AppLauncher(args).app") < source.index("import gymnasium as gym")
    assert "configure_coordinated_training_domain_randomization(cfg, True)" in source
    assert "training_randomization=True" in source
    assert 'TASK_ID = "Isaac-M1-Panda-Coordinated-v0"' in source


def test_probe_checks_seed_diversity_selective_reset_and_physical_response():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "same_seed_reset_match",
        "different_env_reset_diversity",
        "selected_reset_isolation",
        "reset_bounds_passed",
        "controlled_velocity_randomized",
        '"panda_hand"',
        "applied_hand_wrench_nonzero",
        "mount_wrench_response_nonzero",
        "mount_wrench_b",
        "non_finite_count",
        "reset_count",
        "base_contact_count",
        "bad_orientation_count",
    ):
        assert token in source
    assert source.count("torch.manual_seed(args.seed)") >= 2
    assert "_reset_idx(selected_ids)" in source


def test_probe_atomically_writes_one_hard_gate_json_summary():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert "NamedTemporaryFile" in source
    assert "os.replace" in source
    assert "allow_nan=False" in source
    assert 'summary["hard_gates_passed"]' in source
    assert 'return 0 if summary["hard_gates_passed"] else 2' in source


def test_probe_treats_only_positive_finite_velocity_limits_as_metadata():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "usable_velocity_limit" in source
    assert "torch.isfinite(controlled_velocity_limits)" in source
