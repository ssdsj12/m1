from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts/build_m1_single_panda_o6_asset.py"
VERIFIER = PROJECT_ROOT / "scripts/verify_m1_single_panda_o6_asset.py"


def _builder_source() -> str:
    assert BUILDER.is_file()
    return BUILDER.read_text(encoding="utf-8")


def test_builder_freezes_single_arm_topology_and_counts():
    source = _builder_source()
    for token in (
        'ROOT_PRIM = "/M1SinglePandaO6"',
        'PANDA_PRIM = f"{ROOT_PRIM}/Panda"',
        'RIGHT_HAND_PRIM = f"{PANDA_PRIM}/right_o6"',
        'EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"',
        "EXPECTED_ACTIVE_DOF_COUNT = 29",
        "EXPECTED_ASSEMBLY_JOINT_COUNT = 2",
        'PANDA_ARM_URDF = "panda_arm.urdf"',
    ):
        assert token in source
    assert "panda_arm_hand.urdf" not in source


def test_builder_reads_only_the_normalized_right_o6_source():
    source = _builder_source()
    assert 'RIGHT_O6_ENTRY = "o6_right/O6_right.usd"' in source
    assert "o6_left/O6_left.usd" not in source
    tree = ast.parse(source)
    prefix_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "write_right_prefixed_o6"
    ]
    assert len(prefix_calls) == 1
    assert ast.unparse(prefix_calls[0].args[0]) == "o6_source_root / RIGHT_O6_ENTRY"
    assert ast.unparse(prefix_calls[0].args[1]) == "asset_root / 'prefixed/right_o6.usd'"


def test_builder_has_explicit_reopen_and_manifest_phases():
    source = _builder_source()
    for name in (
        "validate_o6_source",
        "ensure_arm_only_panda",
        "write_right_prefixed_o6",
        "create_stage",
        "assemble_panda",
        "assemble_right_o6",
        "remove_child_roots_scenes_and_root_joints",
        "author_convex_collision_approximations",
        "validate_stage_contract",
        "export_reopen_validate_and_manifest",
        "build_asset",
    ):
        assert f"def {name}(" in source


def test_verifier_freezes_runtime_gate_and_reports_required_metrics():
    assert VERIFIER.is_file()
    source = VERIFIER.read_text(encoding="utf-8")
    for token in (
        "EXPECTED_ACTIVE_DOF_COUNT = 29",
        "EXPECTED_PHYSICS_STEPS = 2000",
        'EXPECTED_ARTICULATION_ROOT = "/M1SinglePandaO6/BASE_LINK"',
        '"measured_physical_dof_count"',
        '"four_wheel_contact_ratio"',
        '"max_mount_position_drift_m"',
        '"max_mount_orientation_drift_rad"',
        '"nonfinite_count"',
        '"hard_joint_limit_count"',
        '"unexpected_contact_count"',
        '"unexpected_reset_count"',
        '"base_instability_count"',
        '"hard_gates_passed"',
    ):
        assert token in source


def test_runtime_physical_count_is_compared_to_manifest_not_magic_34():
    source = VERIFIER.read_text(encoding="utf-8")
    assert (
        'runtime["measured_physical_dof_count"] == manifest["physical_dof_count"]'
        in source
    )
    assert 'runtime["measured_physical_dof_count"] == 34' not in source
