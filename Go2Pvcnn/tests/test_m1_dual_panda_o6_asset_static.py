from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts/build_m1_dual_panda_o6_asset.py"


def _source() -> str:
    assert BUILDER.is_file()
    return BUILDER.read_text(encoding="utf-8")


def test_builder_freezes_namespaces_mounts_and_dof_contract():
    source = _source()
    assert 'ROOT_PRIM = "/M1DualPandaO6"' in source
    assert 'PLATFORM_JOINT_NAME = "dual_arm_platform_yaw_joint"' in source
    assert 'LEFT_ARM_PRIM = f"{ROOT_PRIM}/left_arm"' in source
    assert 'RIGHT_ARM_PRIM = f"{ROOT_PRIM}/right_arm"' in source
    assert 'LEFT_HAND_PRIM = f"{LEFT_ARM_PRIM}/left_o6"' in source
    assert 'RIGHT_HAND_PRIM = f"{RIGHT_ARM_PRIM}/right_o6"' in source
    assert "EXPECTED_ACTIVE_DOF_COUNT = 43" in source
    assert "EXPECTED_ASSEMBLY_JOINT_COUNT = 5" in source


def test_builder_uses_arm_only_panda_and_both_normalized_o6_entries():
    source = _source()
    assert 'PANDA_ARM_URDF = "panda_arm.urdf"' in source
    assert "panda_arm_hand.urdf" not in source
    assert '"o6_left/O6_left.usd"' in source
    assert '"o6_right/O6_right.usd"' in source
    assert "panda_finger_joint" in source


def test_builder_freezes_symmetric_mount_transforms_and_platform_limit():
    tree = ast.parse(_source())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"PLATFORM_YAW_LIMIT_RAD", "LEFT_ARM_MOUNT_XYZ", "RIGHT_ARM_MOUNT_XYZ"}
    }
    assert assignments["PLATFORM_YAW_LIMIT_RAD"] == (-1.5707963267948966, 1.5707963267948966)
    assert assignments["LEFT_ARM_MOUNT_XYZ"] == (0.0, 0.2, 0.0)
    assert assignments["RIGHT_ARM_MOUNT_XYZ"] == (0.0, -0.2, 0.0)


def test_builder_never_writes_external_o6_sources():
    tree = ast.parse(_source())
    write_calls = {
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(token in ast.unparse(node.func).lower() for token in ("write", "export", "save"))
    }
    assert all("source_o6_root" not in call for call in write_calls)


def test_builder_exposes_required_phases_and_reopen_validation():
    source = _source()
    for phase in (
        "validate_source_manifests",
        "ensure_arm_only_panda",
        "create_m1_and_platform_stage",
        "author_platform_revolute_joint",
        "assemble_panda",
        "assemble_o6",
        "remove_child_roots_scenes_and_root_joints",
        "author_convex_collision_approximations",
        "validate_stage_contract",
        "export_reopen_validate_and_manifest",
    ):
        assert f"def {phase}(" in source
