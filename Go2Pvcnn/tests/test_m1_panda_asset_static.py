import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "m1_panda"


def _load_verifier_contract_helpers():
    path = ROOT / "scripts" / "verify_m1_panda_asset.py"
    tree = ast.parse(path.read_text())
    helper_names = {
        "_classify_unresolved_dependencies",
        "_articulation_root_errors",
        "_mount_joint_contract_errors",
        "_mount_plane_errors",
        "_surface_gap_errors",
    }
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert helper_names <= functions.keys()
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {
            "BUILTIN_MDL_ALLOWLIST",
            "EXPECTED_ARTICULATION_ROOT",
            "EXPECTED_MOUNT_BODY0",
            "EXPECTED_MOUNT_BODY1",
            "EXPECTED_MOUNT_CHILD_LOCAL_POS",
            "EXPECTED_MOUNT_CHILD_LOCAL_ROT",
        }
    ]
    namespace = {}
    exec(compile(ast.Module(body=constants + [functions[name] for name in helper_names], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_m1_panda_sources_are_project_owned_and_manifested():
    manifest = json.loads((ASSET_ROOT / "source_manifest.json").read_text())
    assert manifest["m1"]["entry"] == "m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd"
    assert manifest["panda"]["entry"] == "panda_source/franka_description/robots/panda_arm_hand.urdf"
    assert (ASSET_ROOT / manifest["m1"]["entry"]).is_file()
    assert (ASSET_ROOT / manifest["panda"]["entry"]).is_file()


def test_m1_overlay_has_only_a_relative_local_sublayer():
    text = (ASSET_ROOT / "m1_floating.usda").read_text()
    assert "@./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@" in text
    assert "/home/" not in text
    assert "omniverse://" not in text
    assert "http://" not in text
    assert "https://" not in text


def test_builder_declares_single_robot_mount_contract():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    assert 'ROOT_PRIM = "/M1Panda"' in source
    assert 'BASE_MOUNT_FRAME = "/BASE_LINK"' in source
    assert 'PANDA_MOUNT_FRAME = "/panda_link0"' in source
    assert 'MOUNT_JOINT_PATH = f"{PANDA_PRIM}/panda_link0/AssemblerFixedJoint"' in source
    assert "assemble_articulations(" in source
    assert "single_robot=True" in source
    assert "EXPECTED_MOUNT_CHILD_LOCAL_POS = (0.0, 0.0, 0.0)" in source
    assert "EXPECTED_MOUNT_CHILD_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)" in source
    assert "GetLocalPos1Attr().Get()" in source
    assert "GetLocalRot1Attr().Get()" in source


def test_builder_uses_exact_zero_mount_clearance_and_top_plane_offset():
    path = ROOT / "scripts" / "build_m1_panda_asset.py"
    tree = ast.parse(path.read_text())
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "MOUNT_CLEARANCE_M"
    }
    assert values == {"MOUNT_CLEARANCE_M": 0.0}
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    namespace = {"np": __import__("numpy")}
    exec(
        compile(
            ast.Module(
                body=[functions["mount_offset_z"]], type_ignores=[]
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    assert namespace["mount_offset_z"](0.42, 0.17, 0.0) == pytest.approx(
        0.25
    )


def test_builder_measures_the_local_visible_mount_patch_not_global_base_maximum():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    assert "MOUNT_PATCH_HALF_EXTENTS_M = (0.11, 0.10)" in source
    assert "def _mount_patch_top_z(" in source
    assert '"/visuals/" not in mesh_path' in source
    assert "base_top_z = _mount_patch_top_z(" in source
    assert "base_top_z = _top_z(" not in source


def test_surface_gap_predicate_rejects_both_air_gap_and_penetration():
    errors = _load_verifier_contract_helpers()["_surface_gap_errors"]
    assert errors(0.0, 1.0e-6) == []
    assert errors(0.9e-6, 1.0e-6) == []
    assert errors(-0.9e-6, 1.0e-6) == []
    assert errors(2.0e-6, 1.0e-6)
    assert errors(-2.0e-6, 1.0e-6)
    assert errors(None, 1.0e-6)


def test_builder_enables_robot_assembler_after_app_startup():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    app_start = source.index("simulation_app = app_launcher.app")
    enable = source.index('enable_extension("isaacsim.robot_setup.assembler")')
    assembler_import = source.index("from isaacsim.robot_setup.assembler import RobotAssembler")
    assert app_start < enable < assembler_import


def test_builder_reports_failures_before_closing_kit_and_exits_reliably():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert {"os", "sys", "traceback"} <= imports
    report = source.index("traceback.print_exc()")
    flush = source.index("sys.stderr.flush()", report)
    failure_exit = source.index("os._exit(1)", flush)
    close = source.index("simulation_app.close()", failure_exit)
    success_exit = source.index("os._exit(0)", close)
    assert report < flush < failure_exit < close < success_exit


def test_builder_reuses_panda_and_supports_isaac_sim_5_assembler():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    assert 'parser.add_argument("--force-panda-conversion"' in source
    assert "force_panda_conversion: bool = False" in source
    guard = source.index(
        "if force_panda_conversion or not panda_usd.is_file():"
    )
    converter = source.index("converter = UrdfConverter(", guard)
    stage = source.index("stage_utils.create_new_stage()", converter)
    assert guard < converter < stage
    assert 'hasattr(assembler, "assemble_articulations")' in source
    assert "assembler.assemble_rigid_bodies(" in source
    assert "MakeMatrixXform().Set(" in source
    assert "panda_root_joint.SetActive(True)" in source
    assert "panda_root_joint.RemoveAPI(UsdPhysics.ArticulationRootAPI)" in source
    assert "panda_root_joint.RemoveAPI(PhysxSchema.PhysxArticulationAPI)" in source
    assert 'GetAttribute("physics:jointEnabled").Set(False)' in source


def test_verifier_checks_offline_and_topology_contracts():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    for token in (
        "ComputeAllDependencies",
        "remote_dependencies",
        "EXPECTED_DOF_COUNT = 25",
        '"BASE_LINK"',
        '"panda_link0"',
        '"panda_hand"',
        '"/M1Panda/Panda/panda_link0/AssemblerFixedJoint"',
    ):
        assert token in source


def test_verifier_emits_json_and_forces_a_reliable_exit_without_kit_cleanup():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    emit = source.index("print(json.dumps(result, sort_keys=True))")
    flush = source.index("sys.stdout.flush()", emit)
    forced_exit = source.index("os._exit(exit_code)", flush)
    assert emit < flush < forced_exit
    assert "simulation_app.close()" not in source


def test_verifier_forces_nonzero_exit_before_kit_cleanup_on_failure():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    assert "sys.stdout.flush()" in source
    assert "os._exit(exit_code)" in source


def test_verifier_reports_dependency_and_physics_evidence_together():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    assert '"unresolved_dependencies"' in source
    assert '"validation_errors"' in source
    assert '"physics_steps": 1' in source


def test_verifier_supplies_required_empty_actuator_mapping():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    assert "actuators={}" in source


def test_builder_serializes_a_relocatable_single_root_asset():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    for token in (
        'Sdf.Reference("/M1Panda")',
        'Sdf.Reference("/M1Panda/Panda")',
        'Sdf.Payload("/M1Panda/Panda")',
        'Sdf.Reference("m1_floating.usda")',
        'Sdf.Reference("panda/panda.usd")',
        "RemoveAPI(UsdPhysics.ArticulationRootAPI)",
        "RemoveAPI(PhysxSchema.PhysxArticulationAPI)",
        'GetAttribute("physics:excludeFromArticulation")',
        'GetAttribute("physics:jointEnabled")',
        "raise RuntimeError",
        "Export(str(combined_usd))",
    ):
        assert token in source


def test_verifier_only_allows_the_declared_builtin_mdl_dependency():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "BUILTIN_MDL_ALLOWLIST"
    }
    assert assignments == {"BUILTIN_MDL_ALLOWLIST": {"OmniPBR.mdl"}}
    assert '"builtin_mdl_dependencies"' in source
    assert '"unresolved_dependencies"' in source
    assert "if item in BUILTIN_MDL_ALLOWLIST" in source


def test_verifier_requires_successful_initialization_with_legal_panda_home_pose():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    for token in (
        '"panda_joint2": -0.569',
        '"panda_joint4": -2.810',
        '"panda_joint6": 3.037',
        '"panda_joint7": 0.741',
        '"panda_finger_joint.*": 0.04',
        "init_state=ArticulationCfg.InitialStateCfg(",
        "robot.is_initialized",
        '"runtime_initialized"',
        '"panda_root_joint_enabled"',
        '"mount_relative_step_delta_m"',
        "MAX_MOUNT_RELATIVE_STEP_DELTA_M",
    ):
        assert token in source


def test_dependency_classification_only_allows_exact_omnipbr():
    helpers = _load_verifier_contract_helpers()
    builtin, unresolved = helpers["_classify_unresolved_dependencies"](
        ["OmniPBR.mdl", "Other.mdl", "textures/OmniPBR.mdl", "/M1Panda"]
    )
    assert builtin == ["OmniPBR.mdl"]
    assert unresolved == ["Other.mdl", "textures/OmniPBR.mdl", "/M1Panda"]


def test_articulation_root_predicate_requires_exact_m1_base_link():
    errors = _load_verifier_contract_helpers()["_articulation_root_errors"]
    assert errors(["/M1Panda/BASE_LINK"]) == []
    assert errors(["/M1Panda/Panda"])
    assert errors([])
    assert errors(["/M1Panda/BASE_LINK", "/M1Panda/Panda"])


def test_mount_contract_predicate_checks_targets_enabled_and_inclusion():
    errors = _load_verifier_contract_helpers()["_mount_joint_contract_errors"]
    expected = (
        ["/M1Panda/BASE_LINK"],
        ["/M1Panda/Panda/panda_link0"],
        True,
        False,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert errors(*expected) == []
    assert errors(["/Wrong"], *expected[1:])
    assert errors(expected[0], ["/Wrong"], *expected[2:])
    assert errors(*expected[:2], False, *expected[3:])
    assert errors(*expected[:3], True, *expected[4:])
    assert errors(*expected[:4], (0.1, 0.0, 0.0), expected[5])
    assert errors(*expected[:5], (0.0, 1.0, 0.0, 0.0))


def test_mount_plane_predicate_enforces_micrometer_tolerance():
    errors = _load_verifier_contract_helpers()["_mount_plane_errors"]
    assert errors(
        (0.0, 0.0, 0.25), (0.0, 0.0, 0.25), 1.0e-6
    ) == []
    assert errors(
        (0.0, 0.0, 0.250002), (0.0, 0.0, 0.25), 1.0e-6
    )
    assert errors(None, (0.0, 0.0, 0.25), 1.0e-6)


def test_builder_exposes_behavioral_cleanup_and_serialized_validation_helpers():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_remove_refresh_asset_edits" in names
    assert "_validate_serialized_asset" in names
