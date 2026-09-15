from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py"
WRAPPER = ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
FRAME_KINEMATICS = ROOT / "go2_pvcnn/control/m1_bimanual_coordination/frame_kinematics.py"
PROBE = ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"
REGISTER = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"


def _source(path: Path) -> str:
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _constants(path: Path) -> dict[str, object]:
    tree = ast.parse(_source(path))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return result


def test_env_is_200_hz_fixed_condition_and_43_effort():
    constants = _constants(ENV_CFG)
    source = _source(ENV_CFG)
    assert constants["BOX_SIZE_M"] == (0.20, 0.18, 0.10)
    assert constants["SUPPORT_SIZE_M"] == (0.08, 0.12, 0.10)
    assert constants["BOX_MASS_KG"] == 0.5
    assert constants["PHYSICS_DT"] == 0.005
    assert constants["PRIVATE_ACTION_DIM"] == 43
    assert "private_action_dim = PRIVATE_ACTION_DIM" in source
    assert "self.sim.dt = PHYSICS_DT" in source
    assert "self.decimation = 1" in source
    assert "preserve_order=True" in source
    assert "M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES" in source
    assert "self.events = None" in source
    assert 'for name in ("left_o6", "right_o6")' in source
    assert "stiffness=80.0" in source
    assert "damping=5.0" in source
    assert "def _box_contact_sensor(" in source
    assert source.count('filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]') == 1
    assert source.count("= _box_contact_sensor(") == 18
    assert "articulation_props.fix_root_link = True" in source


def test_scene_has_fixed_box_table_and_all_contact_groups():
    source = _source(ENV_CFG)
    assert "RigidObjectCfg(" in source
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS_KG)" in source
    assert "kinematic_enabled=True" in source
    assert "pos=(0.65, 0.0, 1.10)" in source
    assert "pos=(0.65, 0.0, 1.20)" in source
    for name in (
        "o6_contacts",
        "palm_contacts",
        "wrist_contacts",
        "platform_contacts",
        "base_contacts",
        "box_contacts",
    ):
        assert f"{name} = ContactSensorCfg(" in source


def test_registry_is_isolated():
    source = _source(REGISTER)
    assert 'id="Isaac-M1-DualPanda-O6-Bimanual-Lift-v0"' in source
    assert "M1DualPandaO6BimanualEnvCfg" in source
    assert '"rsl_rl_cfg_entry_point": None' in source


def test_wrapper_resolves_ids_once_and_rejects_ambiguous_names():
    source = _source(WRAPPER)
    assert "resolve_active_joint_ids(" in source
    assert "def _exact_body_id(" in source
    assert "expected exactly one body" in source
    assert "class M1DualPandaO6SnapshotAdapter" in source
    assert "class M1DualPandaO6BimanualWrapper" in source
    assert "BimanualRuntime(" in source
    assert "command.effort" in source
    assert "set_joint_position_target(" in source
    assert "left_hand_solution.q_ref" in source
    assert "right_hand_solution.q_ref" in source
    assert "box_position[2] <= 1.205" in source
    assert "def _stabilize_preload_arms(" in source
    assert "contact_body_id_candidates" in source
    assert 'f"{side}_pinky_proximal"' in source
    assert "state.bias + 20.0 * (target - state.q) - 8.0 * state.qd" in source


def test_wrapper_handles_root_inclusive_and_legacy_jacobian_body_layouts():
    source = _source(FRAME_KINEMATICS)
    assert "if jacobian_body_count == body_count:" in source
    assert "return body_id" in source
    assert "if jacobian_body_count == body_count - 1 and body_id > 0:" in source
    assert "return body_id - 1" in source


def test_probe_has_required_startup_smoke_cli():
    source = _source(PROBE)
    for option in ("--num-envs", "--steps", "--seed"):
        assert option in source
    # --headless belongs to IsaacLab's AppLauncher, not the probe's parser.
    # Verify executable wiring on the same parser/namespace, in startup order;
    # a comment or an unrelated launcher call must not satisfy this contract.
    tree = ast.parse(source)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert ast.dump(ast.parse("parser = _parser()").body[0], include_attributes=False) in (
        ast.dump(node, include_attributes=False) for node in main.body
    )
    # Pre-launch pin binding is now owned by main; its launch bridge receives
    # that very parser, then registers and parses AppLauncher flags in order.
    assert any(
        isinstance(node, ast.Return)
        and ast.dump(node, include_attributes=False)
        == ast.dump(
            ast.parse("return _run_with_fingertip_prior(parser, binding)").body[0],
            include_attributes=False,
        )
        for node in ast.walk(main)
    )
    launch = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_with_fingertip_prior"
    )
    assert [argument.arg for argument in launch.args.args] == ["parser", "binding"]
    startup = [ast.dump(node, include_attributes=False) for node in launch.body]
    required = (
        "from isaaclab.app import AppLauncher",
        "AppLauncher.add_app_launcher_args(parser)",
        "args = parser.parse_args()",
        "app_launcher = AppLauncher(args)",
    )
    positions = [
        startup.index(ast.dump(ast.parse(statement).body[0], include_attributes=False))
        for statement in required
    ]
    assert positions == sorted(positions)
    assert "finite_snapshot" in source
    assert "action_dim" in source
    assert "unexpected_reset_count" in source
