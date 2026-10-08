from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    assert "def _target_contact_sensor(" in source
    assert source.count('filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]') == 1
    assert source.count("= _target_contact_sensor(") == 18
    assert source.count("_target_contact = _target_contact_sensor(") == 18
    assert 'target_contact_prim_path: str = "{ENV_REGEX_NS}/Box"' in source
    assert "sensor.filter_prim_paths_expr = [self.target_contact_prim_path]" in source
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


def _configured_catalog_scene(object_class):
    # Execute the real config method with lightweight Isaac config containers.
    # The wire and legacy Box otherwise share the exact default spawn center.
    tree = ast.parse(_source(ENV_CFG))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "M1DualPandaO6BimanualEnvCfg")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "__post_init__")
    class Base:
        def __post_init__(self):
            pass
    class Scene(SimpleNamespace):
        def __getattr__(self, name):
            value = SimpleNamespace()
            setattr(self, name, value)
            return value
    target = SimpleNamespace(object_id=f"{object_class}_000", object_class=object_class)
    target_cfg = SimpleNamespace(spawn=SimpleNamespace(rigid_props=None, activate_contact_sensors=True))
    obstacle_cfg = SimpleNamespace(spawn=SimpleNamespace(rigid_props=None, activate_contact_sensors=True))
    namespace = {
        "Base": Base, "PHYSICS_DT": .005,
        "select_target_and_obstacles": lambda *args: (target, ()),
        "build_object_scene_cfg": lambda *args: {target.object_id: target_cfg, "wire_obstacle": obstacle_cfg},
        "validate_scene_object_ids": lambda *args: None,
        "M1DualPandaO6BimanualEventsCfg": SimpleNamespace,
        "sim_utils": SimpleNamespace(RigidBodyPropertiesCfg=SimpleNamespace),
    }
    isolated = ast.Module(body=[ast.ClassDef(name="Config", bases=[ast.Name(id="Base", ctx=ast.Load())],
                         keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(ENV_CFG), "exec"), namespace)
    cfg = namespace["Config"]()
    legacy_box = object()
    cfg.scene = Scene(box=legacy_box, box_contacts=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Box"))
    cfg.sim = SimpleNamespace()
    cfg.target_object_id = None if object_class is None else target.object_id
    cfg.object_catalog = None if object_class is None else object()
    cfg.object_instances = () if object_class is None else (target,)
    cfg.__post_init__()
    return cfg, legacy_box, target_cfg, obstacle_cfg


@pytest.mark.parametrize("object_class", (None, "wire", "cup"))
def test_wire_scene_omits_overlapping_legacy_box_only_for_selected_wire(object_class):
    cfg, legacy_box, _, _ = _configured_catalog_scene(object_class)
    if object_class == "wire":
        assert cfg.scene.box is None
        assert cfg.scene.box_contacts.prim_path == "{ENV_REGEX_NS}/wire_000"
    else:
        assert cfg.scene.box is legacy_box
        assert cfg.scene.box_contacts.prim_path == "{ENV_REGEX_NS}/Box"


@pytest.mark.parametrize("object_class", ("wire", "cup"))
def test_contact_stage_fixes_only_selected_wire_and_preserves_contact_reporting(object_class):
    cfg, _, target, obstacle = _configured_catalog_scene(object_class)
    if object_class == "wire":
        assert target.spawn.rigid_props is not None
        assert target.spawn.rigid_props.kinematic_enabled is True
        assert cfg.scene.left_thumb_distal_target_contact.filter_prim_paths_expr == ["{ENV_REGEX_NS}/wire_000"]
    else:
        assert target.spawn.rigid_props is None
    assert target.spawn.activate_contact_sensors is True
    assert obstacle.spawn.rigid_props is None


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
    assert "left_filtered_target_sensor_names" in source
    assert "right_filtered_target_sensor_names" in source
    assert 'f"{body_name}_target_contact"' in source
    assert "must filter contacts against the selected target" in source
    assert "state.bias + 20.0 * (target - state.q) - 8.0 * state.qd" in source


def test_wrapper_uses_gpu_smoke_validated_deformx_substep_by_default():
    source = _source(WRAPPER)
    assert 'os.environ.get("M1_DEFORMX_PY_DT", "1e-6")' in source


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
