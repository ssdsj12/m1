from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py"
WRAPPER = ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
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
    assert constants["BOX_SIZE_M"] == (0.12, 0.18, 0.10)
    assert constants["BOX_MASS_KG"] == 0.5
    assert constants["PHYSICS_DT"] == 0.005
    assert constants["PRIVATE_ACTION_DIM"] == 43
    assert "private_action_dim = PRIVATE_ACTION_DIM" in source
    assert "self.sim.dt = PHYSICS_DT" in source
    assert "self.decimation = 1" in source
    assert "preserve_order=True" in source
    assert "M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES" in source
    assert "self.events = None" in source


def test_scene_has_fixed_box_table_and_all_contact_groups():
    source = _source(ENV_CFG)
    assert "RigidObjectCfg(" in source
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS_KG)" in source
    assert "kinematic_enabled=True" in source
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


def test_wrapper_handles_root_inclusive_and_legacy_jacobian_body_layouts():
    source = _source(WRAPPER)
    assert "if jacobian_body_count == body_count:" in source
    assert "jacobian_body_id = palm_id" in source
    assert "elif jacobian_body_count == body_count - 1:" in source
    assert "jacobian_body_id = palm_id - 1" in source


def test_probe_has_required_startup_smoke_cli():
    source = _source(PROBE)
    for option in ("--num-envs", "--steps", "--seed", "--headless"):
        assert option in source
    assert "finite_snapshot" in source
    assert "action_dim" in source
    assert "unexpected_reset_count" in source
