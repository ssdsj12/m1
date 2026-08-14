import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "m1_panda"


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
