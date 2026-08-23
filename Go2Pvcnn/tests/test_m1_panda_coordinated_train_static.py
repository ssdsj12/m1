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
