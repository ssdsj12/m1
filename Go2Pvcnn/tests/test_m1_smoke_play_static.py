from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILE = REPO_ROOT / "scripts" / "m1_smoke_play.py"


def test_m1_smoke_play_script_uses_m1_controller_and_task():
    source = SCRIPT_FILE.read_text()

    assert "Isaac-M1-Smoke-v0" in source
    assert "build_m1_smoke_action" in source
    assert "--mode" in source
    assert 'choices=["rolling", "wave"]' in source
    assert "env.step(actions)" in source
    assert "go2_pvcnn.tasks" in source
