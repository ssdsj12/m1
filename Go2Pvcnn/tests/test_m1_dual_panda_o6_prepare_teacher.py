import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_prepare_teacher.py"
SPEC = importlib.util.spec_from_file_location("prepare_teacher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepare_keeps_only_finite_feasible_rows_and_makes_groups(tmp_path):
    count = 12
    np.savez_compressed(
        tmp_path / "teacher_failures.npz",
        state=np.zeros((count, 3), dtype=np.float32),
        teacher_action=np.zeros((count, 2, 43), dtype=np.float32),
        teacher_task=np.zeros((count, 2, 4), dtype=np.float32),
        phase=np.zeros((count, 1), dtype=np.float32),
        seed=np.full(count, 42, dtype=np.int64),
        trial=np.zeros(count, dtype=np.int64),
        reason=np.asarray(["mission_not_done"] * 11 + ["infeasible"]),
        action_order=np.asarray([f"joint_{i}" for i in range(43)]),
    )
    report = MODULE.prepare(tmp_path, groups=3)
    assert report["prepared_rows"] == 11
    with np.load(tmp_path / "teacher_success.npz", allow_pickle=False) as data:
        assert data["teacher_action"].shape == (11, 2, 43)
        assert set(data["trial"].tolist()) == {0, 1, 2}
