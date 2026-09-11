import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_10k_sweep.py"
SPEC = importlib.util.spec_from_file_location("t500_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_budget_and_early_failure_refund(tmp_path):
    ledger = MODULE.StepLedger()
    for index in range(4):
        ledger.record(f"candidate-{index}", 350, 350)
    for seed in (42, 43, 44):
        ledger.record(f"seed-{seed}", 1600, 1600)
    assert ledger.remaining == 3800
    ledger.record("teacher", 3800, 3700)
    assert ledger.remaining == 100
    with pytest.raises(ValueError, match="exceeded"):
        ledger.record("overflow", 101, 101)
    path = tmp_path / "state.json"
    ledger.save(path)
    restored = MODULE.StepLedger.load(path)
    assert restored.consumed == 9900
    assert restored.remaining == 100
    with pytest.raises(ValueError, match="already recorded"):
        restored.record("candidate-0", 350, 350)


def _report(rate, *, feasible=1.0, hard=0):
    return {
        "metadata": {"tracking_angular_rate_max_rad_s": rate},
        "trials": [{
            "arm_mpc_feasible_rates": [feasible, feasible],
            "hand_mpc_feasible_rates": [1.0, 1.0],
            "nonfinite_count": 0,
            "reset_count": 0,
            "hard_failure_count": hard,
            "limit_violation_count": 0,
            "first_safety_failure": None,
            "hold_orientation_error_rad": 0.0,
            "relative_palm_slip_m": 0.002,
        }],
    }


def test_selector_rejects_hard_failure_and_prefers_rate_after_feasibility():
    selected = MODULE.select_candidate([
        _report(0.125, hard=1),
        _report(0.175),
        _report(0.25, feasible=0.99),
    ])
    assert selected["metadata"]["tracking_angular_rate_max_rad_s"] == 0.175
