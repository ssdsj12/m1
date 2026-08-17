import math

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.impedance import apply_impedance
from go2_pvcnn.control.m1_panda_coordination.safety import (
    BalanceSafetySupervisor,
    SafetyCfg,
    SafetyState,
)


def _impedance_inputs():
    return {
        "q": torch.zeros(23, dtype=torch.float64),
        "qd": torch.zeros(23, dtype=torch.float64),
        "q_des": torch.ones(23, dtype=torch.float64),
        "qd_des": torch.full((23,), 0.5, dtype=torch.float64),
        "tau_ff": torch.full((23,), 0.25, dtype=torch.float64),
        "kp": torch.full((23,), 2.0, dtype=torch.float64),
        "kd": torch.full((23,), 3.0, dtype=torch.float64),
        "effort_limit": torch.full((23,), 100.0, dtype=torch.float64),
    }


def test_impedance_adds_feedforward_position_and_velocity_feedback():
    inputs = _impedance_inputs()

    effort = apply_impedance(**inputs)

    assert torch.equal(effort, torch.full((23,), 3.75, dtype=torch.float64))


def test_impedance_clamps_symmetrically_and_supports_batches():
    inputs = _impedance_inputs()
    for name, value in tuple(inputs.items()):
        inputs[name] = torch.stack((value, -value))
    inputs["effort_limit"] = torch.ones(2, 23, dtype=torch.float64)

    effort = apply_impedance(**inputs)

    assert effort.shape == (2, 23)
    assert torch.max(effort).item() <= 1.0
    assert torch.min(effort).item() >= -1.0


def test_impedance_rejects_non_finite_or_mismatched_inputs_atomically():
    inputs = _impedance_inputs()
    inputs["tau_ff"][4] = float("nan")
    with pytest.raises(ValueError, match="tau_ff must contain only finite values"):
        apply_impedance(**inputs)

    inputs = _impedance_inputs()
    inputs["kp"] = torch.ones(22, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"kp must end with shape \(23,\)"):
        apply_impedance(**inputs)


def test_safety_default_thresholds_are_frozen():
    cfg = SafetyCfg()

    assert cfg.warning_angle_rad == pytest.approx(math.radians(7.0))
    assert cfg.critical_angle_rad == pytest.approx(math.radians(10.0))
    assert cfg.required_wheel_contacts == 4
    assert cfg.max_lateral_slip == pytest.approx(0.05)
    assert cfg.unsafe_samples_to_advance == 2
    assert cfg.safe_samples_to_recover == 20


def _supervisor():
    return BalanceSafetySupervisor(
        SafetyCfg(retract_rate_rad_per_step=0.05),
        safe_arm_target=torch.zeros(7, dtype=torch.float64),
    )


def _update(supervisor, **overrides):
    values = {
        "roll": 0.0,
        "pitch": 0.0,
        "wheel_contact_count": 4,
        "max_lateral_slip": 0.0,
        "qp_success": True,
        "signals_finite": True,
        "current_arm_target": torch.ones(7, dtype=torch.float64),
    }
    values.update(overrides)
    return supervisor.update(**values)


def test_two_consecutive_unsafe_samples_advance_one_state_at_a_time():
    supervisor = _supervisor()

    expected = (
        SafetyState.SCALE,
        SafetyState.HOLD,
        SafetyState.RETRACT,
        SafetyState.TERMINATE,
    )
    for state in expected:
        first = _update(supervisor, roll=math.radians(8.0))
        second = _update(supervisor, roll=math.radians(8.0))
        assert first.state != state
        assert second.state == state


@pytest.mark.parametrize(
    "overrides",
    [
        {"roll": math.radians(8.0)},
        {"pitch": math.radians(-11.0)},
        {"wheel_contact_count": 3},
        {"max_lateral_slip": 0.051},
        {"qp_success": False},
    ],
)
def test_each_balance_failure_signal_advances_to_scale(overrides):
    supervisor = _supervisor()

    _update(supervisor, **overrides)
    decision = _update(supervisor, **overrides)

    assert decision.state == SafetyState.SCALE
    assert decision.twist_scale < 1.0
    assert not decision.stop_wheels


def test_hold_and_retract_stop_wheels_and_retract_without_snap():
    supervisor = _supervisor()
    unsafe = {"roll": math.radians(8.0)}
    for _ in range(4):
        decision = _update(supervisor, **unsafe)

    assert decision.state == SafetyState.HOLD
    assert decision.stop_wheels
    held = decision.arm_target.clone()
    assert torch.equal(held, torch.ones(7, dtype=torch.float64))

    _update(supervisor, **unsafe)
    retract = _update(supervisor, **unsafe)
    assert retract.state == SafetyState.RETRACT
    assert retract.stop_wheels
    assert torch.all(retract.arm_target < held)
    assert torch.allclose(retract.arm_target, held - 0.05)
    assert not torch.equal(retract.arm_target, torch.zeros_like(held))


def test_twenty_safe_samples_recover_exactly_one_level():
    supervisor = _supervisor()
    for _ in range(6):
        _update(supervisor, roll=math.radians(8.0))
    assert supervisor.state == SafetyState.RETRACT

    for _ in range(19):
        decision = _update(supervisor)
        assert decision.state == SafetyState.RETRACT
    decision = _update(supervisor)
    assert decision.state == SafetyState.HOLD


def test_terminal_state_latches_until_reset():
    supervisor = _supervisor()
    for _ in range(8):
        decision = _update(supervisor, roll=math.radians(8.0))
    assert decision.state == SafetyState.TERMINATE
    assert decision.terminate

    for _ in range(100):
        decision = _update(supervisor)
    assert decision.state == SafetyState.TERMINATE

    supervisor.reset(torch.full((7,), 0.25, dtype=torch.float64))
    decision = _update(
        supervisor, current_arm_target=torch.full((7,), 0.25, dtype=torch.float64)
    )
    assert decision.state == SafetyState.TRACK
    assert not decision.terminate
    assert decision.twist_scale == pytest.approx(1.0)


def test_non_finite_signal_terminates_and_reuses_last_finite_target():
    supervisor = _supervisor()
    finite_target = torch.full((7,), 0.4, dtype=torch.float64)
    _update(supervisor, current_arm_target=finite_target)

    decision = _update(
        supervisor,
        roll=float("nan"),
        signals_finite=False,
        current_arm_target=torch.full((7,), float("nan"), dtype=torch.float64),
    )

    assert decision.state == SafetyState.TERMINATE
    assert decision.terminate
    assert decision.stop_wheels
    assert torch.equal(decision.arm_target, finite_target)
    assert torch.isfinite(decision.arm_target).all()


def test_reset_rejects_non_finite_arm_target():
    supervisor = _supervisor()
    with pytest.raises(ValueError, match="current_arm_target must contain only finite values"):
        supervisor.reset(torch.full((7,), float("nan"), dtype=torch.float64))
