"""Kinematic regression from seed42 full capture, step2321 (not step1457)."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import ArmMpcCfg, ArmMpcInput
from go2_pvcnn.control.m1_panda_coordination import arm_mpc
from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import SafetyProjection
from tests.test_m1_bimanual_safety_projection import _input
from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime
from tests.test_m1_bimanual_object_mpc import _snapshot
from test_m1_bimanual_prior_startup_binding import _load_wrapper


def _captured_arm():
    # Reuse invariant dynamics/limits from the earlier captured fixture, but
    # explicitly reproduce the later failing state from full_seed42 step2321.
    data = json.loads((Path(__file__).parent / "fixtures/wire_contact_left_arm_qp_seed42.json").read_text())
    sample = ArmMpcInput(**{k: torch.tensor(v, dtype=torch.float64) for k, v in data["input"].items()})
    return replace(sample,
        q=torch.tensor([1.4184274673461914, -1.2630927562713623, -1.1884880065917969,
                        -2.2387969493865967, -1.3757792711257935, 2.2516582012176514,
                        2.172950029373169], dtype=torch.float64),
        qd=torch.tensor([-0.18320956826210022, -0.104712575674057, -0.11334368586540222,
                         0.14258232712745667, 0.8750501871109009, 0.14804400503635406,
                         2.5809223651885986], dtype=torch.float64))


def _safety_sample(arm):
    sample = _input()
    q, qd, lower, upper, speed = [getattr(sample, name).clone() for name in
                                  ("active_q", "active_qd", "q_min", "q_max", "qd_max")]
    q[17:24], qd[17:24] = arm.q, arm.qd
    lower[17:24], upper[17:24], speed[17:24] = arm.q_min, arm.q_max, arm.qd_max
    right = replace(arm, q=(arm.q_min + arm.q_max) / 2, qd=torch.zeros(7, dtype=torch.float64))
    q[24:31], lower[24:31], upper[24:31], speed[24:31] = right.q, right.q_min, right.q_max, right.qd_max
    return replace(sample, active_q=q, active_qd=qd, q_min=lower, q_max=upper, qd_max=speed,
                   arm_mpc_context=((arm, ArmMpcCfg()), (right, ArmMpcCfg())))


def test_captured_state_is_one_step_safe_but_next_horizon_impossible():
    arm = _captured_arm()
    # Node 16 of the actual MPC grid cannot satisfy the upper position bound
    # under ANY acceleration sequence with |qdd| <= 8.
    assert arm.q[6] + .32 * arm.qd[6] - .5 * .32**2 * arm.qdd_max[6] > arm.q_max[6]
    sample = _safety_sample(arm)
    result = SafetyProjection().project(sample)
    assert not result.feasible
    assert result.fallback_reason == "arm_mpc_recoverability"


@pytest.mark.parametrize("sign", [-1., 1.])
def test_braking_gate_checks_both_position_boundaries_and_preserves_inward_motion(sign):
    sample = replace(_captured_arm(), q_min=-torch.ones(7, dtype=torch.float64),
                     q_max=torch.ones(7, dtype=torch.float64),
                     q=torch.full((7,), sign * .99, dtype=torch.float64),
                     qd=torch.full((7,), sign, dtype=torch.float64))
    assert not arm_mpc.arm_mpc_recoverability_ok(sample, ArmMpcCfg())
    assert arm_mpc.arm_mpc_recoverability_ok(replace(sample, qd=-sample.qd), ArmMpcCfg())


def test_grid_feasible_small_velocity_reversal_is_not_rejected_by_continuous_stop_rule():
    sample = replace(_captured_arm(), q_min=-torch.ones(7, dtype=torch.float64),
                     q_max=torch.ones(7, dtype=torch.float64),
                     q=torch.full((7,), .99995, dtype=torch.float64),
                     qd=torch.full((7,), .01, dtype=torch.float64))
    # Acceleration -1 gives q1=q0, v1=-.01; next accelerate +.5 to rest.
    assert arm_mpc.arm_mpc_recoverability_ok(sample, ArmMpcCfg())


def test_gate_uses_configured_acceleration_and_first_node_velocity_bound():
    sample = _captured_arm()
    assert not arm_mpc.arm_mpc_recoverability_ok(sample, ArmMpcCfg())
    assert arm_mpc.arm_mpc_recoverability_ok(replace(sample, qdd_max=2 * sample.qdd_max), ArmMpcCfg())
    centered = replace(sample, q=(sample.q_min+sample.q_max)/2, qd=sample.qd_max + .17)
    assert not arm_mpc.arm_mpc_recoverability_ok(centered, ArmMpcCfg())
    assert arm_mpc.arm_mpc_recoverability_ok(centered, ArmMpcCfg(dt=.04))


def test_safe_candidate_and_last_safe_effort_are_preserved():
    arm = replace(_captured_arm(), qd=torch.zeros(7, dtype=torch.float64))
    sample = _safety_sample(arm)
    result = SafetyProjection().project(sample)
    assert result.feasible
    assert torch.equal(result.effort, sample.candidate_effort)
    unsafe = _safety_sample(_captured_arm())
    unsafe = replace(unsafe, safe_effort=torch.full((43,), .25, dtype=torch.float64))
    result = SafetyProjection().project(unsafe)
    expected = unsafe.safe_effort.clone()
    expected[12:16] = 0
    assert torch.equal(result.effort, expected)
    assert result.effort.shape == (43,)


def test_runtime_exposes_configured_per_arm_limits_and_timing():
    left = _captured_arm()
    right = replace(left, qdd_max=left.qdd_max * 2)
    runtime = BimanualRuntime(arm_input_provider=lambda snapshot: (left, right))
    runtime.arm_mpc.right.cfg = ArmMpcCfg(dt=.04)
    context = runtime.arm_recoverability_context(_snapshot())
    assert context == ((left, runtime.arm_mpc.left.cfg), (right, runtime.arm_mpc.right.cfg))
    # Injected legacy planners without a grid retain their existing contract.
    runtime.arm_mpc = SimpleNamespace()
    assert runtime.arm_recoverability_context(_snapshot()) == ()


def test_wrapper_passes_actual_arm_context_to_final_safety_boundary(monkeypatch):
    module = _load_wrapper(monkeypatch)
    wrapper = object.__new__(module.M1DualPandaO6BimanualWrapper)
    sample = _safety_sample(_captured_arm())
    left, right = (entry[0] for entry in sample.arm_mpc_context)
    runtime = BimanualRuntime(arm_input_provider=lambda snapshot: (left, right))
    data = SimpleNamespace(
        soft_joint_pos_limits=torch.stack((sample.q_min, sample.q_max), dim=-1)[None],
        soft_joint_vel_limits=sample.qd_max[None],
        root_state_w=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=torch.float64),
    )
    lane = SimpleNamespace(
        runtime=runtime, env_index=0,
        adapter=SimpleNamespace(
            active_joint_ids=tuple(range(43)),
            left_arm_ids=tuple(range(17, 24)), right_arm_ids=tuple(range(24, 31)),
            robot=SimpleNamespace(data=data),
        ),
        baseline_command=SimpleNamespace(effort=sample.safe_effort),
        base_reference_w=data.root_state_w[0],
    )
    wrapper._active_q_qd = lambda lane: (sample.active_q, sample.active_qd)
    wrapper._effort_limits = sample.effort_limits
    actual = wrapper._safety_input(lane, _snapshot(), sample.dynamics, sample.candidate_effort)
    assert len(actual.arm_mpc_context) == 2
    torch.testing.assert_close(actual.arm_mpc_context[0][0].q, left.q)
    torch.testing.assert_close(actual.arm_mpc_context[1][0].q, right.q)
    result = SafetyProjection().project(actual)
    assert not result.feasible
    assert result.fallback_reason == "arm_mpc_recoverability"


@pytest.mark.parametrize('path', ['candidate', 'projected', 'fallback'])
def test_external_hand_servo_cannot_change_effort_after_safety_certificate(path):
    # Seed42 step2320 projected nonzero O6 efforts, then the wrapper discarded
    # them. The certified Panda braking acceleration consequently changed sign.
    sample = _input()
    candidate = sample.candidate_effort.clone()
    candidate[31:43] = 3.
    if path == 'projected':
        candidate[17] = 1.e6
    elif path == 'fallback':
        candidate[0] = torch.nan
    sample = replace(sample, candidate_effort=candidate,
                     safe_effort=torch.full((43,), .25, dtype=torch.float64))
    result = SafetyProjection(externally_servoed_hand=True).project(sample)
    assert result.feasible is (path != 'fallback')
    assert torch.equal(result.effort[31:43], torch.zeros(12, dtype=torch.float64))
