from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from tests.test_m1_bimanual_runtime import _runtime, _WbcController
from tests.test_m1_bimanual_object_mpc import _snapshot


def test_policy_palm_goals_do_not_call_object_or_motion_planner():
    runtime = _runtime()
    runtime.object_mpc.plan = lambda sample: pytest.fail("PPO must own the task goals")
    action = torch.zeros(15)
    action[0] = 1.
    runtime.set_ppo_command(action)
    snapshot = _snapshot()
    runtime.compute(snapshot)
    goal = runtime.latest_solutions["object"]
    assert goal.left_palm_pose[-1, 0] > snapshot.left_arm.palm_pose_b[0]
    assert goal.right_palm_pose[-1, 0] > snapshot.right_arm.palm_pose_b[0]


def test_policy_hand_synergies_are_separate_and_limited():
    from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_control import ppo_hand_reference

    snapshot = _snapshot()
    a = torch.zeros(15)
    a[6] = 1.
    left = ppo_hand_reference(snapshot.left_hand.q, a, "left")
    right = ppo_hand_reference(snapshot.right_hand.q, a, "right")
    assert not torch.equal(left, snapshot.left_hand.q)
    assert torch.equal(right, snapshot.right_hand.q)
    for value in (-1., 1.):
        result = ppo_hand_reference(torch.full((6,), 2., dtype=torch.float64), torch.full((15,), value), "left")
        assert torch.all(result >= 0.)
        assert torch.all(result <= result.new_tensor([.58, 1.36, 1.6, 1.6, 1.6, 1.6]))


def test_policy_phase_force_has_real_phase_and_bilateral_wrench_effects():
    baseline = torch.zeros(15)
    baseline_runtime = _runtime()
    baseline_runtime.set_ppo_command(baseline)
    baseline_runtime.compute(_snapshot())
    baseline_solution = baseline_runtime.latest_solutions["object"]
    baseline_force = (
        torch.linalg.vector_norm(baseline_solution.left_wrench[0, :3])
        + torch.linalg.vector_norm(baseline_solution.right_wrench[0, :3])
    )

    phase = baseline.clone()
    phase[12] = -.4  # PRELOAD
    phase_runtime = _runtime()
    phase_runtime.set_ppo_command(phase)
    phase_runtime.compute(_snapshot())
    assert phase_runtime.last_ppo_requested_phase is BimanualPhase.PRELOAD

    clamp = baseline.clone()
    clamp[13] = 1.
    clamp_runtime = _runtime()
    clamp_runtime.set_ppo_command(clamp)
    clamp_runtime.compute(_snapshot())
    clamp_solution = clamp_runtime.latest_solutions["object"]
    clamp_force = (
        torch.linalg.vector_norm(clamp_solution.left_wrench[0, :3])
        + torch.linalg.vector_norm(clamp_solution.right_wrench[0, :3])
    )
    assert clamp_force > baseline_force

    split = baseline.clone()
    split[14] = .8  # favor left
    split_runtime = _runtime()
    split_runtime.set_ppo_command(split)
    split_runtime.compute(_snapshot())
    split_solution = split_runtime.latest_solutions["object"]
    assert (
        torch.linalg.vector_norm(split_solution.left_wrench[0, :3])
        > torch.linalg.vector_norm(split_solution.right_wrench[0, :3])
        > 0.
    )


def test_policy_grasp_margin_requires_real_bilateral_contact():
    from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_control import ppo_object_reference
    snapshot = _snapshot()
    for side in ("left", "right"):
        hand = getattr(snapshot, f"{side}_hand")
        forces = torch.zeros(5, 3, dtype=torch.float64)
        forces[:2, 2] = .4
        snapshot = replace(snapshot, **{f"{side}_hand": replace(hand,
            fingertip_forces_b=forces, contact_mask=torch.tensor([True, True, False, False, False]))})
    assert ppo_object_reference(snapshot, torch.zeros(15)).diagnostics.force_closure_margin > 0.
    snapshot = replace(snapshot, right_hand=replace(snapshot.right_hand, contact_mask=torch.zeros(5, dtype=torch.bool)))
    assert ppo_object_reference(snapshot, torch.zeros(15)).diagnostics.force_closure_margin == 0.


def test_policy_fallback_does_not_publish_failed_new_targets():
    runtime = _runtime(wbc=_WbcController(fail_at=2))
    runtime.set_ppo_command(torch.zeros(15))
    first = runtime.compute(_snapshot())
    safe = runtime._last_object
    runtime.set_ppo_command(torch.ones(15))
    second = runtime.compute(replace(_snapshot(), timestamp_ns=10_000_000))
    assert not second.feasible
    assert torch.equal(second.effort, first.effort)
    assert runtime._last_object is safe


def test_wrapper_exposes_last_latent_and_clips_hand_servo_to_real_limits(monkeypatch):
    from tests.test_m1_bimanual_prior_startup_binding import _load_wrapper
    M1DualPandaO6BimanualWrapper = _load_wrapper(monkeypatch).M1DualPandaO6BimanualWrapper
    wrapper = object.__new__(M1DualPandaO6BimanualWrapper)
    wrapper.step = lambda: None
    wrapper.step_latent(torch.ones(15))
    assert torch.equal(wrapper.last_ppo_latent, torch.ones(15, dtype=torch.float64))
    wrapper.last_ppo_latent.zero_()
    assert torch.count_nonzero(wrapper.last_ppo_latent) == 15
    lane = SimpleNamespace(runtime=SimpleNamespace(mission=SimpleNamespace(phase=BimanualPhase.GRASP), latest_solutions={
        "left_hand": SimpleNamespace(q_ref=torch.full((6,), 9., dtype=torch.float64)),
        "right_hand": SimpleNamespace(q_ref=torch.full((6,), -9., dtype=torch.float64))}),
        adapter=SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(
            soft_joint_pos_limits=torch.tensor([[[.1, .5]]] * 12).reshape(1, 12, 2))),
            left_hand_ids=range(6), right_hand_ids=range(6, 12)), env_index=0,
            wire_acceptance_tracker=None)
    left, right = wrapper._lane_hand_targets(lane, _snapshot())
    assert torch.all(left <= .5) and torch.all(right >= .1)


def test_actual_runtime_hand_inputs_carry_independent_policy_targets():
    from tests.test_m1_bimanual_runtime import _RecordingHandController
    runtime = _runtime()
    runtime.left_hand_mpc = _RecordingHandController()
    runtime.right_hand_mpc = _RecordingHandController()
    snapshot = _snapshot()
    snapshot = replace(snapshot, left_hand=replace(snapshot.left_hand,
        fingertip_jacobian_b=torch.eye(15, 6, dtype=torch.float64)))
    action = torch.zeros(15)
    action[6] = 1.
    runtime.set_ppo_command(action)
    runtime.compute(snapshot)
    left, right = runtime.left_hand_mpc.samples[-1], runtime.right_hand_mpc.samples[-1]
    assert not torch.equal(left.fingertip_target_positions_b, snapshot.left_hand.fingertip_positions_b)
    assert torch.equal(right.fingertip_target_positions_b, snapshot.right_hand.fingertip_positions_b)


def test_wrapper_ppo_uses_safety_projection_without_teacher(monkeypatch):
    from tests.test_m1_wire_contact_diagnostics import _acceptance_wrapper
    _, wrapper, lane, snapshot = _acceptance_wrapper(monkeypatch)
    wrapper._ppo_latent_action = torch.zeros(15)
    lane.teacher = SimpleNamespace(plan=lambda sample: pytest.fail("teacher called"))
    lane.safety.project = lambda sample: SimpleNamespace(
        effort=torch.zeros(43, dtype=torch.float64), feasible=False, fallback_reason="test_reject")
    command = wrapper._compute_lane_command(lane, snapshot, None, None)
    assert not command.feasible
    assert "test_reject" in command.fallback_reasons
    assert torch.equal(command.effort, lane.last_safety_result.effort)


def test_ppo_staged_runtime_updates_reduced_state_without_object_planning():
    from tests.test_m1_bimanual_runtime import _staged_runtime, _wire_tick
    runtime, backend = _staged_runtime()
    runtime.object_mpc.plan = lambda _: pytest.fail("Object MPC called in PPO")
    runtime.set_ppo_command(torch.zeros(15))
    for tick in range(1, 15):
        _wire_tick(runtime, tick * 5_000_000)
    assert runtime.latest_reduced_wire_update is not None
    assert torch.isfinite(torch.tensor(runtime.latest_reduced_wire_update.state.positions)).all()


@pytest.mark.parametrize("phase_command", [-1., -.4])
def test_real_hand_projection_holds_zero_policy_before_contact(phase_command):
    from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
    runtime = _runtime()
    runtime.left_hand_mpc = O6HandMpc()
    runtime.right_hand_mpc = O6HandMpc()
    snapshot = _snapshot()
    snapshot = replace(snapshot, left_hand=replace(snapshot.left_hand,
        q=torch.full((6,), .3, dtype=torch.float64), contact_mask=torch.zeros(5, dtype=torch.bool),
        fingertip_jacobian_b=torch.eye(15, 6, dtype=torch.float64)))
    action = torch.zeros(15)
    action[12] = phase_command
    runtime.set_ppo_command(action)
    runtime.compute(snapshot)
    target = runtime.latest_solutions["left_hand"].q_ref
    assert torch.allclose(target, snapshot.left_hand.q, atol=1e-10)
    action[6] = 1.
    runtime.set_ppo_command(action)
    runtime.compute(replace(snapshot, timestamp_ns=5_000_001))
    assert runtime.latest_solutions["left_hand"].q_ref[0] > target[0]
