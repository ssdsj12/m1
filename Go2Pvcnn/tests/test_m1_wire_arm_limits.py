from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime
from go2_pvcnn.control.m1_bimanual_coordination.safety_projection import SafetyProjection
from go2_pvcnn.control.m1_panda_coordination.arm_mpc import LinearizedArmMpc
from tests.test_m1_bimanual_object_mpc import _snapshot
from test_m1_bimanual_prior_startup_binding import _load_wrapper


def _lane():
    runtime = BimanualRuntime()
    inputs = runtime._arm_inputs(_snapshot())
    # Same Isaac .9 soft-limit contraction used by the real asset.
    limits = []
    for sample in inputs:
        center = (sample.q_min + sample.q_max) / 2
        half = .45 * (sample.q_max - sample.q_min)
        limits.append(torch.stack((center - half, center + half), dim=-1))
    adapter = SimpleNamespace(
        left_arm_ids=tuple(range(7)), right_arm_ids=tuple(range(7, 14)),
        robot=SimpleNamespace(data=SimpleNamespace(
            soft_joint_pos_limits=torch.cat(limits).unsqueeze(0))),
    )
    return SimpleNamespace(runtime=runtime, adapter=adapter, env_index=0,
                           safety=SafetyProjection(joint_position_margin_rad=.02))


@pytest.mark.parametrize("wire", [True, False])
def test_wire_arm_limits_share_strict_safety_boundary(monkeypatch, wire):
    module = _load_wrapper(monkeypatch)
    wrapper = object.__new__(module.M1DualPandaO6BimanualWrapper)
    wrapper._wire_geometry_provider = object() if wire else None
    lane = _lane()
    before = lane.runtime._arm_inputs(_snapshot())
    wrapper._configure_wire_arm_limits(lane)
    after = lane.runtime._arm_inputs(_snapshot())
    for index, (old, new) in enumerate(zip(before, after)):
        if not wire:
            torch.testing.assert_close(new.q_min, old.q_min)
            torch.testing.assert_close(new.q_max, old.q_max)
            continue
        limits = lane.adapter.robot.data.soft_joint_pos_limits[0, index*7:(index+1)*7]
        torch.testing.assert_close(new.q_min, limits[:, 0] + .02)
        torch.testing.assert_close(new.q_max, limits[:, 1] - .02)
        assert new.q_min[1].item() == pytest.approx(-1.56652)
        assert torch.all(new.q_min >= old.q_min)
        assert torch.all(new.q_max <= old.q_max)


def test_wire_planner_rejects_observed_unrecoverable_q2_state(monkeypatch):
    module = _load_wrapper(monkeypatch)
    wrapper = object.__new__(module.M1DualPandaO6BimanualWrapper)
    wrapper._wire_geometry_provider = object()
    lane = _lane()
    snapshot = _snapshot()
    q = torch.tensor([1.7228876, -1.56586027, -1.2487347, -2.507447,
                      -2.2117004, 1.8649896, 2.3312364], dtype=torch.float64)
    qd = torch.tensor([.1832326, -.69238424, .0707437, -.0717354,
                       -.2617144, -.5177861, -.3366131], dtype=torch.float64)
    snapshot = replace(snapshot, left_arm=replace(snapshot.left_arm, q=q, qd=qd))
    old, _ = lane.runtime._arm_inputs(snapshot)
    assert LinearizedArmMpc().plan(old).diagnostics.feasible
    wrapper._configure_wire_arm_limits(lane)
    corrected, _ = lane.runtime._arm_inputs(snapshot)
    # Even maximal braking cannot remain in the safety domain at the next
    # Arm MPC node. The old hard-limit contract incorrectly accepts it.
    next_q2 = q[1] + .02 * qd[1] + .5 * .02**2 * corrected.qdd_max[1]
    assert next_q2 < corrected.q_min[1]
    assert not LinearizedArmMpc().plan(corrected).diagnostics.feasible


def test_wire_limit_binding_preserves_custom_tighter_arm_provider(monkeypatch):
    module = _load_wrapper(monkeypatch)
    wrapper = object.__new__(module.M1DualPandaO6BimanualWrapper)
    wrapper._wire_geometry_provider = object()
    lane = _lane()
    baseline = lane.runtime._arm_inputs(_snapshot())
    custom = tuple(replace(s, q_min=s.q_min + .4, q_max=s.q_max - .4) for s in baseline)
    lane.runtime._arm_input_provider = lambda _snapshot: custom
    wrapper._configure_wire_arm_limits(lane)
    bounded = lane.runtime._arm_inputs(_snapshot())
    for old, new in zip(custom, bounded):
        assert torch.all(new.q_min >= old.q_min)
        assert torch.all(new.q_max <= old.q_max)
        torch.testing.assert_close(new.qdd_max, old.qdd_max)
