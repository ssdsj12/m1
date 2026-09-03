from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualPhase,
    BimanualSnapshot,
    BoxState,
    SideArmState,
    SideHandState,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_mpc import (
    BimanualObjectMpc,
    ObjectMpcCfg,
    ObjectMpcInput,
)


DTYPE = torch.float64


def _arm(palm_y: float) -> SideArmState:
    pose = torch.zeros(6, dtype=DTYPE)
    pose[1] = palm_y
    return SideArmState(
        q=torch.zeros(7, dtype=DTYPE),
        qd=torch.zeros(7, dtype=DTYPE),
        palm_pose_b=pose,
        palm_twist_b=torch.zeros(6, dtype=DTYPE),
        jacobian_b=torch.eye(6, 7, dtype=DTYPE),
        mass_matrix=torch.eye(7, dtype=DTYPE),
        bias=torch.zeros(7, dtype=DTYPE),
    )


def _hand() -> SideHandState:
    return SideHandState(
        q=torch.zeros(6, dtype=DTYPE),
        qd=torch.zeros(6, dtype=DTYPE),
        fingertip_forces_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_jacobian_b=torch.zeros(15, 6, dtype=DTYPE),
        contact_mask=torch.ones(5, dtype=torch.bool),
    )


def _snapshot() -> BimanualSnapshot:
    box_pose = torch.tensor([0.45, 0.0, 0.72, 0.0, 0.0, 0.0], dtype=DTYPE)
    return BimanualSnapshot(
        timestamp_ns=1,
        base_state=torch.zeros(13, dtype=DTYPE),
        m1_q=torch.zeros(16, dtype=DTYPE),
        m1_qd=torch.zeros(16, dtype=DTYPE),
        platform_q_qd=torch.zeros(2, dtype=DTYPE),
        left_arm=_arm(0.12),
        right_arm=_arm(-0.12),
        left_hand=_hand(),
        right_hand=_hand(),
        box=BoxState(
            pose_b=box_pose,
            twist_b=torch.zeros(6, dtype=DTYPE),
            mass=torch.tensor(0.5, dtype=DTYPE),
            inertia_b=torch.diag(torch.tensor([0.003, 0.006, 0.007], dtype=DTYPE)),
            supported=False,
        ),
    )


def _input(**overrides) -> ObjectMpcInput:
    snapshot = overrides.pop("snapshot", _snapshot())
    target = snapshot.box.pose_b.repeat(25, 1)
    values = {
        "snapshot": snapshot,
        "target_box_pose_b": target,
        "phase": BimanualPhase.HOLD,
        "previous_solution": None,
    }
    values.update(overrides)
    return ObjectMpcInput(**values)


def test_default_cfg_freezes_25_hz_one_second_horizon():
    cfg = ObjectMpcCfg()
    assert cfg.dt == pytest.approx(0.04)
    assert cfg.horizon_steps == 25
    assert cfg.horizon_seconds == pytest.approx(1.0)
    assert cfg.friction_coefficient == pytest.approx(0.8)
    assert cfg.per_hand_normal_force_min == pytest.approx(8.0)
    assert cfg.per_hand_normal_force_max == pytest.approx(15.0)


def test_static_box_solution_balances_gravity_and_is_mirrored():
    solution = BimanualObjectMpc().plan(_input())

    assert solution.diagnostics.feasible
    assert not solution.diagnostics.fallback_used
    total_force = solution.left_wrench[0, :3] + solution.right_wrench[0, :3]
    assert total_force[2].item() == pytest.approx(0.5 * 9.81, abs=1.0e-5)
    assert total_force[:2].tolist() == pytest.approx([0.0, 0.0], abs=1.0e-7)
    assert solution.left_palm_pose[0, 1].item() == pytest.approx(
        -solution.right_palm_pose[0, 1].item()
    )
    assert torch.allclose(solution.box_pose, _snapshot().box.pose_b.expand(25, -1), atol=1.0e-7)


def test_palm_targets_preserve_each_mounted_hand_orientation():
    snapshot = _snapshot()
    left_pose = snapshot.left_arm.palm_pose_b.clone()
    right_pose = snapshot.right_arm.palm_pose_b.clone()
    left_pose[3:] = torch.tensor([0.2, -0.3, 0.4], dtype=DTYPE)
    right_pose[3:] = torch.tensor([-0.2, 0.3, -0.4], dtype=DTYPE)
    snapshot = replace(
        snapshot,
        left_arm=replace(snapshot.left_arm, palm_pose_b=left_pose),
        right_arm=replace(snapshot.right_arm, palm_pose_b=right_pose),
    )
    solution = BimanualObjectMpc().plan(_input(snapshot=snapshot))
    assert torch.allclose(solution.left_palm_pose[:, 3:], left_pose[3:].expand(25, -1))
    assert torch.allclose(solution.right_palm_pose[:, 3:], right_pose[3:].expand(25, -1))


def test_wrenches_obey_normal_bounds_and_four_sided_friction_pyramids():
    planner = BimanualObjectMpc()
    solution = planner.plan(_input())
    mu = planner.cfg.friction_coefficient
    left_normal = -solution.left_wrench[:, 1]
    right_normal = solution.right_wrench[:, 1]
    assert torch.all(left_normal >= planner.cfg.per_hand_normal_force_min - 1.0e-7)
    assert torch.all(right_normal >= planner.cfg.per_hand_normal_force_min - 1.0e-7)
    assert torch.all(left_normal <= planner.cfg.per_hand_normal_force_max + 1.0e-7)
    assert torch.all(right_normal <= planner.cfg.per_hand_normal_force_max + 1.0e-7)
    for wrench, normal in (
        (solution.left_wrench, left_normal),
        (solution.right_wrench, right_normal),
    ):
        assert torch.all(wrench[:, 0].abs() <= mu * normal + 1.0e-7)
        assert torch.all(wrench[:, 2].abs() <= mu * normal + 1.0e-7)
    assert solution.diagnostics.force_closure_margin > 0.0


def test_platform_yaw_obeys_position_and_velocity_limits():
    snapshot = replace(
        _snapshot(), platform_q_qd=torch.tensor([0.4, 0.0], dtype=DTYPE)
    )
    solution = BimanualObjectMpc().plan(_input(snapshot=snapshot))
    delta = torch.diff(torch.cat((snapshot.platform_q_qd[:1], solution.platform_yaw)))
    assert torch.all(solution.platform_yaw.abs() <= torch.pi / 2 + 1.0e-8)
    assert torch.all(delta.abs() <= 0.25 * 0.04 + 1.0e-8)


def test_infeasible_target_returns_last_safe_without_lift_progress():
    planner = BimanualObjectMpc()
    safe = planner.plan(_input())
    unreachable = _snapshot().box.pose_b.repeat(25, 1)
    unreachable[:, 2] += 2.0
    fallback = planner.plan(_input(target_box_pose_b=unreachable, previous_solution=safe))

    assert not fallback.diagnostics.feasible
    assert fallback.diagnostics.fallback_used
    assert fallback.diagnostics.fallback_reason == "target_unreachable"
    assert torch.equal(fallback.left_palm_pose, safe.left_palm_pose)
    assert torch.equal(fallback.right_palm_pose, safe.right_palm_pose)
    assert torch.equal(fallback.box_pose, safe.box_pose)


def test_input_rejects_bad_horizon_dtype_and_nonfinite_target():
    with pytest.raises(ValueError, match="shape"):
        _input(target_box_pose_b=torch.zeros(24, 6, dtype=DTYPE))
    with pytest.raises(TypeError, match="float64"):
        _input(target_box_pose_b=torch.zeros(25, 6, dtype=torch.float32))
    target = torch.zeros(25, 6, dtype=DTYPE)
    target[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _input(target_box_pose_b=target)


def test_repeated_plans_are_deterministic():
    left = BimanualObjectMpc().plan(_input())
    right = BimanualObjectMpc().plan(_input())
    assert torch.equal(left.left_wrench, right.left_wrench)
    assert torch.equal(left.right_wrench, right.right_wrench)
    assert torch.equal(left.platform_yaw, right.platform_yaw)
    assert left.diagnostics == right.diagnostics
