from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.object_mpc import (
    BimanualObjectMpc, ObjectMpcCfg, ObjectMpcInput, build_object_qp,
)
from go2_pvcnn.control.m1_bimanual_coordination.reduced_cosserat import ReducedCosseratPrediction
from tests.test_m1_bimanual_object_mpc import _input


def _prediction(speed=0.0):
    endpoints = np.tile([[.45, .12, .72], [.45, -.12, .72]], (25, 1, 1))
    endpoints[:, :, 2] += np.arange(1, 26)[:, None] * .04 * speed
    velocities = np.zeros_like(endpoints)
    velocities[:, :, 2] = speed
    positions = np.stack((endpoints[:, 0], endpoints.mean(1), endpoints[:, 1]), axis=1)
    return ReducedCosseratPrediction(
        positions=positions, tangents=np.tile([0., -1., 0.], (25, 3, 1)),
        endpoint_positions=endpoints, endpoint_velocities=velocities,
        endpoint_separation_m=np.full(25, .24), axial_stretch=np.zeros(25),
        bending_proxy=np.zeros(25), left_slip_m=np.zeros(25), right_slip_m=np.zeros(25),
        friction_feasible=np.ones(25, bool), slip_feasible=np.ones(25, bool),
        feasible=np.ones(25, bool), failure_reasons=(None,) * 25,
    )


def _wire(prediction=None, **kwargs):
    assert 'wire_prediction' in {field.name for field in fields(ObjectMpcInput)}, 'wire prediction field missing'
    prediction = _prediction() if prediction is None else prediction
    sample = _input(hold_measured_palm_orientation=True, **kwargs)
    arms = {}
    for side, index in [('left', 0), ('right', 1)]:
        arm = getattr(sample.snapshot, side + '_arm')
        pose = arm.palm_pose_b.clone()
        pose[:3] = torch.tensor(prediction.endpoint_positions[0, index] - .04 * prediction.endpoint_velocities[0, index])
        arms[side + '_arm'] = replace(arm, palm_pose_b=pose)
    sample = replace(sample, snapshot=replace(sample.snapshot, **arms))
    target = sample.target_box_pose_b.clone()
    target[:, :3] = torch.tensor(prediction.endpoint_positions.mean(1))
    return replace(sample, target_box_pose_b=target, target_object_pose_b=target,
                   wire_prediction=prediction)


def test_wire_endpoint_tracking_force_allocation_and_finite_hard_constraints():
    sample = _wire()
    problem = build_object_qp(sample, ObjectMpcCfg())
    assert all(torch.isfinite(getattr(problem, field.name)).all() for field in fields(problem))
    result = BimanualObjectMpc().plan(sample)
    assert result.diagnostics.feasible
    for side, index in [('left', 0), ('right', 1)]:
        assert torch.allclose(getattr(result, side + '_palm_pose')[:, :3],
                              torch.tensor(sample.wire_prediction.endpoint_positions[:, index]), atol=.002)
    assert torch.allclose(result.left_wrench[:, 2], result.right_wrench[:, 2], atol=1e-7)
    assert (result.left_wrench[:, 2] + result.right_wrench[:, 2]).mean().item() == pytest.approx(.5 * 9.81, abs=.02)
    assert result.diagnostics.wire_stretch_max == pytest.approx(0, abs=1e-6)
    assert result.diagnostics.wire_slip_max_m <= .02
    assert result.diagnostics.wire_friction_feasible
    assert (-result.left_wrench[:, 1] >= 8 - 1e-7).all()
    assert (result.right_wrench[:, 1] >= 8 - 1e-7).all()
    controls = torch.cat((result.left_wrench, result.right_wrench, result.platform_yaw[:, None]), dim=1).flatten()
    assert (problem.equality_matrix @ controls - problem.equality_rhs).abs().max() <= 1e-7
    assert (problem.inequality_matrix @ controls - problem.inequality_upper).max() <= 1e-7


@pytest.mark.parametrize('field,reason', [
    ('friction_feasible', 'wire_friction_infeasible'),
    ('slip_feasible', 'wire_slip_infeasible'), ('feasible', 'wire_dynamics_infeasible'),
])
def test_entire_prediction_veto_is_atomic_and_deterministic(field, reason):
    planner = BimanualObjectMpc()
    sample = _wire()
    safe = planner.plan(sample)
    assert safe.diagnostics.feasible
    flags = np.ones(25, bool)
    flags[-1] = False
    bad = replace(sample, wire_prediction=replace(sample.wire_prediction, **{field: flags}))
    for _ in range(2):
        result = planner.plan(bad)
        assert not result.diagnostics.feasible
        assert result.diagnostics.fallback_reason == reason
        assert not result.diagnostics.wire_lift_feasible
        for name in ('box_pose', 'box_twist', 'left_palm_pose', 'right_palm_pose', 'left_wrench', 'right_wrench'):
            assert torch.equal(getattr(result, name), getattr(safe, name))
            assert torch.equal(getattr(planner._last_safe, name), getattr(safe, name))


def test_wire_lift_tracks_synchronized_endpoint_velocity():
    sample = _wire(_prediction(.02), phase=BimanualPhase.LIFT)
    result = BimanualObjectMpc().plan(sample)
    assert result.diagnostics.feasible
    assert result.diagnostics.wire_lift_feasible
    assert result.box_pose[-1, 2] > sample.snapshot.box.pose_b[2] + .015
    assert torch.allclose(result.left_palm_pose[:, 2], result.right_palm_pose[:, 2], atol=1e-8)
    assert torch.allclose(result.box_twist[:, 2], torch.full((25,), .02, dtype=torch.float64), atol=.006)


@pytest.mark.parametrize('kind,reason', [
    ('stretch', 'wire_stretch_infeasible'), ('slip', 'wire_slip_infeasible'),
    ('bend', 'wire_bending_infeasible'), ('separation', 'wire_separation_infeasible'),
    ('contact', 'wire_contact_infeasible'), ('nonfinite', 'wire_nonfinite_prediction'),
])
def test_wire_hard_veto_cannot_be_bypassed_by_feasible_flags(kind, reason):
    sample = _wire()
    prediction = sample.wire_prediction
    if kind == 'contact':
        sample = replace(sample, snapshot=replace(sample.snapshot, left_hand=replace(
            sample.snapshot.left_hand, contact_mask=torch.zeros(5, dtype=torch.bool))))
    else:
        name, value = {'stretch': ('axial_stretch', .5), 'slip': ('left_slip_m', .03),
                       'bend': ('bending_proxy', 21.), 'separation': ('endpoint_separation_m', 2.),
                       'nonfinite': ('endpoint_positions', np.nan)}[kind]
        corrupted = np.array(getattr(prediction, name), copy=True)
        corrupted[-1] = value
        # Simulate a corrupt third-party adapter after dataclass construction.
        object.__setattr__(prediction, name, corrupted)
    result = BimanualObjectMpc().plan(sample)
    assert not result.diagnostics.feasible
    assert result.diagnostics.fallback_reason == reason
    assert torch.equal(result.box_twist, torch.zeros_like(result.box_twist))
    assert torch.equal(result.left_palm_pose, sample.snapshot.left_arm.palm_pose_b.repeat(25, 1))
    if kind == 'stretch':
        assert result.diagnostics.wire_stretch_max == .5
    if kind == 'slip':
        assert result.diagnostics.wire_slip_max_m == .03


def test_wire_prediction_horizon_is_checked_at_boundary():
    sample = _wire()
    prediction = _prediction()
    object.__setattr__(prediction, 'endpoint_positions', prediction.endpoint_positions[:-1])
    with pytest.raises(ValueError, match='horizon|shape'):
        replace(sample, wire_prediction=prediction)


def test_omitted_prediction_preserves_rigid_qp_and_solution_bitwise():
    sample = _input()
    assert 'wire_prediction' in {field.name for field in fields(ObjectMpcInput)}, 'wire prediction field missing'
    explicit = replace(sample, wire_prediction=None)
    a, b = build_object_qp(sample, ObjectMpcCfg()), build_object_qp(explicit, ObjectMpcCfg())
    assert all(torch.equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    a, b = BimanualObjectMpc().plan(sample), BimanualObjectMpc().plan(explicit)
    assert torch.equal(a.left_wrench, b.left_wrench)
    assert torch.equal(a.box_pose, b.box_pose)


def test_solver_infeasibility_preserves_safe_wrenches_and_ignores_unsafe_previous():
    planner = BimanualObjectMpc()
    sample = _wire()
    safe = planner.plan(sample)
    planner.cfg = replace(planner.cfg, friction_coefficient=.001)
    previous = replace(safe, left_wrench=torch.full_like(safe.left_wrench, 999),
                       diagnostics=replace(safe.diagnostics, feasible=False))
    rejected = planner.plan(replace(sample, previous_solution=previous))
    assert rejected.diagnostics.fallback_reason == 'wire_qp_infeasible'
    assert torch.equal(rejected.left_wrench, safe.left_wrench)
    assert torch.equal(planner._last_safe.right_wrench, safe.right_wrench)


def test_asymmetric_lift_reference_synchronizes_actual_palm_motion():
    prediction = _prediction(.01)
    endpoints = prediction.endpoint_positions.copy()
    endpoints[:, 0, 2] += np.arange(1, 26) * .04 * .005
    velocities = prediction.endpoint_velocities.copy()
    velocities[:, 0, 2] = .015
    positions = np.stack((endpoints[:, 0], endpoints.mean(1), endpoints[:, 1]), axis=1)
    prediction = replace(prediction, endpoint_positions=endpoints, endpoint_velocities=velocities,
                         positions=positions, endpoint_separation_m=np.linalg.norm(endpoints[:, 0]-endpoints[:, 1], axis=1))
    result = BimanualObjectMpc().plan(_wire(prediction, phase=BimanualPhase.LIFT))
    assert result.diagnostics.feasible
    left_dz = torch.diff(result.left_palm_pose[:, 2])
    right_dz = torch.diff(result.right_palm_pose[:, 2])
    assert torch.allclose(left_dz, right_dz, atol=1e-7)
    assert result.box_twist[:, 2].max() <= .01 + 1e-7


def test_prediction_geometry_far_from_current_palms_is_rejected():
    sample = _wire()
    # Object target remains reachable, but a bad endpoint adapter jumps 2 m.
    prediction = sample.wire_prediction
    endpoints = prediction.endpoint_positions.copy()
    endpoints[:, :, 0] += 2
    positions = prediction.positions.copy()
    positions[:, :, 0] += 2
    result = BimanualObjectMpc().plan(replace(sample, wire_prediction=replace(
        prediction, endpoint_positions=endpoints, positions=positions)))
    assert not result.diagnostics.feasible
    assert result.diagnostics.fallback_reason == 'wire_endpoint_unreachable'


@pytest.mark.parametrize('field,value', [('max_slip_m', float('nan')),
                                       ('max_axial_stretch', 0),
                                       ('max_endpoint_separation_m', .0001)])
def test_wire_limits_reject_invalid_values(field, value):
    from go2_pvcnn.control.m1_bimanual_coordination.contracts import WireMpcLimits
    with pytest.raises(ValueError):
        replace(WireMpcLimits(), **{field: value})


def test_accelerating_wire_requires_inertial_force_above_gravity():
    prediction = _prediction()
    elapsed = np.arange(1, 26) * .04
    endpoints = prediction.endpoint_positions.copy()
    endpoints[:, :, 2] += .5 * .02 * elapsed[:, None] ** 2
    velocities = prediction.endpoint_velocities.copy()
    velocities[:, :, 2] = .02 * elapsed[:, None]
    positions = np.stack((endpoints[:, 0], endpoints.mean(1), endpoints[:, 1]), axis=1)
    prediction = replace(prediction, endpoint_positions=endpoints, endpoint_velocities=velocities, positions=positions)
    result = BimanualObjectMpc().plan(_wire(prediction, phase=BimanualPhase.LIFT))
    assert result.diagnostics.feasible
    force = result.left_wrench[:, 2] + result.right_wrench[:, 2]
    assert force.mean().item() == pytest.approx(.5 * (9.81 + .02), abs=.002)


def test_extreme_finite_velocity_rejects_before_nonfinite_qp_reaches_backend():
    sample = _wire()
    prediction = replace(sample.wire_prediction, endpoint_velocities=np.full((25, 2, 3), 1e308))
    result = BimanualObjectMpc().plan(replace(sample, wire_prediction=prediction))
    assert not result.diagnostics.feasible
    assert result.diagnostics.fallback_reason == 'wire_nonfinite_qp'


def test_malformed_adapter_output_does_not_break_fallback_diagnostics():
    sample = _wire()
    object.__setattr__(sample.wire_prediction, 'right_slip_m', np.zeros(24))
    result = BimanualObjectMpc().plan(sample)
    assert not result.diagnostics.feasible
    assert result.diagnostics.fallback_reason == 'wire_invalid_prediction'
