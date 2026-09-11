import math

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import (
    compose_orientation_horizon,
    interpolate_orientation,
    matrix_to_rotvec,
    rotvec_to_matrix,
    spatial_angular_velocity,
    spatial_orientation_error,
)

DTYPE = torch.float64


@pytest.mark.parametrize('angle', [0., 1.e-10, math.pi / 2, math.pi - 1.e-8, math.pi, -math.pi])
def test_so3_roundtrip(angle):
    axis = torch.tensor([1., -2., 3.], dtype=DTYPE)
    axis /= axis.norm()
    rotation = rotvec_to_matrix(axis * angle)
    assert torch.allclose(rotation, torch.matrix_exp(torch.tensor(
        [[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
         [-axis[1], axis[0], 0.]], dtype=DTYPE) * angle), atol=1.e-10, rtol=0.)
    assert torch.allclose(rotvec_to_matrix(matrix_to_rotvec(rotation)), rotation,
                          atol=1.e-8, rtol=0.)


def test_spatial_orientation_error_uses_group_relative_rotation():
    measured = torch.tensor([1.237889, 1.983522, .331056], dtype=DTYPE)
    target = torch.tensor([1.19, 1.91, .49], dtype=DTYPE)
    actual = spatial_orientation_error(target, measured)
    expected = matrix_to_rotvec(
        rotvec_to_matrix(target) @ rotvec_to_matrix(measured).T
    )
    assert torch.allclose(actual, expected, atol=1.e-12, rtol=0.)
    assert not torch.allclose(actual, target - measured, atol=1.e-3, rtol=0.)
    assert torch.allclose(
        spatial_orientation_error(measured, measured),
        torch.zeros(3, dtype=DTYPE),
        atol=1.e-12,
        rtol=0.,
    )


def test_so3_interpolation_preserves_nonidentity_endpoints_and_short_path():
    axis = torch.tensor([1., -2., .5], dtype=DTYPE)
    axis /= axis.norm()
    start = axis * (math.pi - 1.e-4)
    end = -axis * (math.pi - 2.e-4)
    at_start = interpolate_orientation(start, end, 0.)
    halfway = interpolate_orientation(start, end, .5)
    at_end = interpolate_orientation(start, end, 1.)
    assert torch.allclose(rotvec_to_matrix(at_start), rotvec_to_matrix(start), atol=1.e-10)
    assert torch.allclose(rotvec_to_matrix(at_end), rotvec_to_matrix(end), atol=1.e-10)
    full_error = spatial_orientation_error(end, start)
    half_error = spatial_orientation_error(halfway, start)
    assert torch.linalg.vector_norm(full_error).item() < 4.e-4
    assert torch.allclose(half_error, .5 * full_error, atol=1.e-8, rtol=0.)


def test_spatial_angular_velocity_matches_relative_rotation_log():
    previous = torch.tensor([.7, -.3, .2], dtype=DTYPE)
    target_error = torch.tensor([.02, -.03, .01], dtype=DTYPE)
    next_rotation = rotvec_to_matrix(target_error) @ rotvec_to_matrix(previous)
    following = matrix_to_rotvec(next_rotation)
    velocity = spatial_angular_velocity(previous, following, .02)
    assert torch.allclose(velocity, target_error / .02, atol=1.e-10, rtol=0.)


def test_compose_one_and_three_axis_bases():
    entry = torch.tensor([.4, -.2, .1], dtype=DTYPE)
    for basis in (torch.eye(3, dtype=DTYPE)[:1], torch.eye(3, dtype=DTYPE)):
        coefficients = torch.full((25, len(basis)), .15, dtype=DTYPE)
        actual = compose_orientation_horizon(entry, basis, coefficients)
        expected = rotvec_to_matrix(entry) @ rotvec_to_matrix(coefficients[0] @ basis)
        assert actual.shape == (25, 3)
        assert actual.dtype == DTYPE and actual.device.type == 'cpu'
        assert torch.allclose(rotvec_to_matrix(actual[0]), expected, atol=1.e-10)


@pytest.mark.parametrize('bad', [torch.zeros(3), torch.zeros(2, dtype=DTYPE),
    torch.full((3,), float('nan'), dtype=DTYPE), torch.zeros(3, device='meta', dtype=DTYPE)])
def test_so3_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        rotvec_to_matrix(bad)


@pytest.mark.parametrize('basis', [torch.zeros(1, 3, dtype=DTYPE),
    torch.ones(2, 3, dtype=DTYPE), torch.eye(3, dtype=DTYPE) * 2])
def test_compose_rejects_invalid_basis(basis):
    with pytest.raises(ValueError):
        compose_orientation_horizon(torch.zeros(3, dtype=DTYPE), basis,
                                    torch.zeros(25, len(basis), dtype=DTYPE))


def test_nearest_point_on_rotated_box():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import nearest_point_on_oriented_box
    box = torch.tensor([1., 2., 3., 0., 0., math.pi / 2], dtype=DTYPE)
    point = torch.tensor([1., 1., 3.], dtype=DTYPE)
    actual = nearest_point_on_oriented_box(point, box,
                                          torch.tensor([.1, .09, .05], dtype=DTYPE))
    assert torch.allclose(actual, torch.tensor([1., 1.9, 3.], dtype=DTYPE))


def test_lead_margin_and_digit_tie():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import evaluate_lead_margin
    ray = torch.tensor([0., 1., 0.], dtype=DTYPE)
    tips = torch.tensor([[0., .15, 0.]] * 5, dtype=DTYPE)
    housing = torch.tensor([[0., .04, 0.]] * 8, dtype=DTYPE)
    result = evaluate_lead_margin(torch.eye(3, dtype=DTYPE), ray, tips, housing)
    assert result.lead_margin_m == pytest.approx(.11)
    assert result.leading_digit_index == 0


def test_axis_search_improves_lead_and_repeats():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import select_single_axis
    args = (torch.eye(3, dtype=DTYPE), torch.tensor([0., 1., 0.], dtype=DTYPE),
            torch.tensor([[0., 0., .2]] * 5, dtype=DTYPE), torch.zeros(8, 3, dtype=DTYPE))
    first = select_single_axis(*args)
    second = select_single_axis(*args)
    assert torch.equal(first.axis_local, torch.tensor([1., 0., 0.], dtype=DTYPE))
    assert first.target_angle_rad == pytest.approx(-.35)
    assert first.geometry.lead_margin_m == pytest.approx(.2 * math.sin(.35))
    assert first.target_angle_rad == second.target_angle_rad
    assert torch.equal(first.axis_local, second.axis_local)


def test_zero_angle_and_axis_ties_choose_zero_x():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import select_single_axis
    result = select_single_axis(torch.eye(3, dtype=DTYPE),
        torch.tensor([0., 1., 0.], dtype=DTYPE), torch.zeros(5, 3, dtype=DTYPE),
        torch.zeros(8, 3, dtype=DTYPE))
    assert result.target_angle_rad == 0.
    assert torch.equal(result.axis_local, torch.tensor([1., 0., 0.], dtype=DTYPE))


def test_runtime_housing_support_matches_verified_manifest():
    import json
    from pathlib import Path
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import HOUSING_SUPPORT_POINTS_LOCAL_M
    manifest = json.loads((Path(__file__).resolve().parents[1] /
        'assets/m1_dual_panda_o6/asset_manifest.json').read_text())
    assert [list(p) for p in HOUSING_SUPPORT_POINTS_LOCAL_M] == manifest[
        'right_palm_housing_support']['support_points_local_m']


def test_qp_bounds_rates_and_repeatability():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import PalmOrientationMpcCfg, build_palm_orientation_qp
    from go2_pvcnn.control.m1_panda_coordination.qp_backend import solve_reference_qp
    cfg = PalmOrientationMpcCfg()
    problem = build_palm_orientation_qp(.1, .35, cfg)
    result = solve_reference_qp(problem, tolerance=cfg.qp_tolerance, max_iterations=cfg.qp_max_iterations)
    again = solve_reference_qp(problem, tolerance=cfg.qp_tolerance, max_iterations=cfg.qp_max_iterations)
    assert result.success
    assert torch.equal(result.solution, again.solution)
    assert bool((result.solution.abs() <= .35 + 1.e-10).all())
    deltas = torch.diff(torch.cat((torch.tensor([.1], dtype=DTYPE), result.solution)))
    assert bool((deltas.abs() <= .014 + 1.e-10).all())


def test_qp_constant_target_has_no_artificial_boundary_acceleration():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import PalmOrientationMpcCfg, build_palm_orientation_qp
    cfg = PalmOrientationMpcCfg()
    qp = build_palm_orientation_qp(.2, .2, cfg)
    # A constant horizon has zero slew and acceleration; only regularization remains.
    gradient = qp.hessian @ torch.full((25,), .2, dtype=DTYPE) + qp.gradient
    assert torch.allclose(gradient, torch.full((25,), 2 * cfg.regularization * .2, dtype=DTYPE), atol=1.e-12)


def _orientation_input():
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import PalmOrientationInput
    from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
    return PalmOrientationInput(
        torch.tensor([0., -.2, 0., 0., 0., 0.], dtype=DTYPE),
        torch.tensor([[0., -.2, .2]] * 5, dtype=DTYPE),
        torch.zeros(6, dtype=DTYPE), torch.zeros(5, dtype=torch.bool), BimanualPhase.APPROACH)


def test_controller_replay_contact_hold_and_rejection():
    from dataclasses import replace
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import RightPalmOrientationMpc
    planner = RightPalmOrientationMpc()
    sample = _orientation_input()
    first = planner.plan(sample)
    assert first.diagnostics.feasible
    assert first.orientation_rotvec_b.shape == (25, 3)
    assert abs(float(first.angle_rad[0])) <= .014 + 1.e-10
    planner.reset()
    replay = planner.plan(sample)
    assert torch.equal(replay.angle_rad, first.angle_rad)
    bad = replace(sample, fingertip_positions_b=torch.full((5, 3), float('nan'), dtype=DTYPE))
    rejected = planner.plan(bad)
    assert not rejected.diagnostics.feasible
    assert torch.equal(rejected.angle_rad, first.angle_rad)
    # Returned buffers must not alias internal last-safe storage.
    rejected.angle_rad.fill_(99.)
    assert torch.equal(planner.plan(bad).angle_rad, first.angle_rad)
    pose = sample.palm_pose_b.clone()
    pose[3:] = first.orientation_rotvec_b[0]
    latched = planner.plan(replace(sample, palm_pose_b=pose, contact_mask=torch.ones(5, dtype=torch.bool)))
    assert latched.diagnostics.feasible
    assert torch.allclose(latched.angle_rad, first.angle_rad[0].expand(25), atol=1.e-10)
    assert torch.equal(planner.plan(sample).angle_rad, latched.angle_rad)


def test_first_cycle_rejection_has_finite_measured_hold():
    from dataclasses import replace
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import RightPalmOrientationMpc
    sample = _orientation_input()
    sample = replace(sample, fingertip_positions_b=torch.zeros(5, 3))
    result = RightPalmOrientationMpc().plan(sample)
    assert not result.diagnostics.feasible
    assert torch.equal(result.orientation_rotvec_b, sample.palm_pose_b[3:].repeat(25, 1))


def test_failed_qp_does_not_advance_axis_angle_or_safe_state(monkeypatch):
    from dataclasses import replace
    from go2_pvcnn.control.m1_bimanual_coordination import palm_orientation_mpc as module
    planner = module.RightPalmOrientationMpc()
    sample = _orientation_input()
    good = planner.plan(sample)
    angle = planner._committed_angle
    axis = planner._axis_local.clone()
    original = module.solve_reference_qp
    monkeypatch.setattr(module, 'solve_reference_qp',
                        lambda *a, **k: replace(original(*a, **k), success=False))
    rejected = planner.plan(sample)
    assert not rejected.diagnostics.feasible
    assert rejected.diagnostics.fallback_reason == 'qp_infeasible'
    assert planner._committed_angle == angle
    assert torch.equal(planner._axis_local, axis)
    assert not planner._contact_latched
    assert torch.equal(planner._last_safe.angle_rad, good.angle_rad)


@pytest.mark.parametrize('kwargs', [{'dt': float('nan')}, {'active_basis_dim': 3},
    {'candidate_count': 30}, {'qp_max_iterations': True}, {'horizon_steps': 20}])
def test_invalid_configuration_is_rejected(kwargs):
    from go2_pvcnn.control.m1_bimanual_coordination.palm_orientation_mpc import PalmOrientationMpcCfg
    with pytest.raises(ValueError):
        PalmOrientationMpcCfg(**kwargs)
