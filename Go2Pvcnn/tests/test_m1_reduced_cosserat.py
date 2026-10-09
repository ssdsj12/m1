from __future__ import annotations

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import DloCenterlineState
from go2_pvcnn.control.m1_bimanual_coordination.reduced_cosserat import (
    ReducedCosseratPrediction,
    ReducedCosseratPredictor,
)


def _centerline() -> DloCenterlineState:
    return DloCenterlineState(
        np.array(
            [[0.40, 0.00, 1.20], [0.50, 0.00, 1.20], [0.60, 0.00, 1.20]],
            dtype=np.float64,
        ),
        radius_m=0.004,
    )


def _boundary(position: tuple[float, float, float], velocity=(0.0, 0.0, 0.0)):
    return {"pose": np.asarray(position, dtype=np.float64), "twist": np.asarray(velocity, dtype=np.float64)}


def _wrenches(left=(0.0, 0.0, 1.0), right=(0.0, 0.0, 1.0)):
    return np.asarray([left, right], dtype=np.float64)


def test_reset_state_is_finite_cpu_float64_and_has_centerline_samples():
    predictor = ReducedCosseratPredictor()
    state = predictor.reset(_centerline(), radius_m=0.004, material={"friction_coefficient": 0.7}, dt=0.01)

    assert state.positions.shape == (3, 3)
    assert state.tangents.shape == (3, 3)
    assert state.positions.dtype == np.float64
    assert state.endpoint_separation_m == pytest.approx(0.2)
    assert state.axial_stretch == pytest.approx(0.0)
    assert state.feasible
    assert state.finite
    assert all(np.isfinite(value).all() for value in (state.positions, state.tangents))


def test_step_is_deterministic_and_reports_separation_and_stretch():
    kwargs = dict(radius_m=0.004, material={"friction_coefficient": 0.7}, dt=0.01)
    first = ReducedCosseratPredictor()
    second = ReducedCosseratPredictor()
    first.reset(_centerline(), **kwargs)
    second.reset(_centerline(), **kwargs)
    left = _boundary((0.39, 0.0, 1.2), velocity=(0.1, 0.0, 0.0))
    right = _boundary((0.63, 0.0, 1.2), velocity=(0.1, 0.0, 0.0))
    result_a = first.step(left, right, _wrenches())
    result_b = second.step(left, right, _wrenches())

    np.testing.assert_allclose(result_a.positions, result_b.positions)
    assert result_a.endpoint_separation_m == pytest.approx(0.24)
    assert result_a.axial_stretch == pytest.approx(0.2)
    assert result_a.endpoint_velocities.shape == (2, 3)
    assert result_a.timestamp_s == pytest.approx(0.01)


def test_step_transports_tangent_samples_and_checks_friction_and_slip():
    predictor = ReducedCosseratPredictor(max_slip_m=0.02, friction_coefficient=0.5)
    predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    state = predictor.step(
        _boundary((0.40, 0.00, 1.20)),
        _boundary((0.60, 0.02, 1.20)),
        _wrenches((0.30, 0.0, 1.0), (-0.30, 0.0, 1.0)),
    )

    assert state.tangents.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(state.tangents, axis=1), 1.0)
    assert state.friction_feasible
    assert state.slip_feasible
    assert state.left_slip_m == pytest.approx(0.0)
    assert state.right_slip_m == pytest.approx(0.02)


def test_predict_has_horizon_shapes_and_does_not_mutate_last_state():
    predictor = ReducedCosseratPredictor()
    predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    before = predictor.state
    prediction = predictor.predict(
        4,
        [_boundary((0.40, 0.0, 1.2)), _boundary((0.41, 0.0, 1.2))],
        [_boundary((0.60, 0.0, 1.2)), _boundary((0.61, 0.0, 1.2))],
        [_wrenches(), _wrenches()],
    )

    assert prediction.positions.shape == (4, 3, 3)
    assert prediction.endpoint_positions.shape == (4, 2, 3)
    assert prediction.tangents.shape == (4, 3, 3)
    assert prediction.feasible.shape == (4,)
    np.testing.assert_allclose(predictor.state.positions, before.positions)
    assert predictor.state.timestamp_s == before.timestamp_s


def test_nonfinite_rejection_preserves_last_accepted_state():
    predictor = ReducedCosseratPredictor()
    accepted = predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    with pytest.raises(ValueError, match="finite"):
        predictor.step(
            _boundary((float("nan"), 0.0, 1.2)),
            _boundary((0.60, 0.0, 1.2)),
            _wrenches(),
        )
    np.testing.assert_allclose(predictor.state.positions, accepted.positions)
    assert predictor.state.timestamp_s == accepted.timestamp_s


def test_infeasible_friction_is_reported_without_overwriting_safe_state():
    predictor = ReducedCosseratPredictor(friction_coefficient=0.2)
    safe = predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    candidate = predictor.step(
        _boundary((0.40, 0.0, 1.2)),
        _boundary((0.60, 0.0, 1.2)),
        _wrenches((1.0, 0.0, 0.1), (-1.0, 0.0, 0.1)),
    )

    assert not candidate.friction_feasible
    assert not candidate.feasible
    np.testing.assert_allclose(predictor.state.positions, safe.positions)
    assert predictor.state.timestamp_s == safe.timestamp_s


def test_predict_rejection_restores_last_state_after_partial_horizon():
    predictor = ReducedCosseratPredictor()
    safe = predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(
            2,
            [_boundary((0.40, 0.0, 1.2)), _boundary((float("nan"), 0.0, 1.2))],
            [_boundary((0.60, 0.0, 1.2)), _boundary((0.60, 0.0, 1.2))],
            [_wrenches(), _wrenches()],
        )
    np.testing.assert_allclose(predictor.state.positions, safe.positions)
    assert predictor.state.timestamp_s == safe.timestamp_s


@pytest.mark.parametrize("invalid_material", [
    {"damping": float("nan")},
    {"effective_mass_kg": 0.0},
    {"max_slip_m": -1.0},
])
def test_reset_rejects_invalid_material_atomically(invalid_material):
    predictor = ReducedCosseratPredictor()
    safe = predictor.reset(_centerline(), radius_m=0.004, material={"damping": 0.25}, dt=0.01)
    before_material = {
        name: getattr(predictor, name)
        for name in ("effective_mass_kg", "damping", "friction_coefficient", "max_axial_stretch", "max_bending_proxy", "max_slip_m", "normal_force_min_n")
    }

    with pytest.raises(ValueError):
        predictor.reset(
            _centerline(), radius_m=0.008,
            material={"effective_mass_kg": 0.1, **invalid_material}, dt=0.02,
        )

    assert predictor.state is safe
    assert predictor.state.timestamp_s == safe.timestamp_s
    for name, value in before_material.items():
        assert getattr(predictor, name) == value
    next_state = predictor.step(_boundary((0.40, 0.0, 1.2)), _boundary((0.60, 0.0, 1.2)), _wrenches())
    assert next_state.timestamp_s == pytest.approx(0.01)
    assert next_state.radius_m == safe.radius_m


@pytest.mark.parametrize("later_offset", [0.1, 0.0])
def test_predict_does_not_advance_from_rejected_candidate(later_offset):
    predictor = ReducedCosseratPredictor(max_slip_m=0.05)
    safe = predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    prediction = predictor.predict(
        2,
        [_boundary((0.50, 0.0, 1.2)), _boundary((0.40 + later_offset, 0.0, 1.2))],
        [_boundary((0.70, 0.0, 1.2)), _boundary((0.60 + later_offset, 0.0, 1.2))],
        [_wrenches(), _wrenches()],
    )

    assert not prediction.feasible[0]
    assert not prediction.feasible[1]
    assert prediction.failure_reasons[0] == "slip"
    assert prediction.failure_reasons[1] == "slip"
    np.testing.assert_allclose(prediction.positions[1], prediction.positions[0])
    assert predictor.state is safe


def test_predict_accepts_sequence_of_wrench_mappings():
    predictor = ReducedCosseratPredictor()
    predictor.reset(_centerline(), radius_m=0.004, material={}, dt=0.01)
    prediction = predictor.predict(
        1,
        [_boundary((0.40, 0.0, 1.2))],
        [_boundary((0.60, 0.0, 1.2))],
        [[{"wrench": (0.0, 0.0, 1.0)}, {"wrench": (0.0, 0.0, 1.0)}]],
    )
    assert prediction.feasible.tolist() == [True]


@pytest.mark.parametrize("field", ["positions", "tangents", "endpoint_positions", "endpoint_velocities"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_prediction_rejects_nonfinite_geometry(field, nonfinite):
    finite = np.zeros((1, 2, 3), dtype=np.float64)
    geometry = {name: finite.copy() for name in ("positions", "tangents", "endpoint_positions", "endpoint_velocities")}
    geometry[field][0, 1, 0] = nonfinite
    with pytest.raises(ValueError, match="finite"):
        ReducedCosseratPrediction(
            **geometry,
            endpoint_separation_m=np.zeros(1),
            axial_stretch=np.zeros(1),
            bending_proxy=np.zeros(1),
            left_slip_m=np.zeros(1),
            right_slip_m=np.zeros(1),
            friction_feasible=np.ones(1, dtype=bool),
            slip_feasible=np.ones(1, dtype=bool),
            feasible=np.ones(1, dtype=bool),
            failure_reasons=(None,),
        )


def test_reset_derived_overflow_preserves_every_predictor_field():
    predictor = ReducedCosseratPredictor()
    safe = predictor.reset(_centerline(), material={"damping": 0.25})
    before = vars(predictor).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            predictor.reset(
                np.array([[0., 0., 0.], [1.e200, 0., 0.], [2.e200, 0., 0.]]),
                radius_m=0.008, dt=0.02,
                material={"effective_mass_kg": 0.1, "damping": 0.5},
            )
    assert predictor.state is safe
    assert vars(predictor) == before
    next_state = predictor.step(_boundary(safe.positions[0]), _boundary(safe.positions[-1]), _wrenches())
    assert next_state.timestamp_s == pytest.approx(0.01)
    assert next_state.radius_m == safe.radius_m


@pytest.mark.parametrize("angle", [0.0, np.pi / 2, np.pi])
def test_step_transports_curved_centerline_under_rigid_endpoint_motion(angle):
    predictor = ReducedCosseratPredictor(max_slip_m=1.0)
    points = np.array([[0., 0., 1.2], [0.1, 0.03, 1.2], [0.2, 0., 1.2]])
    safe = predictor.reset(points, radius_m=0.004)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    expected = points @ rotation.T + [0.01, 0.02, 0.0]
    result = predictor.step(_boundary(expected[0]), _boundary(expected[-1]), _wrenches())
    assert result.feasible
    np.testing.assert_allclose(result.positions, expected, atol=1.e-12)
    np.testing.assert_allclose(result.tangents, safe.tangents @ rotation.T, atol=1.e-12)
    assert result.bending_proxy == pytest.approx(safe.bending_proxy)


def test_curved_step_still_rejects_excess_bending():
    predictor = ReducedCosseratPredictor(max_bending_proxy=1.0)
    points = np.array([[0., 0., 1.2], [0.1, 0.03, 1.2], [0.2, 0., 1.2]])
    safe = predictor.reset(points, radius_m=0.004)
    result = predictor.step(_boundary(points[0]), _boundary(points[-1]), _wrenches())
    assert result.bending_proxy == pytest.approx(safe.bending_proxy)
    assert not result.feasible
    assert result.failure_reason == "bending_proxy"
    assert predictor.state is safe


@pytest.mark.parametrize("endpoint", [0, 1])
def test_endpoint_moment_changes_bending_and_rejects_without_state_mutation(endpoint):
    predictor = ReducedCosseratPredictor(max_bending_proxy=0.01)
    safe = predictor.reset(_centerline())
    wrench = np.zeros((2, 6))
    wrench[:, 2] = 1.0
    wrench[endpoint, 5] = 0.1
    result = predictor.step(_boundary(safe.positions[0]), _boundary(safe.positions[-1]), wrench)
    assert np.isfinite(result.bending_proxy)
    assert result.bending_proxy > predictor.max_bending_proxy
    assert not result.feasible
    assert result.failure_reason == "bending_proxy"
    assert predictor.state is safe


def test_extreme_endpoint_moment_has_bounded_bending_proxy():
    predictor = ReducedCosseratPredictor(max_bending_proxy=0.01)
    safe = predictor.reset(_centerline())
    wrench = np.array([[0., 0., 1., 0., 0., 1.e200], [0., 0., 1., 0., 0., 0.]])
    result = predictor.step(_boundary(safe.positions[0]), _boundary(safe.positions[-1]), wrench)
    segment_length = safe.rest_length_m / (len(safe.positions) - 1)
    assert predictor.max_bending_proxy < result.bending_proxy <= np.pi / segment_length
    assert not result.feasible
    assert predictor.state is safe


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("normal", [(0., 0., 1.), (1., 0., 0.)])
def test_tensile_normal_force_is_rejected_before_friction_allowance(endpoint, normal):
    predictor = ReducedCosseratPredictor()
    safe = predictor.reset(_centerline())
    wrenches = [{"force": np.array(normal), "normal": normal} for _ in range(2)]
    wrenches[endpoint]["force"] *= -1.0
    result = predictor.step(_boundary(safe.positions[0]), _boundary(safe.positions[-1]), wrenches)
    assert not result.friction_feasible
    assert not result.feasible
    assert result.failure_reason == "friction_cone"
    assert (result.left_normal_force_n, result.right_normal_force_n)[endpoint] == -1.0
    assert predictor.state is safe
