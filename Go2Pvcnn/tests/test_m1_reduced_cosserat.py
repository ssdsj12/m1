from __future__ import annotations

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import DloCenterlineState
from go2_pvcnn.control.m1_bimanual_coordination.reduced_cosserat import (
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
