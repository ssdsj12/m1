from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import deformx_adapter
from go2_pvcnn.control.m1_bimanual_coordination.deformx_adapter import DualDloStepResult
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import DloCenterlineState
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_trajectory import DloWireTrajectoryPlanner
from go2_pvcnn.control.m1_bimanual_coordination.reduced_cosserat import ReducedCosseratPredictor
from tests.test_m1_bimanual_object_mpc import _snapshot
from tests.test_m1_bimanual_dual_arm_mpc import _object_solution


def _runtime_type():
    name = 'go2_pvcnn.control.m1_bimanual_coordination.reduced_wire_runtime'
    assert importlib.util.find_spec(name) is not None, 'ReducedWireRuntime facade is missing'
    return importlib.import_module(name).ReducedWireRuntime


def _geometry(snapshot):
    endpoints = np.stack([getattr(snapshot, f'{side}_arm').palm_pose_b[:3].numpy()
                          for side in ('left', 'right')])
    return DloCenterlineState(np.linspace(*endpoints, 5), radius_m=.004)


def _moving(snapshot, speed):
    return replace(snapshot, **{f'{side}_arm': replace(
        getattr(snapshot, f'{side}_arm'),
        palm_twist_b=torch.tensor([0., 0., speed, 0., 0., 0.], dtype=torch.float64),
    ) for side in ('left', 'right')})


class Backend:
    def __init__(self, state):
        self.state = state
        self.is_bound = False
        self.bind_count = 0
        self.steps = 0
        self.elapsed = .01
        self.impulse = np.zeros(3)

    def bind(self, initial_state, left_frame, right_frame):
        assert not self.is_bound
        np.testing.assert_allclose(left_frame['position'], initial_state.positions[0])
        np.testing.assert_allclose(right_frame['position'], initial_state.positions[-1])
        self.is_bound = True
        self.bind_count += 1

    def step(self, left_frame, right_frame):
        self.steps += 1
        self.frames = (left_frame, right_frame)
        return DualDloStepResult(self.state, self.impulse, np.zeros(3),
                                 self.impulse, np.zeros(3), self.elapsed)

    def reset(self):
        self.is_bound = False
        self.bind_count = 0


def test_without_backend_is_lazy_and_deterministic(monkeypatch):
    def forbidden():
        pytest.fail('DeformX must not load for reduced-only operation')
    monkeypatch.setattr(deformx_adapter, '_load_deformx', forbidden)
    snapshot = _moving(_snapshot(), .01)
    runtimes = [_runtime_type()(_geometry(snapshot), horizon_steps=4) for _ in range(2)]
    results = [runtime.update(snapshot, torch.zeros(2, 6), .01) for runtime in runtimes]
    for result in results:
        assert result.failure_reason is None
        assert result.endpoint_targets.shape == (4, 2, 3)
        assert result.endpoint_targets.dtype == snapshot.left_arm.palm_pose_b.dtype
        assert result.endpoint_targets.device == snapshot.left_arm.palm_pose_b.device
        assert result.prediction.positions.shape == (4, 5, 3)
        assert bool((result.lift_velocity > 0).all())
    torch.testing.assert_close(results[0].endpoint_targets, results[1].endpoint_targets)
    np.testing.assert_array_equal(results[0].prediction.positions, results[1].prediction.positions)


def test_backend_binds_once_and_uses_common_base_frames():
    snapshot = _snapshot()
    snapshot = replace(snapshot, base_state=torch.arange(13, dtype=torch.float64))
    backend = Backend(_geometry(snapshot))
    runtime = _runtime_type()(backend.state, backend=backend)
    assert not backend.is_bound
    for _ in range(2):
        assert runtime.update(snapshot, np.zeros((2, 6)), .01).failure_reason is None
    assert backend.bind_count == 1 and backend.steps == 2
    for frame, side in zip(backend.frames, ('left', 'right')):
        np.testing.assert_array_equal(frame['position'], getattr(snapshot, f'{side}_arm').palm_pose_b[:3])
    runtime.reset()
    assert not backend.is_bound
    assert runtime.update(snapshot, np.zeros((2, 6)), .01).failure_reason is None
    assert backend.bind_count == 1


def test_impulse_conversion_preserves_sides_sign_and_bounds_vector_norm():
    convert = getattr(deformx_adapter, 'endpoint_wrenches_from_impulses', None)
    assert callable(convert), 'impulse conversion is missing'
    state = _geometry(_snapshot())
    step = DualDloStepResult(state, np.array([3., 4., 0.]), np.array([0., 0., -2.]),
                            np.array([-1., 0., 0.]), np.array([0., .01, 0.]), .1)
    wrenches = convert(step, max_force_n=10., max_moment_nm=2.)
    np.testing.assert_allclose(wrenches, [[6., 8., 0., 0., 0., -2.], [-10., 0., 0., 0., .1, 0.]])


@pytest.mark.parametrize('elapsed', [0., -1., np.nan, np.inf])
def test_invalid_impulse_elapsed_rejects_atomically(elapsed):
    snapshot = _snapshot()
    backend = Backend(_geometry(snapshot))
    runtime = _runtime_type()(backend.state, backend=backend)
    accepted = runtime.update(snapshot, np.zeros((2, 6)), .01)
    backend.elapsed = elapsed
    rejected = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert rejected.failure_reason == 'invalid_backend_output'
    assert rejected.state is accepted.state
    assert runtime.state is accepted.state
    assert torch.count_nonzero(rejected.lift_velocity) == 0


def test_backend_impulse_is_used_as_endpoint_wrench():
    snapshot = _snapshot()
    backend = Backend(_geometry(snapshot))
    backend.impulse = np.array([0., 0., .001])
    runtime = _runtime_type()(backend.state, backend=backend, horizon_steps=2)
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert result.failure_reason is None
    assert result.state.left_normal_force_n == pytest.approx(.1)
    assert result.state.right_normal_force_n == pytest.approx(.1)


@pytest.mark.parametrize('wrench', [np.full((2, 6), np.nan), np.zeros((2, 5)), np.full((2, 6), np.inf)])
def test_bad_wrench_does_not_advance_backend_or_accepted_state(wrench):
    snapshot = _snapshot()
    backend = Backend(_geometry(snapshot))
    runtime = _runtime_type()(backend.state, backend=backend)
    accepted = runtime.update(snapshot, np.zeros((2, 6)), .01)
    rejected = runtime.update(snapshot, wrench, .01)
    assert rejected.failure_reason == 'invalid_endpoint_wrench'
    assert backend.steps == 1
    assert rejected.state is accepted.state


def test_infeasible_future_rolls_back_even_successful_current_step():
    snapshot = _snapshot()
    runtime = _runtime_type()(_geometry(snapshot), predictor=ReducedCosseratPredictor(max_slip_m=.002))
    accepted = runtime.update(snapshot, np.zeros((2, 6)), .01)
    rejected = runtime.update(_moving(snapshot, 1.), np.zeros((2, 6)), .01)
    assert rejected.failure_reason.startswith('infeasible_prediction:')
    assert rejected.state is accepted.state and runtime.state is accepted.state
    assert rejected.prediction is None
    torch.testing.assert_close(rejected.endpoint_targets,
        torch.tensor(accepted.state.endpoint_positions.copy()).repeat(25, 1, 1))
    assert torch.count_nonzero(rejected.endpoint_velocities) == 0
    recovered = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert recovered.state.timestamp_s == pytest.approx(accepted.state.timestamp_s + .01)


def test_planner_consumes_prediction_and_stops_lift_on_rejection():
    snapshot = _moving(_snapshot(), .01)
    runtime = _runtime_type()(_geometry(snapshot))
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    planner = DloWireTrajectoryPlanner(dt=.01, palm_speed_m_s=10.)
    assert hasattr(planner, 'plan_prediction'), 'prediction planning is missing'
    original = _object_solution()
    planned = planner.plan_prediction(snapshot, original, result).object_solution
    for side, index in (('left', 0), ('right', 1)):
        path = getattr(planned, f'{side}_palm_pose')
        torch.testing.assert_close(path[:, 2], result.endpoint_targets[:, index, 2])
        torch.testing.assert_close(path[:, 3:], getattr(snapshot, f'{side}_arm').palm_pose_b[3:].repeat(25, 1))
    torch.testing.assert_close(planned.box_twist[:, 2], result.lift_velocity)
    assert planned.diagnostics is original.diagnostics
    rejected = runtime.update(snapshot, np.full((2, 6), np.nan), .01)
    held = planner.plan_prediction(snapshot, original, rejected).object_solution
    assert torch.count_nonzero(held.box_twist[:, 2]) == 0
    for side in ('left', 'right'):
        torch.testing.assert_close(getattr(held, f'{side}_palm_pose'),
            getattr(snapshot, f'{side}_arm').palm_pose_b.repeat(25, 1))


def test_impulse_conversion_handles_subnormal_interval_without_losing_direction():
    step = DualDloStepResult(_geometry(_snapshot()), np.array([1.e308, 1.e308, 0.]),
                            np.zeros(3), np.zeros(3), np.zeros(3), np.nextafter(0., 1.))
    result = deformx_adapter.endpoint_wrenches_from_impulses(step, max_force_n=10.)
    np.testing.assert_allclose(result[0, :3], np.array([1., 1., 0.]) * 10. / np.sqrt(2.))


@pytest.mark.parametrize('operation', ['bind', 'step'])
def test_backend_exception_keeps_last_safe_state(operation):
    class Broken(Backend):
        def bind(self, *args, **kwargs):
            if operation == 'bind':
                raise RuntimeError('engine bind failed')
            super().bind(*args, **kwargs)

        def step(self, *args, **kwargs):
            raise RuntimeError('engine step failed')
    snapshot = _snapshot()
    runtime = _runtime_type()(_geometry(snapshot), backend=Broken(_geometry(snapshot)))
    before = runtime.state
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert result.failure_reason == 'backend_update_failed'
    assert runtime.state is before and result.state is before


def test_rejects_malformed_prediction_atomically():
    class Malformed(ReducedCosseratPredictor):
        def predict(self, *args, **kwargs):
            prediction = super().predict(*args, **kwargs)
            # Simulate a predictor implementation violating the public contract.
            object.__setattr__(prediction, 'endpoint_positions', np.full((25, 2, 3), np.nan))
            return prediction
    snapshot = _snapshot()
    runtime = _runtime_type()(_geometry(snapshot), predictor=Malformed())
    before = runtime.state
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert result.failure_reason == 'invalid_prediction'
    assert runtime.state is before
    assert bool(torch.isfinite(result.endpoint_targets).all())


def test_updated_dt_is_used_for_prediction_and_normals_are_in_base_frame():
    snapshot = _moving(_snapshot(), .01)
    runtime = _runtime_type()(_geometry(snapshot), horizon_steps=3)
    wrench = {'left': {'force': [0., .01, 0.], 'normal': [0., 1., 0.]},
              'right': {'force': [0., -.01, 0.], 'normal': [0., -1., 0.]}}
    result = runtime.update(snapshot, wrench, .04)
    assert result.failure_reason is None
    assert result.state.left_normal_force_n == pytest.approx(.01)
    assert result.state.right_normal_force_n == pytest.approx(.01)
    np.testing.assert_allclose(result.prediction.endpoint_positions[0],
        result.state.endpoint_positions + .04 * result.state.endpoint_velocities)


def test_backend_centerline_curvature_is_observed():
    snapshot = _snapshot()
    initial = _geometry(snapshot)
    backend = Backend(initial)
    runtime = _runtime_type()(initial, backend=backend)
    points = initial.positions.copy()
    points[2, 2] -= .001
    backend.state = DloCenterlineState(points, initial.radius_m)
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    assert result.failure_reason is None
    np.testing.assert_array_equal(result.state.positions, points)
    assert result.state.bending_proxy > 0.


def test_planner_rejects_corrupted_targets_and_keeps_each_node_speed_bounded():
    snapshot = _moving(_snapshot(), .01)
    runtime = _runtime_type()(_geometry(snapshot))
    result = runtime.update(snapshot, np.zeros((2, 6)), .01)
    planner = DloWireTrajectoryPlanner(dt=.01, palm_speed_m_s=.02)
    plan = planner.plan_prediction(snapshot, _object_solution(), result).object_solution
    for side in ('left', 'right'):
        path = getattr(plan, f'{side}_palm_pose')[:, :3]
        current = getattr(snapshot, f'{side}_arm').palm_pose_b[:3]
        assert bool((torch.linalg.vector_norm(torch.diff(torch.cat((current[None], path)), dim=0), dim=1) <= .0002 + 1.e-12).all())
    bad = replace(result, endpoint_targets=torch.full_like(result.endpoint_targets, torch.nan))
    with pytest.raises(ValueError, match='finite'):
        planner.plan_prediction(snapshot, _object_solution(), bad)


def test_lift_reference_does_not_outpace_either_bounded_palm_path():
    snapshot = _moving(_snapshot(), .01)
    result = _runtime_type()(_geometry(snapshot)).update(snapshot, np.zeros((2, 6)), .01)
    planner = DloWireTrajectoryPlanner(dt=.01, palm_speed_m_s=.02)
    plan = planner.plan_prediction(snapshot, _object_solution(), result).object_solution
    for side in ('left', 'right'):
        initial = getattr(snapshot, f'{side}_arm').palm_pose_b[2:3]
        heights = getattr(plan, f'{side}_palm_pose')[:, 2]
        realized_velocity = torch.diff(torch.cat((initial, heights))) / .01
        assert bool((plan.box_twist[:, 2] <= realized_velocity.clamp_min(0.) + 1.e-12).all())


@pytest.mark.parametrize('dt', [0., -1., np.nan, np.inf])
def test_bad_dt_does_not_bind_backend(dt):
    snapshot = _snapshot()
    backend = Backend(_geometry(snapshot))
    runtime = _runtime_type()(backend.state, backend=backend)
    result = runtime.update(snapshot, np.zeros((2, 6)), dt)
    assert result.failure_reason == 'invalid_dt'
    assert not backend.is_bound and backend.steps == 0


def test_invalid_current_state_does_not_replace_last_safe():
    snapshot = _snapshot()
    runtime = _runtime_type()(_geometry(snapshot))
    before = runtime.state
    pose = snapshot.left_arm.palm_pose_b.clone()
    pose[2] += .1
    changed = replace(snapshot, left_arm=replace(snapshot.left_arm, palm_pose_b=pose))
    result = runtime.update(changed, np.zeros((2, 6)), .01)
    assert result.failure_reason.startswith('infeasible_state:')
    assert result.state is before and runtime.state is before
