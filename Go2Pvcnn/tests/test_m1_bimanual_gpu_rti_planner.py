import gc
import warnings

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.lq_problem import CoupledRtiInput
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.planner import (
    EagerBimanualRtiPlanner,
    REASON_ACCEPTED,
    REASON_DYNAMICS_INVALID,
    REASON_INPUT_INVALID,
    REASON_LINE_SEARCH_REJECTED,
    REASON_NOMINAL_INVALID,
    REASON_NO_SAFE_HORIZON,
    REASON_NONFINITE_RESULT,
)
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.reduced_dynamics import GpuReducedDynamics


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")
STATE_ATOL = 5.0e-5
ACTION_ATOL = 5.0e-4
ACTION_RTOL = 5.0e-4
RESERVED_MEMORY_DELTA_BYTES = 2 * 1024 * 1024


def _storage_pointers(values):
    leaves, _ = tree_flatten(values)
    return {
        value.untyped_storage().data_ptr()
        for value in leaves
        if isinstance(value, torch.Tensor)
    }


class _NoNewStorage(TorchDispatchMode):
    def __init__(self, known):
        self.known = known

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert "_local_scalar_dense" not in str(func), f"host scalar extraction: {func}"
        output = func(*args, **(kwargs or {}))
        assert _storage_pointers(output) <= self.known, f"new tensor storage: {func}"
        return output


def _fixture(batch=2, *, equality=True):
    device = torch.device("cuda:0")
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
    mass = torch.eye(59, dtype=torch.float32, device=device).expand(batch, -1, -1).clone()
    selection = z(batch, 59, 43)
    selection[:, torch.arange(6, 49), torch.arange(43)] = 1.0
    wheel = z(batch, 12, 59)
    wheel[:, torch.arange(12), torch.arange(47, 59)] = 1.0
    dynamics = GpuReducedDynamics(batch=batch)
    dynamics.step(
        mass_matrix=mass,
        bias=z(batch, 59),
        actuation_matrix=selection,
        wheel_contact_jacobian=wheel,
        wheel_contact_bias=z(batch, 12),
    )
    active_selector = z(batch, 43, 59)
    active_selector[:, torch.arange(43), torch.arange(6, 49)] = 1.0
    palms = z(batch, 2, 6, 59)
    palms[:, :, torch.arange(6), torch.arange(6)] = 1.0
    box_map = z(batch, 6, 12)
    box_map[:, torch.arange(6), torch.arange(6)] = 1.0
    box_map[:, torch.arange(6), torch.arange(6) + 6] = 1.0
    nominal = z(batch, 25, 55)
    nominal[:, :, :43] = torch.linspace(0.01, 0.03, 25, device=device)[None, :, None]
    nominal[:, :, 43:] = 0.005
    lower = nominal.clone() if equality else torch.full_like(nominal, -10.0)
    upper = nominal.clone() if equality else torch.full_like(nominal, 10.0)
    target = z(batch, 26, 110)
    state_lower = torch.full_like(target, -100.0)
    state_upper = torch.full_like(target, 100.0)
    measured = z(batch, 111)
    measured[:, 3] = 1.0
    base_pose = z(batch, 6)
    base_pose[:, 0:3] = torch.tensor((0.1, -0.2, 0.3), device=device)
    base_pose[:, 5] = 0.2
    base_quaternion = torch.tensor((0.9238795, 0.3826834, 0.0, 0.0), device=device).expand(batch, -1).clone()
    problem = CoupledRtiInput(
        measured_state=measured,
        base_pose0=base_pose,
        base_twist0=z(batch, 6),
        base_quaternion0=base_quaternion,
        active_q0=z(batch, 43),
        active_qd0=z(batch, 43),
        box_pose0=z(batch, 6),
        box_twist0=z(batch, 6),
        active_selector=active_selector,
        palm_jacobian=palms,
        box_wrench_map=box_map,
        box_gravity=z(batch, 6),
        nominal_control=nominal,
        control_lower=lower,
        control_upper=upper,
        state_target=target,
        state_weight=torch.ones(110, dtype=torch.float32, device=device),
        control_weight=torch.full((55,), 0.1, dtype=torch.float32, device=device),
        terminal_weight=torch.ones(110, dtype=torch.float32, device=device),
        palm_residual_offset=z(batch, 25, 12),
        palm_state_jacobian=z(batch, 25, 12, 110),
        palm_weight=torch.zeros(12, dtype=torch.float32, device=device),
        state_lower=state_lower,
        state_upper=state_upper,
        wrench_inequality_matrix=z(batch, 25, 1, 12),
        wrench_inequality_upper=torch.full((batch, 25, 1), 100.0, device=device),
        hard_inequality_matrix=z(batch, 25, 1, 55),
        hard_inequality_upper=torch.full((batch, 25, 1), 100.0, device=device),
        input_valid=torch.ones(batch, dtype=torch.bool, device=device),
    )
    identity = torch.stack(
        (torch.arange(batch, device=device), torch.zeros(batch, device=device), torch.ones(batch, device=device)),
        dim=1,
    ).to(torch.int64)
    reset = torch.zeros(batch, dtype=torch.bool, device=device)
    planner = EagerBimanualRtiPlanner(
        batch=batch, max_wrench_constraints=1, max_hard_constraints=1
    )
    return planner, dynamics, problem, identity, reset


def _expected_quaternion(q0, tangent):
    q0 = q0 / torch.linalg.vector_norm(q0)
    angle = torch.linalg.vector_norm(tangent)
    if angle == 0:
        delta = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=q0.dtype, device=q0.device)
    else:
        delta = torch.cat((torch.cos(angle / 2).view(1), tangent * (torch.sin(angle / 2) / angle)))
    aw, ax, ay, az = q0
    bw, bx, by, bz = delta
    result = torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ))
    return result / torch.linalg.vector_norm(result)


def _force_strict_line_search_rejection(planner, dynamics, problem):
    problem.control_lower.fill_(-10.0)
    problem.control_upper.fill_(10.0)
    problem.control_weight.zero_()
    planner.lq.assemble(problem, dynamics)
    problem.state_target.copy_(planner.lq.rollout(problem.nominal_control))


def test_one_rti_call_public_shape_mapping_and_persistent_quaternion(monkeypatch):
    planner, dynamics, problem, identity, reset = _fixture(batch=2)
    solve_calls = 0
    search_calls = 0
    original_solve = planner.lq.solve_direction
    original_search = planner.line_search.step

    def solve_once():
        nonlocal solve_calls
        solve_calls += 1
        return original_solve()

    def search_once(workspace):
        nonlocal search_calls
        search_calls += 1
        return original_search(workspace)

    monkeypatch.setattr(planner.lq, "solve_direction", solve_once)
    monkeypatch.setattr(planner.line_search, "step", search_once)
    result = planner.step(problem, dynamics, identity, reset)
    assert solve_calls == search_calls == 1
    assert result.action.shape == (2, 25, 43)
    assert result.rti_iterations == 1
    assert result.accepted.tolist() == [True, True]
    assert result.safe_available.tolist() == [True, True]
    assert result.reason_code.tolist() == [REASON_ACCEPTED, REASON_ACCEPTED]
    assert torch.equal(result.action, problem.nominal_control[..., :43])
    assert planner.warm.accepted_action.shape[-1] == 43
    reporting = planner.warm.accepted_state
    torch.testing.assert_close(reporting[:, 0, 0:3], problem.base_pose0[:, 0:3])
    torch.testing.assert_close(reporting[:, 0, 7:13], problem.base_twist0)
    torch.testing.assert_close(reporting[:, :, 99:111], planner.line_search.result_state[:, :, 98:110])
    expected = _expected_quaternion(problem.base_quaternion0[0], problem.base_pose0[0, 3:6])
    torch.testing.assert_close(reporting[0, 0, 3:7], expected, atol=STATE_ATOL, rtol=0.0)
    assert torch.sum(reporting[:, :, 3:7] * problem.base_quaternion0[:, None], dim=-1).ge(0).all()


def test_row_local_repeated_last_safe_fallback_reset_and_identity_invalidation():
    planner, dynamics, problem, identity, reset = _fixture(batch=2)
    seeded = planner.step(problem, dynamics, identity, reset)
    safe = seeded.action.clone()

    problem.input_valid[0] = False
    _force_strict_line_search_rejection(planner, dynamics, problem)
    failed = planner.step(problem, dynamics, identity, reset)
    assert failed.accepted.tolist() == [False, False]
    assert failed.safe_available.tolist() == [True, True]
    assert failed.reason_code.tolist() == [REASON_INPUT_INVALID, REASON_LINE_SEARCH_REJECTED]
    assert torch.equal(failed.action[0, :-1], safe[0, 1:])
    assert torch.equal(failed.action[0, -1], safe[0, -1])
    assert torch.equal(failed.action[1, :-1], safe[1, 1:])
    shifted_once = failed.action.clone()

    failed_again = planner.step(problem, dynamics, identity, reset)
    assert failed_again.accepted.tolist() == [False, False]
    assert failed_again.safe_available.tolist() == [True, True]
    assert torch.equal(failed_again.action[:, :-1], shifted_once[:, 1:])

    reset[0] = True
    identity[1, 2] += 1
    invalidated = planner.step(problem, dynamics, identity, reset)
    assert invalidated.accepted.tolist() == [False, False]
    assert invalidated.safe_available.tolist() == [False, False]
    assert invalidated.reason_code.tolist() == [REASON_NO_SAFE_HORIZON] * 2
    assert not invalidated.action.any()


def test_left_side_nonfinite_failure_cannot_alter_accepted_neighbor():
    planner, dynamics, problem, identity, reset = _fixture(batch=2)
    planner.step(problem, dynamics, identity, reset)
    problem.palm_jacobian[0, 0, 0, 0] = float("nan")
    result = planner.step(problem, dynamics, identity, reset)
    assert result.accepted.tolist() == [False, True]
    assert result.safe_available.tolist() == [True, True]
    assert result.reason_code.tolist() == [REASON_NONFINITE_RESULT, REASON_ACCEPTED]
    assert torch.equal(result.action[1], problem.nominal_control[1, :, :43])


@pytest.mark.parametrize(
    "cause,reason",
    [
        ("input", REASON_INPUT_INVALID),
        ("dynamics", REASON_DYNAMICS_INVALID),
        ("nonfinite", REASON_NONFINITE_RESULT),
        ("nominal", REASON_NOMINAL_INVALID),
        ("line_search", REASON_LINE_SEARCH_REJECTED),
    ],
)
def test_failure_reason_precedence_with_available_fallback(cause, reason):
    planner, dynamics, problem, identity, reset = _fixture(batch=1)
    planner.step(problem, dynamics, identity, reset)
    if cause == "input":
        problem.input_valid.zero_()
    elif cause == "dynamics":
        dynamics.valid.zero_()
    elif cause == "nonfinite":
        problem.base_quaternion0[0, 0] = float("nan")
    elif cause == "nominal":
        problem.control_lower.fill_(-10.0)
        problem.control_upper.fill_(10.0)
        problem.state_upper[:, 0, 0] = problem.base_pose0[:, 0] - 1.0
    else:
        _force_strict_line_search_rejection(planner, dynamics, problem)
    result = planner.step(problem, dynamics, identity, reset)
    assert not result.accepted.item() and result.safe_available.item()
    assert result.reason_code.item() == reason


def test_no_safe_horizon_overrides_attempt_reason():
    planner, dynamics, problem, identity, reset = _fixture(batch=1, equality=False)
    problem.input_valid.zero_()
    result = planner.step(problem, dynamics, identity, reset)
    assert not result.accepted.item() and not result.safe_available.item()
    assert result.reason_code.item() == REASON_NO_SAFE_HORIZON
    assert not result.action.any()


def _cpu_rollout(problem, dynamics_inputs, control):
    mass = dynamics_inputs["mass_matrix"].double().cpu()[0]
    actuation = dynamics_inputs["actuation_matrix"].double().cpu()[0]
    wheel = dynamics_inputs["wheel_contact_jacobian"].double().cpu()[0]
    kkt = torch.zeros(71, 71, dtype=torch.float64)
    kkt[:59, :59] = mass
    kkt[:59, 59:] = -wheel.T
    kkt[59:, :59] = wheel
    rhs = torch.zeros(71, 56, dtype=torch.float64)
    rhs[:59, 1:44] = actuation
    palms = problem.palm_jacobian.double().cpu()[0]
    rhs[:59, 44:50] = -palms[0].T
    rhs[:59, 50:56] = -palms[1].T
    solution = torch.linalg.solve(kkt, rhs)
    qdd_offset = solution[:59, 0]
    qdd_effort = solution[:59, 1:44]
    qdd_wrench = solution[:59, 44:56]
    selector = problem.active_selector.double().cpu()[0]
    active_effort = selector @ qdd_effort
    active_wrench = selector @ qdd_wrench
    base_effort = qdd_effort[:6]
    base_wrench = qdd_wrench[:6]
    box_map = problem.box_wrench_map.double().cpu()[0]
    a = torch.eye(110, dtype=torch.float64)
    a[0:6, 6:12] = 0.04 * torch.eye(6, dtype=torch.float64)
    a[12:55, 55:98] = 0.04 * torch.eye(43, dtype=torch.float64)
    a[98:104, 104:110] = 0.04 * torch.eye(6, dtype=torch.float64)
    b = torch.zeros(110, 55, dtype=torch.float64)
    b[0:6, :43] = 0.0008 * base_effort
    b[0:6, 43:] = 0.0008 * base_wrench
    b[6:12, :43] = 0.04 * base_effort
    b[6:12, 43:] = 0.04 * base_wrench
    b[12:55, :43] = 0.0008 * active_effort
    b[12:55, 43:] = 0.0008 * active_wrench
    b[55:98, :43] = 0.04 * active_effort
    b[55:98, 43:] = 0.04 * active_wrench
    b[98:104, 43:] = 0.0008 * box_map
    b[104:110, 43:] = 0.04 * box_map
    x0 = torch.cat((
        problem.base_pose0.double().cpu()[0], problem.base_twist0.double().cpu()[0],
        problem.active_q0.double().cpu()[0], problem.active_qd0.double().cpu()[0],
        problem.box_pose0.double().cpu()[0], problem.box_twist0.double().cpu()[0],
    ))
    states = torch.empty(26, 110, dtype=torch.float64)
    states[0] = x0
    for node in range(25):
        states[node + 1] = a @ states[node] + b @ control[node]
    return states, a, b


def _cpu_reference(problem, dynamics_inputs):
    nominal = problem.nominal_control.double().cpu()[0]
    nominal_state, a, b = _cpu_rollout(problem, dynamics_inputs, nominal)
    jac = torch.zeros(25, 110, 1375, dtype=torch.float64)
    for node in range(25):
        if node:
            jac[node] = a @ jac[node - 1]
        jac[node, :, node * 55 : (node + 1) * 55] += b
    target = problem.state_target.double().cpu()[0]
    sw = problem.state_weight.double().cpu()
    tw = problem.terminal_weight.double().cpu()
    cw = problem.control_weight.double().cpu()
    hessian = torch.zeros(1375, 1375, dtype=torch.float64)
    gradient = torch.zeros(1375, dtype=torch.float64)
    for node in range(25):
        weight = sw + (tw if node == 24 else 0.0)
        hessian += jac[node].T @ (weight[:, None] * jac[node])
        gradient += jac[node].T @ (weight * (nominal_state[node + 1] - target[node + 1]))
        block = slice(node * 55, (node + 1) * 55)
        hessian[block, block].diagonal().add_(cw)
        gradient[block] += cw * nominal[node]
    hessian.mul_(2.0)
    gradient.mul_(2.0)
    hessian.diagonal().add_(2.0e-6)
    direction = torch.linalg.solve(hessian, -gradient).view(25, 55)
    lower = problem.control_lower.double().cpu()[0]
    upper = problem.control_upper.double().cpu()[0]
    equality = lower == upper

    def merit(state, control):
        value = ((state[1:] - target[1:]).square() * sw).sum()
        value += ((state[-1] - target[-1]).square() * tw).sum()
        value += (control.square() * cw).sum()
        return value

    nominal_merit = merit(nominal_state, nominal)
    candidates = []
    for alpha in (1.0, 0.5, 0.25, 0.125):
        control = (nominal + alpha * direction).clamp(min=lower, max=upper)
        control[equality] = nominal[equality]
        state, _, _ = _cpu_rollout(problem, dynamics_inputs, control)
        valid = bool(((control >= lower) & (control <= upper)).all())
        candidates.append((merit(state, control) if valid else torch.tensor(float("inf")), control, state))
    best = min(range(4), key=lambda index: float(candidates[index][0]))
    accepted = bool(candidates[best][0] < nominal_merit)
    return candidates[best][1], candidates[best][2], accepted


def test_cpu_float64_independent_parity_active_effort_bound_and_equality_bypass():
    planner, dynamics, problem, identity, reset = _fixture(batch=1, equality=False)
    dynamics_inputs = dynamics._inputs
    desired = problem.nominal_control.clone()
    desired[:, :, 0] = 0.8
    cpu_target, _, _ = _cpu_rollout(problem, dynamics_inputs, desired.double().cpu()[0])
    problem.state_target.copy_(cpu_target.float().cuda().unsqueeze(0))
    problem.control_upper[:, :, 0] = 0.3
    cpu_control, cpu_state, cpu_accepted = _cpu_reference(problem, dynamics_inputs)
    result = planner.step(problem, dynamics, identity, reset)
    assert result.accepted.item() == cpu_accepted
    torch.testing.assert_close(
        planner.line_search.result_state[0].double().cpu(), cpu_state, atol=STATE_ATOL, rtol=0.0
    )
    torch.testing.assert_close(
        result.action[0, 0].double().cpu(), cpu_control[0, :43], atol=ACTION_ATOL, rtol=ACTION_RTOL
    )
    assert result.action[0, :, 0].le(0.3).all()

    problem.control_lower.copy_(problem.nominal_control)
    problem.control_upper.copy_(problem.nominal_control)
    bypass = planner.step(problem, dynamics, identity, reset)
    assert bypass.accepted.item()
    assert torch.equal(bypass.action, problem.nominal_control[..., :43])


def test_planner_reuses_storage_and_dispatches_without_host_scalars_or_new_storage():
    planner, dynamics, problem, identity, reset = _fixture(batch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(10):
            planner.step(problem, dynamics, identity, reset)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    resident = (vars(planner), vars(planner.lq), vars(planner.line_search), vars(planner.warm))
    pointers = _storage_pointers(resident)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    known = _storage_pointers((vars(planner), vars(planner.lq), vars(planner.lq._finite),
                               vars(planner.line_search), vars(planner.warm), vars(planner.warm._finite),
                               vars(planner._finite), vars(dynamics), vars(dynamics._finite), vars(problem),
                               identity, reset))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _NoNewStorage(known):
            for _ in range(100):
                result = planner.step(problem, dynamics, identity, reset)
    torch.cuda.synchronize()
    assert result.action.data_ptr() == planner.action.data_ptr()
    assert pointers == _storage_pointers(resident)
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= RESERVED_MEMORY_DELTA_BYTES
    print(f"CUDA0 planner storages={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")
