from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.lq_problem import (
    CoupledLqWorkspace,
    CoupledRtiDims,
    CoupledRtiInput,
)
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.reduced_dynamics import (
    GpuReducedDynamics,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")
PARITY_ATOL = 3.0e-5
PARITY_RTOL = 3.0e-5
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


def _dynamics_inputs(batch=2):
    generator = torch.Generator().manual_seed(11)
    raw = torch.randn(batch, 59, 59, dtype=torch.float64, generator=generator)
    mass = raw @ raw.transpose(1, 2) / 59.0 + 2.0 * torch.eye(59, dtype=torch.float64)
    bias = torch.linspace(-0.2, 0.2, 59, dtype=torch.float64).expand(batch, -1).clone()
    selection = torch.zeros(batch, 59, 43, dtype=torch.float64)
    selection[:, torch.arange(6, 49), torch.arange(43)] = 1.0
    wheel = torch.zeros(batch, 12, 59, dtype=torch.float64)
    wheel[:, torch.arange(12), torch.arange(47, 59)] = 1.0
    wheel_bias = torch.linspace(-0.05, 0.05, 12, dtype=torch.float64).expand(batch, -1).clone()
    cpu = {
        "mass_matrix": mass,
        "bias": bias,
        "actuation_matrix": selection,
        "wheel_contact_jacobian": wheel,
        "wheel_contact_bias": wheel_bias,
    }
    gpu = {name: value.to("cuda:0", torch.float32) for name, value in cpu.items()}
    return cpu, gpu


def _problem(batch=2, wrench_constraints=3, hard_constraints=2):
    device = torch.device("cuda:0")
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
    measured = z(batch, 111)
    measured[:, 0] = 99.0  # Must not be treated as the private optimizer layout.
    base_pose = z(batch, 6)
    base_twist = z(batch, 6)
    base_twist[:, :3] = torch.tensor([0.3, -0.2, 0.1], device=device)
    active_q = z(batch, 43)
    active_qd = z(batch, 43)
    box_pose = z(batch, 6)
    box_twist = base_twist.clone()
    selector = z(batch, 43, 59)
    selector[:, torch.arange(43), torch.arange(6, 49)] = 1.0
    palms = z(batch, 2, 6, 59)
    palms[:, :, torch.arange(6), torch.arange(6)] = 1.0
    box_map = z(batch, 6, 12)
    box_map[:, torch.arange(6), torch.arange(6)] = 1.0
    box_map[:, torch.arange(6), torch.arange(6) + 6] = 1.0
    nominal = z(batch, 25, 55)
    nominal[:, :, :43] = 0.02
    nominal[:, :, 43:] = 0.01
    lower = torch.full_like(nominal, -10.0)
    upper = torch.full_like(nominal, 10.0)
    target = z(batch, 26, 110)
    target[:, :, 98] = 0.15
    state_lower = torch.full_like(target, -100.0)
    state_upper = torch.full_like(target, 100.0)
    wrench_matrix = z(batch, 25, wrench_constraints, 12)
    wrench_upper = torch.full((batch, 25, wrench_constraints), 100.0, dtype=torch.float32, device=device)
    hard_matrix = z(batch, 25, hard_constraints, 55)
    hard_upper = torch.full((batch, 25, hard_constraints), 100.0, dtype=torch.float32, device=device)
    return CoupledRtiInput(
        measured_state=measured,
        base_pose0=base_pose,
        base_twist0=base_twist,
        base_quaternion0=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch, -1).clone(),
        active_q0=active_q,
        active_qd0=active_qd,
        box_pose0=box_pose,
        box_twist0=box_twist,
        active_selector=selector,
        palm_jacobian=palms,
        box_wrench_map=box_map,
        box_gravity=z(batch, 6),
        nominal_control=nominal,
        control_lower=lower,
        control_upper=upper,
        state_target=target,
        state_weight=torch.ones(110, dtype=torch.float32, device=device),
        control_weight=torch.full((55,), 0.1, dtype=torch.float32, device=device),
        terminal_weight=torch.full((110,), 2.0, dtype=torch.float32, device=device),
        palm_residual_offset=z(batch, 25, 12),
        palm_state_jacobian=z(batch, 25, 12, 110),
        palm_weight=torch.ones(12, dtype=torch.float32, device=device),
        state_lower=state_lower,
        state_upper=state_upper,
        wrench_inequality_matrix=wrench_matrix,
        wrench_inequality_upper=wrench_upper,
        hard_inequality_matrix=hard_matrix,
        hard_inequality_upper=hard_upper,
        input_valid=torch.ones(batch, dtype=torch.bool, device=device),
    )


def _setup(batch=2):
    cpu, gpu = _dynamics_inputs(batch)
    dynamics = GpuReducedDynamics(batch=batch)
    dynamics.step(**gpu)
    problem = _problem(batch)
    work = CoupledLqWorkspace(
        batch=batch,
        max_wrench_constraints=problem.wrench_inequality_matrix.shape[2],
        max_hard_constraints=problem.hard_inequality_matrix.shape[2],
    )
    return cpu, gpu, dynamics, problem, work


def _optimizer_x0(problem):
    return torch.cat(
        (
            problem.base_pose0,
            problem.base_twist0,
            problem.active_q0,
            problem.active_qd0,
            problem.box_pose0,
            problem.box_twist0,
        ),
        dim=1,
    )


def test_fixed_contract_and_float32_workspace():
    dims = CoupledRtiDims()
    assert (dims.horizon, dims.state_dim, dims.effort_dim, dims.wrench_dim, dims.control_dim) == (25, 110, 43, 12, 55)
    with pytest.raises(FrozenInstanceError):
        dims.horizon = 26
    _, _, _, _, work = _setup()
    assert work.A.shape == (2, 25, 110, 110)
    assert work.B.shape == (2, 25, 110, 55)
    assert work.c.shape == (2, 25, 110)
    assert work.H.shape == (2, 1375, 1375)
    leaves, _ = tree_flatten(vars(work))
    assert all(value.dtype == torch.float32 for value in leaves if isinstance(value, torch.Tensor) and value.is_floating_point())


def test_augmented_kkt_wrench_reaction_matches_float64_reference_and_power_cancels():
    cpu, _, dynamics, problem, work = _setup()
    assert work.assemble(problem, dynamics).tolist() == [True, True]
    for row in range(2):
        kkt = torch.zeros(71, 71, dtype=torch.float64)
        kkt[:59, :59] = cpu["mass_matrix"][row]
        kkt[:59, 59:] = -cpu["wheel_contact_jacobian"][row].T
        kkt[59:, :59] = cpu["wheel_contact_jacobian"][row]
        rhs = torch.zeros(71, 56, dtype=torch.float64)
        rhs[:59, 0] = -cpu["bias"][row]
        rhs[59:, 0] = -cpu["wheel_contact_bias"][row]
        rhs[:59, 1:44] = cpu["actuation_matrix"][row]
        rhs[:59, 44:50] = -problem.palm_jacobian[row, 0].cpu().double().T
        rhs[:59, 50:56] = -problem.palm_jacobian[row, 1].cpu().double().T
        reference = torch.linalg.solve(kkt, rhs)
        torch.testing.assert_close(work.qdd_wrench[row].cpu().double(), reference[:59, 44:56], atol=PARITY_ATOL, rtol=PARITY_RTOL)
        assert torch.equal(work.wrench_rhs[row, :59, :6], -problem.palm_jacobian[row, 0].T)
        assert torch.equal(work.wrench_rhs[row, :59, 6:], -problem.palm_jacobian[row, 1].T)
        assert not work.wrench_rhs[row, 59:].any()

    qd = torch.zeros(2, 59, device="cuda:0")
    qd[:, :6] = problem.box_twist0
    wrench = torch.tensor([0.4, -0.1, 0.3, 0.2, -0.5, 0.7, -0.2, 0.6, 0.1, -0.4, 0.3, 0.8], device="cuda:0").expand(2, -1)
    palm_twist = torch.einsum("bsij,bj->bsi", problem.palm_jacobian, qd)
    robot_power = -(palm_twist * wrench.view(2, 2, 6)).sum(dim=(1, 2))
    box_power = (problem.box_twist0 * wrench.view(2, 2, 6).sum(dim=1)).sum(dim=1)
    torch.testing.assert_close(
        robot_power + box_power,
        torch.zeros_like(robot_power),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )


def test_rollout_uses_explicit_initial_state_and_fixed_arm_hand_substeps():
    _, _, dynamics, problem, work = _setup()
    work.assemble(problem, dynamics)
    state = work.rollout(problem.nominal_control)
    assert state.shape == (2, 26, 110)
    assert torch.equal(state[:, 0], _optimizer_x0(problem))
    assert not torch.equal(state[:, 0, :1], problem.measured_state[:, :1])
    assert work.arm_substep_state.shape == (2, 25, 2, 98)
    assert work.hand_substep_state.shape == (2, 25, 4, 24)

    qdd = work.qdd_offset + torch.bmm(work.qdd_from_effort, problem.nominal_control[:, 0, :43, None]).squeeze(-1)
    qdd = qdd + torch.bmm(work.qdd_wrench, problem.nominal_control[:, 0, 43:, None]).squeeze(-1)
    active_qdd = torch.bmm(problem.active_selector, qdd[:, :, None]).squeeze(-1)
    for substep, elapsed in enumerate((0.02, 0.04)):
        expected_q = problem.active_q0 + elapsed * problem.active_qd0 + 0.5 * elapsed**2 * active_qdd
        expected_qd = problem.active_qd0 + elapsed * active_qdd
        torch.testing.assert_close(work.arm_substep_state[:, 0, substep, 12:55], expected_q)
        torch.testing.assert_close(work.arm_substep_state[:, 0, substep, 55:98], expected_qd)
    for substep, elapsed in enumerate((0.01, 0.02, 0.03, 0.04)):
        expected_q = problem.active_q0[:, 31:] + elapsed * problem.active_qd0[:, 31:] + 0.5 * elapsed**2 * active_qdd[:, 31:]
        expected_qd = problem.active_qd0[:, 31:] + elapsed * active_qdd[:, 31:]
        torch.testing.assert_close(work.hand_substep_state[:, 0, substep], torch.cat((expected_q, expected_qd), dim=1))


@pytest.mark.parametrize(
    "bad",
    ["left_jacobian", "right_jacobian", "input_valid", "inherited_mass", "inherited_rank"],
)
def test_assemble_rejects_only_bad_row(bad):
    _, gpu, dynamics, problem, work = _setup()
    if bad == "left_jacobian":
        problem.palm_jacobian[0, 0, 0, 0] = float("nan")
    elif bad == "right_jacobian":
        problem.palm_jacobian[0, 1, 0, 0] = float("nan")
    elif bad == "input_valid":
        problem.input_valid[0] = False
    elif bad == "inherited_mass":
        gpu["mass_matrix"][0].neg_()
        dynamics.step(**gpu)
    else:
        gpu["actuation_matrix"][0, :, 1].copy_(gpu["actuation_matrix"][0, :, 0])
        dynamics.step(**gpu)
    assert work.assemble(problem, dynamics).tolist() == [False, True]
    assert torch.isfinite(work.qdd_wrench).all()


def test_metadata_and_weight_rejection_happen_before_workspace_writes():
    _, _, dynamics, problem, work = _setup()
    work.assemble(problem, dynamics)
    before = work.state.clone()
    with pytest.raises(ValueError):
        work.assemble(replace(problem, active_q0=problem.active_q0.double()), dynamics)
    assert torch.equal(work.state, before)
    bad_weight = problem.state_weight.clone()
    bad_weight[0] = -1.0
    assert work.assemble(replace(problem, state_weight=bad_weight), dynamics).tolist() == [False, False]


@pytest.mark.parametrize(
    "bad",
    [
        "effort",
        "arm_substep",
        "hand_substep",
        "wrench_friction",
        "wrench_moment",
        "wrench_negative_overflow",
        "hard",
        "hard_negative_overflow",
        "nonfinite",
        "palm_residual",
    ],
)
def test_validation_rejects_only_bad_row_and_prior_off_zero_wrench_is_valid(bad):
    _, _, dynamics, problem, work = _setup()
    problem.nominal_control.zero_()
    work.assemble(problem, dynamics)
    state = work.rollout(problem.nominal_control)
    assert work.validate(state, problem.nominal_control).tolist() == [True, True]
    if bad == "effort":
        problem.control_upper[0, 3, 0] = -1.0
    elif bad == "arm_substep":
        # Choose qd so the t=.02 position is an extremum and the endpoint
        # returns to q0; then exclude only that intermediate extremum.
        problem.active_qd0[0, 0] = -0.02 * work._active_qdd[0, 0]
        work.assemble(problem, dynamics)
        state = work.rollout(problem.nominal_control)
        intermediate = work.arm_substep_state[0, 0, 0, 12]
        endpoint = state[0, 1, 12]
        boundary = 0.5 * (intermediate + endpoint)
        if intermediate.item() > endpoint.item():
            problem.state_upper[0, 1, 12] = boundary
        else:
            problem.state_lower[0, 1, 12] = boundary
    elif bad == "hand_substep":
        problem.active_qd0[0, 31] = -0.02 * work._active_qdd[0, 31]
        work.assemble(problem, dynamics)
        state = work.rollout(problem.nominal_control)
        intermediate = work.hand_substep_state[0, 0, 1, 0]
        endpoint = state[0, 1, 43]
        boundary = 0.5 * (intermediate + endpoint)
        if intermediate.item() > endpoint.item():
            problem.state_upper[0, 1, 43] = boundary
        else:
            problem.state_lower[0, 1, 43] = boundary
    elif bad == "wrench_friction":
        problem.wrench_inequality_matrix[0, 0, 0, 0] = 1.0
        problem.wrench_inequality_matrix[0, 0, 0, 2] = -0.5
        problem.nominal_control[0, 0, 43] = 1.0
        problem.wrench_inequality_upper[0, 0, 0] = 0.0
    elif bad == "wrench_moment":
        problem.wrench_inequality_matrix[0, 0, 0, 3] = 1.0
        problem.nominal_control[0, 0, 46] = 1.0
        problem.wrench_inequality_upper[0, 0, 0] = 0.2
    elif bad == "wrench_negative_overflow":
        problem.wrench_inequality_matrix[0, 0, 0, 0] = torch.finfo(torch.float32).max
        problem.nominal_control[0, 0, 43] = -2.0
        problem.wrench_inequality_upper[0, 0, 0] = 0.0
    elif bad == "hard":
        problem.hard_inequality_matrix[0, 0, 0, 0] = 1.0
        problem.hard_inequality_upper[0, 0, 0] = -1.0
    elif bad == "hard_negative_overflow":
        problem.hard_inequality_matrix[0, 0, 0, 0] = torch.finfo(torch.float32).max
        problem.nominal_control[0, 0, 0] = -2.0
        problem.hard_inequality_upper[0, 0, 0] = 0.0
    elif bad == "nonfinite":
        state[0, 4, 0] = float("nan")
    else:
        largest = torch.finfo(torch.float32).max
        problem.base_pose0[0, 0] = 2.0
        problem.palm_state_jacobian[0, 0, 0, 0] = largest
    if bad != "nonfinite":
        work.assemble(problem, dynamics)
        state = work.rollout(problem.nominal_control)
    assert work.validate(state, problem.nominal_control).tolist() == [False, True]


def test_dense_gauss_newton_direction_and_four_candidate_interfaces():
    _, _, dynamics, problem, work = _setup()
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    A, B = work.linearize()
    assert A.data_ptr() == work.A.data_ptr() and B.data_ptr() == work.B.data_ptr()
    direction = work.solve_direction()
    residual = torch.bmm(work.H, direction[:, :, None]).squeeze(-1) + work.gradient
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-3, rtol=2.0e-3)
    candidates = problem.nominal_control[:, None].expand(-1, 4, -1, -1).clone()
    candidate_state = work.rollout_candidates(candidates)
    assert candidate_state.shape == (2, 4, 26, 110)
    assert work.validate_candidates(candidate_state, candidates).shape == (2, 4)

    problem.control_lower.copy_(problem.nominal_control)
    problem.control_upper.copy_(problem.nominal_control)
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    work.linearize()
    assert not work.solve_direction().any()


def test_terminal_weight_adds_only_the_terminal_state_contribution():
    _, _, dynamics, problem, work = _setup(batch=1)
    problem.terminal_weight.zero_()
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    work.linearize()
    work.solve_direction()
    base_hessian = work.H.clone()
    base_gradient = work.gradient.clone()

    terminal_delta = torch.linspace(0.0, 1.0, 110, device="cuda:0")
    problem.terminal_weight.copy_(terminal_delta)
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    work.linearize()
    work.solve_direction()
    terminal_jacobian = work.state_control_jacobian[:, -1]
    terminal_residual = work.state[:, -1] - problem.state_target[:, -1]
    expected_hessian_delta = 2.0 * torch.bmm(
        terminal_jacobian.transpose(1, 2),
        terminal_jacobian * terminal_delta[None, :, None],
    )
    expected_gradient_delta = 2.0 * torch.bmm(
        terminal_jacobian.transpose(1, 2),
        (terminal_residual * terminal_delta[None])[:, :, None],
    ).squeeze(-1)
    torch.testing.assert_close(work.H - base_hessian, expected_hessian_delta, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(work.gradient - base_gradient, expected_gradient_delta, atol=2.0e-4, rtol=2.0e-4)


def test_palm_stage_zero_tracks_predicted_x1_and_responds_to_u0():
    _, _, dynamics, problem, work = _setup(batch=1)
    problem.palm_state_jacobian.zero_()
    problem.palm_state_jacobian[:, 0, 0, 0] = 1.0
    problem.palm_residual_offset[:, 0, 0] = 0.25
    problem.palm_weight.zero_()
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    work.linearize()
    work.solve_direction()
    base_hessian = work.H.clone()
    base_gradient = work.gradient.clone()

    problem.palm_weight[0] = 1.0
    work.assemble(problem, dynamics)
    work.rollout(problem.nominal_control)
    work.linearize()
    work.solve_direction()
    expected_residual = 0.25 + work.state[:, 1, 0]
    old_node_residual = 0.25 + work.state[:, 0, 0]
    torch.testing.assert_close(work._palm_residual[:, 0, 0], expected_residual)
    assert not torch.equal(expected_residual, old_node_residual)
    stage_jacobian = work.state_control_jacobian[:, 0, 0]
    expected_hessian_delta = 2.0 * stage_jacobian[:, :, None] * stage_jacobian[:, None, :]
    expected_gradient_delta = 2.0 * stage_jacobian * expected_residual[:, None]
    torch.testing.assert_close(work.H - base_hessian, expected_hessian_delta, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(work.gradient - base_gradient, expected_gradient_delta, atol=2.0e-4, rtol=2.0e-4)


def test_resident_methods_keep_storage_and_allocator_stable_after_warmup():
    _, _, dynamics, problem, work = _setup(batch=1)
    candidates = problem.nominal_control[:, None].expand(-1, 4, -1, -1).clone()

    def resident_cycle():
        work.assemble(problem, dynamics)
        work.rollout(problem.nominal_control)
        work.rollout_candidates(candidates)
        work.linearize()
        work.solve_direction()
        work.validate(work.state, problem.nominal_control)
        work.validate_candidates(work.candidate_state, candidates)

    for _ in range(10):
        resident_cycle()
    torch.cuda.synchronize()
    pointers = _storage_pointers(vars(work))
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    known = _storage_pointers((vars(work), vars(work._finite), vars(dynamics), vars(dynamics._finite), vars(problem), candidates))
    with _NoNewStorage(known):
        for _ in range(100):
            resident_cycle()
    torch.cuda.synchronize()
    assert pointers == _storage_pointers(vars(work))
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= RESERVED_MEMORY_DELTA_BYTES
    print(f"CUDA0 coupled LQ storages={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")
