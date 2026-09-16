import warnings

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.line_search import (
    ParallelLineSearch,
)
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.lq_problem import (
    CoupledLqWorkspace,
    CoupledRtiInput,
)
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.reduced_dynamics import (
    GpuReducedDynamics,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")
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


def _problem(batch, wrench_constraints=1, hard_constraints=1):
    device = torch.device("cuda:0")
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
    selector = z(batch, 43, 59)
    selector[:, torch.arange(43), torch.arange(6, 49)] = 1.0
    palms = z(batch, 2, 6, 59)
    palms[:, :, torch.arange(6), torch.arange(6)] = 1.0
    box_map = z(batch, 6, 12)
    box_map[:, torch.arange(6), torch.arange(6)] = 1.0
    box_map[:, torch.arange(6), torch.arange(6) + 6] = 1.0
    nominal = z(batch, 25, 55)
    target = z(batch, 26, 110)
    return CoupledRtiInput(
        measured_state=z(batch, 111),
        base_pose0=z(batch, 6),
        base_twist0=z(batch, 6),
        base_quaternion0=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch, -1).clone(),
        active_q0=z(batch, 43),
        active_qd0=z(batch, 43),
        box_pose0=z(batch, 6),
        box_twist0=z(batch, 6),
        active_selector=selector,
        palm_jacobian=palms,
        box_wrench_map=box_map,
        box_gravity=z(batch, 6),
        nominal_control=nominal,
        control_lower=torch.full_like(nominal, -10.0),
        control_upper=torch.full_like(nominal, 10.0),
        state_target=target,
        state_weight=torch.ones(110, dtype=torch.float32, device=device),
        control_weight=torch.zeros(55, dtype=torch.float32, device=device),
        terminal_weight=torch.zeros(110, dtype=torch.float32, device=device),
        palm_residual_offset=z(batch, 25, 12),
        palm_state_jacobian=z(batch, 25, 12, 110),
        palm_weight=torch.zeros(12, dtype=torch.float32, device=device),
        state_lower=torch.full_like(target, -100.0),
        state_upper=torch.full_like(target, 100.0),
        wrench_inequality_matrix=z(batch, 25, wrench_constraints, 12),
        wrench_inequality_upper=torch.full(
            (batch, 25, wrench_constraints), 100.0, dtype=torch.float32, device=device
        ),
        hard_inequality_matrix=z(batch, 25, hard_constraints, 55),
        hard_inequality_upper=torch.full(
            (batch, 25, hard_constraints), 100.0, dtype=torch.float32, device=device
        ),
        input_valid=torch.ones(batch, dtype=torch.bool, device=device),
    )


def _setup(batch, wrench_constraints=1, hard_constraints=1):
    generator = torch.Generator().manual_seed(23)
    raw = torch.randn(batch, 59, 59, dtype=torch.float64, generator=generator)
    mass = raw @ raw.transpose(1, 2) / 59.0 + 2.0 * torch.eye(59, dtype=torch.float64)
    selection = torch.zeros(batch, 59, 43, dtype=torch.float64)
    selection[:, torch.arange(6, 49), torch.arange(43)] = 1.0
    wheel = torch.zeros(batch, 12, 59, dtype=torch.float64)
    wheel[:, torch.arange(12), torch.arange(47, 59)] = 1.0
    dynamics = GpuReducedDynamics(batch=batch)
    dynamics.step(
        mass_matrix=mass.to("cuda:0", torch.float32),
        bias=torch.zeros(batch, 59, device="cuda:0"),
        actuation_matrix=selection.to("cuda:0", torch.float32),
        wheel_contact_jacobian=wheel.to("cuda:0", torch.float32),
        wheel_contact_bias=torch.zeros(batch, 12, device="cuda:0"),
    )
    problem = _problem(batch, wrench_constraints, hard_constraints)
    workspace = CoupledLqWorkspace(
        batch=batch,
        max_wrench_constraints=wrench_constraints,
        max_hard_constraints=hard_constraints,
    )
    return dynamics, problem, workspace


def _stage_target_from_alpha(workspace, problem, direction, target_alpha):
    target_control = problem.nominal_control + target_alpha[:, None, None] * direction
    target_control.clamp_(min=-10.0, max=10.0)
    if target_control.shape[0] > 1:
        target_control[1, :, 1] = 0.0
        target_control[1, :, 2].clamp_(max=0.3)
    if target_control.shape[0] > 6:
        target_control[6].zero_()
    problem.state_target.copy_(workspace.rollout(target_control))


def test_four_candidates_select_atomically_over_the_complete_horizon():
    dynamics, problem, workspace = _setup(batch=8)
    direction = torch.zeros(8, 25, 55, dtype=torch.float32, device="cuda:0")
    direction[0, :, 0] = 1.0  # alpha 1 and .5 clamp to the same optimum.
    direction[1, :, 0] = 1.0  # alpha 1 violates the node-0 hard bound.
    direction[1, :, 1] = 5.0  # equality-bound element must stay nominal.
    direction[1, :, 2] = 1.0  # independently clamped upper bound.
    direction[2, :, 0] = 1.0  # every candidate violates the node-0 hard bound.
    direction[3, :, 0] = 1.0  # replaced with NaN after target staging.
    direction[4, 24, 0] = 1.0  # only node 24 fails for alpha 1.
    direction[5, :, 43] = 1.0  # only the left-wrench inequality fails.
    direction[6, :, 0] = 3.0  # all controls are equality-bound.
    direction[7, :, 0] = 1.0  # all candidates valid but worse than nominal.

    problem.control_upper[0, :, 0] = 0.5
    problem.control_lower[1, :, 1] = 0.0
    problem.control_upper[1, :, 1] = 0.0
    problem.control_upper[1, :, 2] = 0.3
    problem.control_lower[6].zero_()
    problem.control_upper[6].zero_()
    problem.hard_inequality_matrix[1, 0, 0, 0] = 1.0
    problem.hard_inequality_upper[1, 0, 0] = 0.75
    problem.hard_inequality_matrix[2, 0, 0, 0] = 1.0
    problem.hard_inequality_upper[2, 0, 0] = 0.1
    problem.hard_inequality_matrix[4, 24, 0, 0] = 1.0
    problem.hard_inequality_upper[4, 24, 0] = 0.75
    problem.wrench_inequality_matrix[5, 0, 0, 0] = 1.0
    problem.wrench_inequality_upper[5, 0, 0] = 0.75

    assert workspace.assemble(problem, dynamics).tolist() == [True] * 8
    target_alpha = torch.tensor([0.5, 0.5, 0.125, 0.5, 0.5, 0.5, 0.0, 0.0], device="cuda:0")
    _stage_target_from_alpha(workspace, problem, direction, target_alpha)
    assert workspace.assemble(problem, dynamics).tolist() == [True] * 8
    direction[3].fill_(float("nan"))
    workspace.direction.copy_(direction.flatten(1))

    search = ParallelLineSearch(batch=8, max_wrench_constraints=1, max_hard_constraints=1)
    result = search.step(workspace)

    assert search.candidate_control.shape == (8, 4, 25, 55)
    assert search.candidate_control.dtype == torch.float32
    assert search.candidate_control.device == torch.device("cuda:0")
    assert search.alphas.tolist() == [1.0, 0.5, 0.25, 0.125]
    assert result.alpha_index.tolist() == [0, 1, -1, -1, 1, 1, 0, -1]
    assert result.accepted.tolist() == [True, True, False, False, True, True, True, False]
    assert workspace.candidate_valid[1].tolist() == [False, True, True, True]
    assert workspace.candidate_valid[2].tolist() == [False, False, False, False]
    assert workspace.candidate_valid[3].tolist() == [False, False, False, False]
    assert workspace.candidate_valid[4].tolist() == [False, True, True, True]
    assert workspace.candidate_valid[5].tolist() == [False, True, True, True]
    for row, candidate in ((0, 0), (1, 1), (4, 1), (5, 1)):
        assert torch.equal(result.control[row], search.candidate_control[row, candidate])
        assert torch.equal(result.state[row], workspace.candidate_state[row, candidate])
    assert torch.equal(search.candidate_control[1, :, :, 1], torch.zeros(4, 25, device="cuda:0"))
    torch.testing.assert_close(
        search.candidate_control[1, :, 0, 2],
        torch.tensor([0.3, 0.3, 0.25, 0.125], device="cuda:0"),
    )

    rejected = torch.tensor([2, 3, 7], device="cuda:0")
    torch.testing.assert_close(result.control[rejected], workspace.nominal_control[rejected])
    torch.testing.assert_close(result.state[rejected], workspace.state[rejected])
    assert torch.isinf(result.merit[rejected]).all()
    assert torch.equal(result.control[6], workspace.nominal_control[6])
    assert torch.equal(result.state[6], workspace.state[6])
    assert torch.isfinite(result.merit[6])
    assert torch.equal(result.control[5, :, 49:], workspace.nominal_control[5, :, 49:])

    workspace.state_upper[6, 0, 0] = -1.0
    result = search.step(workspace)
    assert not result.accepted[6]
    assert result.alpha_index[6] == -1
    assert torch.isinf(result.merit[6])
    assert torch.equal(result.control[6], workspace.nominal_control[6])
    assert torch.equal(result.state[6], workspace.state[6])


def test_merit_matches_the_complete_soft_quadratic_objective():
    dynamics, problem, workspace = _setup(batch=1)
    problem.state_target.copy_(
        torch.linspace(-0.2, 0.3, 26 * 110, device="cuda:0").view(1, 26, 110)
    )
    problem.state_weight.copy_(torch.linspace(0.1, 1.1, 110, device="cuda:0"))
    problem.control_weight.copy_(torch.linspace(0.05, 0.2, 55, device="cuda:0"))
    problem.terminal_weight.copy_(torch.linspace(0.2, 0.8, 110, device="cuda:0"))
    problem.palm_state_jacobian[:, :, torch.arange(12), torch.arange(12)] = 0.25
    problem.palm_residual_offset.copy_(
        torch.linspace(-0.1, 0.1, 25 * 12, device="cuda:0").view(1, 25, 12)
    )
    problem.palm_weight.copy_(torch.linspace(0.3, 0.9, 12, device="cuda:0"))
    assert workspace.assemble(problem, dynamics).tolist() == [True]
    workspace.direction.copy_(
        torch.linspace(-0.4, 0.4, 25 * 55, device="cuda:0").view(1, 25 * 55)
    )
    search = ParallelLineSearch(batch=1, max_wrench_constraints=1, max_hard_constraints=1)
    search.step(workspace)

    candidate_state = workspace.candidate_state
    expected_candidate = (
        (candidate_state[:, :, 1:] - workspace.state_target[:, None, 1:]).square()
        * workspace.state_weight[None, None, None]
    ).sum(dim=(2, 3))
    expected_candidate += (
        (candidate_state[:, :, -1] - workspace.state_target[:, None, -1]).square()
        * workspace.terminal_weight[None, None]
    ).sum(dim=2)
    expected_candidate += (
        search.candidate_control.square() * workspace.control_weight[None, None, None]
    ).sum(dim=(2, 3))
    candidate_palm = torch.einsum(
        "bcti,btji->bctj",
        candidate_state[:, :, 1:],
        workspace.palm_state_jacobian,
    ) + workspace.palm_residual_offset[:, None]
    expected_candidate += (candidate_palm.square() * workspace.palm_weight[None, None, None]).sum(dim=(2, 3))

    nominal_state = workspace.state
    expected_nominal = (
        (nominal_state[:, 1:] - workspace.state_target[:, 1:]).square()
        * workspace.state_weight[None, None]
    ).sum(dim=(1, 2))
    expected_nominal += (
        (nominal_state[:, -1] - workspace.state_target[:, -1]).square()
        * workspace.terminal_weight[None]
    ).sum(dim=1)
    expected_nominal += (
        workspace.nominal_control.square() * workspace.control_weight[None, None]
    ).sum(dim=(1, 2))
    nominal_palm = torch.einsum(
        "bti,btji->btj",
        nominal_state[:, 1:],
        workspace.palm_state_jacobian,
    ) + workspace.palm_residual_offset
    expected_nominal += (nominal_palm.square() * workspace.palm_weight[None, None]).sum(dim=(1, 2))

    torch.testing.assert_close(search.merit, expected_candidate)
    torch.testing.assert_close(search.nominal_merit, expected_nominal)


def test_step_reuses_all_cuda_storage_after_warmup():
    dynamics, problem, workspace = _setup(batch=1)
    assert workspace.assemble(problem, dynamics).tolist() == [True]
    direction = torch.zeros(1, 25, 55, dtype=torch.float32, device="cuda:0")
    direction[:, :, 0] = 1.0
    _stage_target_from_alpha(
        workspace,
        problem,
        direction,
        torch.tensor([0.5], dtype=torch.float32, device="cuda:0"),
    )
    assert workspace.assemble(problem, dynamics).tolist() == [True]
    workspace.direction.copy_(direction.flatten(1))
    search = ParallelLineSearch(batch=1, max_wrench_constraints=1, max_hard_constraints=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(10):
            search.step(workspace)
    torch.cuda.synchronize()
    pointers = _storage_pointers(vars(search))
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    known = _storage_pointers(
        (vars(search), vars(workspace), vars(workspace._finite), vars(dynamics), vars(dynamics._finite), vars(problem))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _NoNewStorage(known):
            for _ in range(100):
                result = search.step(workspace)
    torch.cuda.synchronize()
    assert result.control.data_ptr() == search.result_control.data_ptr()
    assert result.state.data_ptr() == search.result_state.data_ptr()
    assert pointers == _storage_pointers(vars(search))
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= RESERVED_MEMORY_DELTA_BYTES
    print(f"CUDA0 line search storages={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")
