import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.reduced_dynamics import GpuReducedDynamics
from go2_pvcnn.control.m1_bimanual_coordination.reduced_dynamics import condense_constrained_dynamics
from go2_pvcnn.control.m1_bimanual_coordination.constraints import effort_limits
from test_m1_bimanual_reduced_dynamics import _state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")
PARITY_ATOL = 3.e-5
PARITY_RTOL = 3.e-5
RESERVED_MEMORY_DELTA_BYTES = 2 * 1024 * 1024


def storage_pointers(values):
    leaves, _ = tree_flatten(values)
    return {value.untyped_storage().data_ptr() for value in leaves if isinstance(value, torch.Tensor)}


class NoNewStorage(TorchDispatchMode):
    def __init__(self, known):
        self.known = known

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert "_local_scalar_dense" not in str(func), f"host scalar extraction: {func}"
        output = func(*args, **(kwargs or {}))
        assert storage_pointers(output) <= self.known, f"new tensor storage: {func}"
        return output


def test_all_workspaces_keep_float32_independent_of_global_default():
    from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.state_adapter import GpuStateAdapter
    from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.warm_start import GpuWarmStart
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for cls in (GpuStateAdapter, GpuReducedDynamics, GpuWarmStart):
            work = cls(batch=2)
            leaves, _ = tree_flatten((vars(work), vars(work._finite)))
            for value in leaves:
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    assert value.dtype == torch.float32
    finally:
        torch.set_default_dtype(old)


def inputs():
    states = [_state(mimic_coupling=.3), _state(mimic_coupling=.7)]
    generator = torch.Generator().manual_seed(42)
    for state in states:
        coupling = torch.randn(59, 59, dtype=torch.float64, generator=generator) * .02
        state.mass_matrix.copy_(coupling @ coupling.T + state.mass_matrix)
    tensors = {name: torch.stack([getattr(s, name) for s in states]).to("cuda:0", torch.float32)
               for name in ("mass_matrix", "bias", "actuation_matrix", "wheel_contact_jacobian", "wheel_contact_bias")}
    return states, tensors


def test_kkt_parity_coupled_spd_mass_and_exact_43_effort_reconstruction():
    states, values = inputs()
    work = GpuReducedDynamics(batch=2)
    work.step(**values)
    assert work.kkt.shape == (2, 71, 71) and work.rhs.shape == (2, 71, 44)
    assert work.valid.tolist() == [True, True]
    # Exercise both active bounds from the unchanged public 43-channel policy.
    limits = effort_limits()
    effort = torch.stack((-limits, limits)).to("cuda:0", torch.float32)
    work.reconstruct(effort)
    for row, state in enumerate(states):
        reference = condense_constrained_dynamics(state)
        for name in ("qdd_offset", "qdd_from_effort", "contact_offset", "contact_from_effort"):
            torch.testing.assert_close(getattr(work, name)[row].cpu().double(), getattr(reference, name), atol=PARITY_ATOL, rtol=PARITY_RTOL)
        torch.testing.assert_close(work.qdd[row].cpu().double(), reference.qdd_offset + reference.qdd_from_effort @ effort[row].cpu().double(), atol=PARITY_ATOL, rtol=PARITY_RTOL)
        torch.testing.assert_close(work.contact[row].cpu().double(), reference.contact_offset + reference.contact_from_effort @ effort[row].cpu().double(), atol=PARITY_ATOL, rtol=PARITY_RTOL)
        torch.testing.assert_close(work.generalized_effort[row].cpu().double(), state.actuation_matrix @ effort[row].cpu().double(), atol=0, rtol=0)
    with pytest.raises(ValueError):
        work.reconstruct(torch.zeros(2, 59, device="cuda:0"))


@pytest.mark.parametrize("bad", ["singular", "nonfinite", "input_invalid"])
def test_invalid_numeric_or_singular_row_does_not_poison_latest_maps(bad):
    _, values = inputs()
    work = GpuReducedDynamics(batch=2)
    work.step(**values)
    previous = work.qdd_from_effort.clone()
    values["bias"][1].add_(.1)
    valid = torch.ones(2, dtype=torch.bool, device="cuda:0")
    if bad == "singular":
        values["wheel_contact_jacobian"][0].zero_()
    elif bad == "nonfinite":
        values["mass_matrix"][0, 0, 0] = float("nan")
    else:
        valid[0] = False
    work.step(**values, input_valid=valid)
    assert work.valid.tolist() == [False, True]
    assert torch.equal(work.qdd_from_effort[0], previous[0])
    assert torch.isfinite(work.qdd_from_effort).all()
    work.reconstruct(torch.zeros(2, 43, device="cuda:0"))
    assert work.output_valid.tolist() == [False, True]


def test_metadata_rejection_is_atomic():
    _, values = inputs()
    work = GpuReducedDynamics(batch=2)
    work.step(**values)
    previous = work.qdd_offset.clone()
    values["actuation_matrix"] = torch.zeros(2, 59, 59, device="cuda:0")
    with pytest.raises(ValueError):
        work.step(**values)
    assert torch.equal(previous, work.qdd_offset)


def test_pointers_and_allocator_memory_stable_after_warmup():
    _, values = inputs()
    work = GpuReducedDynamics(batch=2)
    effort = torch.zeros(2, 43, device="cuda:0")
    for _ in range(10):
        work.step(**values)
        work.reconstruct(effort)
    torch.cuda.synchronize()
    pointers = {name: value.data_ptr() for name, value in vars(work).items() if isinstance(value, torch.Tensor)}
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    known = storage_pointers((vars(work), vars(work._finite), values, effort))
    with NoNewStorage(known):
        for _ in range(100):
            work.step(**values)
            work.reconstruct(effort)
    torch.cuda.synchronize()
    assert pointers == {name: value.data_ptr() for name, value in vars(work).items() if isinstance(value, torch.Tensor)}
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= RESERVED_MEMORY_DELTA_BYTES
    print(f"CUDA0 dynamics pointers={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")


def test_reconstruct_nonfinite_effort_isolated_and_latest_output_preserved():
    _, values = inputs()
    work = GpuReducedDynamics(batch=2)
    work.step(**values)
    effort = torch.ones(2, 43, device="cuda:0")
    work.reconstruct(effort)
    previous = work.qdd.clone()
    effort[0, 0] = float("nan")
    work.reconstruct(effort)
    assert work.output_valid.tolist() == [False, True]
    assert torch.equal(work.qdd[0], previous[0])


@pytest.mark.parametrize("bad", ["asymmetric", "non_spd", "rank_deficient", "near_dependent"])
def test_direct_cuda_inputs_preserve_full_dynamics_invariants(bad):
    _, values = inputs()
    work = GpuReducedDynamics(batch=2)
    work.step(**values)
    previous = work.qdd_from_effort.clone()
    if bad == "asymmetric":
        values["mass_matrix"][0, 0, 1] += .2
    elif bad == "non_spd":
        values["mass_matrix"][0].neg_()
    else:
        values["actuation_matrix"][0, :, 1].copy_(values["actuation_matrix"][0, :, 0])
        if bad == "near_dependent":
            values["actuation_matrix"][0, 7, 1] = 1.e-8
    work.step(**values)
    assert work.valid.tolist() == [False, True]
    assert torch.equal(work.qdd_from_effort[0], previous[0])
