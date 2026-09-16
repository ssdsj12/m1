import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.warm_start import GpuWarmStart

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")


def fixture():
    work = GpuWarmStart(batch=3, state_dim=111)
    action = torch.arange(3 * 25 * 43, device="cuda:0", dtype=torch.float32).reshape(3, 25, 43)
    state = torch.arange(3 * 26 * 111, device="cuda:0", dtype=torch.float32).reshape(3, 26, 111)
    accepted = torch.ones(3, device="cuda:0", dtype=torch.bool)
    identity = torch.tensor([[1, 2, 3], [4, 2, 3], [7, 2, 3]], device="cuda:0")
    return work, action, state, accepted, identity


def test_exact_shift_terminal_hold_and_measured_node_zero():
    work, action, state, accepted, identity = fixture()
    measured = torch.full((3, 111), -1., device="cuda:0")
    work.accept(action, state, accepted, identity)
    work.step(measured, identity)
    assert work.valid.tolist() == [True] * 3
    assert torch.equal(work.shifted_action[:, :-1], action[:, 1:])
    assert torch.equal(work.shifted_action[:, -1], action[:, -1])
    assert torch.equal(work.shifted_state[:, 1:-1], state[:, 2:])
    assert torch.equal(work.shifted_state[:, -1], state[:, -1])
    assert torch.equal(work.shifted_state[:, 0], measured)
    assert torch.equal(work.accepted_action, action)


@pytest.mark.parametrize("column", [0, 1, 2])
def test_identity_layout_phase_mismatch_invalidates_only_affected_row(column):
    work, action, state, accepted, identity = fixture()
    work.accept(action, state, accepted, identity)
    identity[1, column] += 1
    work.step(state[:, 0], identity)
    assert work.valid.tolist() == [True, False, True]
    assert work.accepted_valid.tolist() == [True, False, True]
    identity[1, column] -= 1
    work.step(state[:, 0], identity)
    assert work.valid.tolist() == [True, False, True]


def test_rejected_nonfinite_uninitialized_and_reset_rows_never_claim_horizon():
    work, action, state, accepted, identity = fixture()
    work.step(state[:, 0], identity)
    assert not work.valid.any()
    accepted[0] = False
    action[1, 0, 0] = float("nan")
    work.accept(action, state, accepted, identity)
    work.step(state[:, 0], identity)
    assert work.valid.tolist() == [False, False, True]
    reset = torch.tensor([False, False, True], device="cuda:0")
    work.step(state[:, 0], identity, reset=reset)
    assert not work.valid.any() and not work.accepted_valid.any()


def test_aliasing_stage_and_persistent_pointer_memory_guard():
    work, action, state, accepted, identity = fixture()
    work.accept(action, state, accepted, identity)
    work.step(state[:, 0], identity)
    expected = work.shifted_action.clone()
    work.accept(work.shifted_action, work.shifted_state, accepted, identity)
    assert torch.equal(work.accepted_action, expected)
    measured = work.accepted_state[:, 0]
    for _ in range(10):
        work.step(measured, identity)
    torch.cuda.synchronize()
    pointers = {name: value.data_ptr() for name, value in vars(work).items() if isinstance(value, torch.Tensor)}
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    from test_m1_bimanual_gpu_rti_dynamics import storage_pointers, NoNewStorage
    known = storage_pointers((vars(work), vars(work._finite), accepted, identity))
    with NoNewStorage(known):
        for _ in range(100):
            work.step(measured, identity)
            work.accept(work.shifted_action, work.shifted_state, accepted, identity)
    torch.cuda.synchronize()
    assert pointers == {name: value.data_ptr() for name, value in vars(work).items() if isinstance(value, torch.Tensor)}
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= 2 * 1024 * 1024
    print(f"CUDA0 warm pointers={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")


def test_malformed_input_does_not_overwrite_accepted_horizon():
    work, action, state, accepted, identity = fixture()
    work.accept(action, state, accepted, identity)
    with pytest.raises(ValueError):
        work.accept(action.double(), state, accepted, identity)
    assert torch.equal(work.accepted_action, action)


def test_invalid_measured_row_cannot_publish_candidate_but_neighbors_shift():
    work, action, state, accepted, identity = fixture()
    work.accept(action, state, accepted, identity)
    measured = state[:, 0].clone()
    measured[1, 0] = float("inf")
    work.step(measured, identity)
    assert work.valid.tolist() == [True, False, True]
    assert torch.isfinite(work.shifted_state).all()
