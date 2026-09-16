import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.state_adapter import GpuStateAdapter
from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import pack_state_features
from test_m1_dual_panda_o6_contracts import _snapshot
from test_m1_bimanual_reduced_dynamics import _state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA0")


def test_cpu_upload_matches_authoritative_feature_order_and_dynamics():
    snapshots = [_snapshot(10), _snapshot(20)]
    for row, snap in enumerate(snapshots):
        for index, field in enumerate((snap.base_state, snap.m1_q, snap.m1_qd,
                snap.platform_q_qd, snap.left_arm.q, snap.left_arm.qd,
                snap.right_arm.q, snap.right_arm.qd, snap.left_hand.q,
                snap.left_hand.qd, snap.right_hand.q, snap.right_hand.qd,
                snap.box.pose_b, snap.box.twist_b)):
            field.fill_(index + row * 20)
    mirror = GpuStateAdapter(batch=2)
    mirror.upload_cpu(snapshots, [_state(mimic_coupling=.3), _state()])
    assert torch.equal(mirror.features.cpu(), torch.stack([pack_state_features(s) for s in snapshots]))
    assert mirror.features.dtype == torch.float32 and mirror.features.device == torch.device("cuda:0")
    assert mirror.valid.tolist() == [True, True]
    assert mirror.timestamps_ns.tolist() == [10, 20]
    assert torch.equal(mirror.mass_matrix[0].cpu(), _state(mimic_coupling=.3).mass_matrix.float())


def test_borrowed_cpu_fields_rechecked_and_invalid_row_preserves_latest_state():
    mirror = GpuStateAdapter(batch=2)
    mirror.upload_cpu([_snapshot(10), _snapshot(10)], [_state(), _state()])
    old = mirror.features.clone()
    bad = _snapshot(20)
    bad.left_hand.q[0] = float("nan")
    mirror.upload_cpu([bad, _snapshot(20)], [_state(), _state()])
    assert mirror.valid.tolist() == [False, True]
    assert torch.equal(mirror.features[0], old[0])
    assert mirror.timestamps_ns.tolist() == [10, 20]
    bad_dynamics = _state()
    bad_dynamics.bias[0] = float("inf")
    mirror.upload_cpu([_snapshot(30), _snapshot(30)], [bad_dynamics, _state()])
    assert mirror.valid.tolist() == [False, True]


@pytest.mark.parametrize("replacement", [lambda: torch.zeros(2, 110), lambda: torch.zeros(2, 111), lambda: torch.zeros(2, 111, dtype=torch.float64, device="cuda:0")])
def test_gpu_step_rejects_metadata_before_publication(replacement):
    mirror = GpuStateAdapter(batch=2)
    mirror.upload_cpu([_snapshot(), _snapshot()], [_state(), _state()])
    old = mirror.features.clone()
    with pytest.raises(ValueError):
        mirror.step(replacement(), mirror.timestamps_ns, **mirror.dynamics_inputs())
    assert torch.equal(old, mirror.features)


def test_gpu_step_timestamp_numeric_isolation_and_reset():
    mirror = GpuStateAdapter(batch=2)
    mirror.upload_cpu([_snapshot(), _snapshot()], [_state(), _state()])
    features = mirror.features.clone().add_(1)
    features[0, 0] = float("inf")
    stamps = torch.tensor([20, 20], dtype=torch.int64, device="cuda:0")
    mirror.step(features, stamps, **mirror.dynamics_inputs())
    assert mirror.valid.tolist() == [False, True]
    assert mirror.features[0, 0] == 0 and mirror.features[1, 0] == 1
    mask = torch.tensor([False, True], device="cuda:0")
    mirror.reset(mask)
    assert mirror.valid.tolist() == [False, False]
    assert mirror.timestamps_ns.tolist() == [10, 0]
    features.zero_()
    mirror.step(features, stamps, **mirror.dynamics_inputs())
    assert mirror.valid.tolist() == [True, True]
    mirror.step(features, stamps, **mirror.dynamics_inputs())
    assert mirror.valid.tolist() == [False, False]


def test_exact_nanosecond_timestamp_precision_and_state_memory_guard():
    mirror = GpuStateAdapter(batch=2)
    stamp = 1_790_000_000_000_000_001
    mirror.upload_cpu([_snapshot(stamp), _snapshot(stamp + 1)], [_state(), _state()])
    assert mirror.timestamps_ns.tolist() == [stamp, stamp + 1]
    features = mirror.features.clone()
    stamps = mirror.timestamps_ns.clone()
    inputs = mirror.dynamics_inputs()
    from test_m1_bimanual_gpu_rti_dynamics import storage_pointers, NoNewStorage
    known = storage_pointers((vars(mirror), vars(mirror._finite), features, stamps))
    for _ in range(10):
        stamps.add_(1)
        mirror.step(features, stamps, **inputs)
    torch.cuda.synchronize()
    pointers = storage_pointers((vars(mirror), vars(mirror._finite)))
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    with NoNewStorage(known):
        for _ in range(100):
            stamps.add_(1)
            mirror.step(features, stamps, **inputs)
    torch.cuda.synchronize()
    assert pointers == storage_pointers((vars(mirror), vars(mirror._finite)))
    assert torch.cuda.memory_allocated() == allocated
    delta = torch.cuda.memory_reserved() - reserved
    assert delta <= 2 * 1024 * 1024
    print(f"CUDA0 state storages={len(pointers)} stable; allocated_delta=0; reserved_delta={delta}")
