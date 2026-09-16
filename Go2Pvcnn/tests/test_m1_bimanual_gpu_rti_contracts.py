from dataclasses import FrozenInstanceError

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti import GpuRtiCfg, GpuRtiTrajectory
from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.config import resolve_backend_selection


def test_frozen_exact_config():
    cfg = GpuRtiCfg()
    assert (cfg.horizon, cfg.action_dim, cfg.padded_horizon, cfg.rti_iterations) == (25, 43, 32, 1)
    assert cfg.line_search_alphas == (1.0, 0.5, 0.25, 0.125)
    assert cfg.dtype == torch.float32 and cfg.device == "cuda:0"
    with pytest.raises(FrozenInstanceError):
        cfg.horizon = 32


@pytest.mark.parametrize("kwargs", [{"horizon": 32}, {"action_dim": 59}, {"padded_horizon": 25}, {"rti_iterations": 2}, {"dtype": torch.float64}, {"device": "cpu"}, {"line_search_alphas": (1.0,)}])
def test_config_rejects_contract_drift(kwargs):
    with pytest.raises(ValueError):
        GpuRtiCfg(**kwargs)


def test_cpu_leakage_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        GpuRtiTrajectory(torch.zeros(2, 25, 43), torch.zeros(2, dtype=torch.bool), torch.zeros(2))
    with pytest.raises(ValueError, match="CUDA"):
        GpuRtiTrajectory.zeros(batch=2, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
def test_gpu_trajectory_contract_is_private_float32_cuda():
    traj = GpuRtiTrajectory.zeros(batch=2, device="cuda:0")
    assert traj.action.shape == (2, 25, 43)
    assert traj.action.dtype == torch.float32 and traj.action.device.type == "cuda"
    assert traj.accepted.shape == traj.merit.shape == (2,)
    assert traj.accepted.dtype == torch.bool and traj.merit.dtype == torch.float32
    assert traj.accepted.device == traj.merit.device == traj.action.device
    assert not traj.action.any() and not traj.accepted.any() and not traj.merit.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA")
@pytest.mark.parametrize("field,replacement", [("action", lambda: torch.zeros(2, 32, 43, device="cuda:0")), ("action", lambda: torch.zeros(2, 25, 59, device="cuda:0")), ("action", lambda: torch.zeros(2, 25, 43, dtype=torch.float64, device="cuda:0")), ("accepted", lambda: torch.zeros(1, dtype=torch.bool, device="cuda:0")), ("accepted", lambda: torch.zeros(2, device="cuda:0")), ("merit", lambda: torch.zeros(2)), ("merit", lambda: torch.zeros(2, dtype=torch.float64, device="cuda:0"))])
def test_trajectory_rejects_invalid_fields(field, replacement):
    traj = GpuRtiTrajectory.zeros(batch=2, device="cuda:0")
    fields = dict(action=traj.action, accepted=traj.accepted, merit=traj.merit)
    fields[field] = replacement()
    with pytest.raises(ValueError):
        GpuRtiTrajectory(**fields)


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_invalid_batch_rejected_before_allocation(batch):
    with pytest.raises(ValueError, match="batch"):
        GpuRtiTrajectory.zeros(batch=batch, device="cuda:0")


def test_cpu_selection_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU selection probed CUDA"))
    selection = resolve_backend_selection("reference-cpu", "reference-cpu", "cuda:0")
    assert selection == dict(mpc_backend_requested="reference-cpu", mpc_backend_actual="reference-cpu", qp_backend_requested="reference-cpu", qp_backend_actual="reference-cpu", control_device_requested="cuda:0", control_device_actual="cpu")


@pytest.mark.parametrize("mpc,qp", [("bimanual-rti-cuda", "reference-cpu"), ("reference-cpu", "osqp-cuda")])
@pytest.mark.parametrize("available", [False, True])
def test_unimplemented_cuda_never_silently_falls_back(monkeypatch, mpc, qp, available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    with pytest.raises(ValueError, match="not implemented"):
        resolve_backend_selection(mpc, qp, "cuda:0")


@pytest.mark.parametrize("mpc,qp", [("auto", "reference-cpu"), ("reference-cpu", "auto")])
def test_auto_rejects_missing_benchmarked_manifest(mpc, qp):
    with pytest.raises(ValueError, match="benchmarked.*manifest"):
        resolve_backend_selection(mpc, qp, "cuda:0")


def test_unknown_backends_rejected():
    with pytest.raises(ValueError):
        resolve_backend_selection("invented", "reference-cpu", "cuda:0")
