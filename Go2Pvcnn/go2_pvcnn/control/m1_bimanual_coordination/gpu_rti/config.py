"""Frozen private RTI configuration and fail-closed backend selection."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuRtiCfg:
    horizon: int = 25
    action_dim: int = 43
    padded_horizon: int = 32
    rti_iterations: int = 1
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    dtype: torch.dtype = torch.float32
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if (self.horizon, self.action_dim, self.padded_horizon, self.rti_iterations) != (25, 43, 32, 1):
            raise ValueError("GPU RTI requires horizon=25, action_dim=43, padded_horizon=32, rti_iterations=1")
        if self.line_search_alphas != (1.0, 0.5, 0.25, 0.125):
            raise ValueError("GPU RTI requires the frozen line-search alphas")
        if self.dtype != torch.float32:
            raise ValueError("GPU RTI requires float32")
        if torch.device(self.device).type != "cuda":
            raise ValueError("GPU RTI requires a CUDA device")


def resolve_backend_selection(mpc_backend: str, qp_backend: str, control_device: str) -> dict[str, str]:
    """Resolve only implemented backends, before prior workers or Isaac startup.

    Hardware availability cannot qualify a missing implementation. There is no
    benchmark manifest yet, so auto deliberately has no selectable backend.
    CPU reference control never consults or initializes the CUDA runtime.
    """
    if mpc_backend not in {"reference-cpu", "bimanual-rti-cuda", "auto"}:
        raise ValueError(f"unknown MPC backend: {mpc_backend}")
    if qp_backend not in {"reference-cpu", "osqp-cuda", "auto"}:
        raise ValueError(f"unknown QP backend: {qp_backend}")
    if "auto" in (mpc_backend, qp_backend):
        raise ValueError("auto requires a benchmarked, manifest-recorded backend; no benchmark manifest is available")
    if mpc_backend == "bimanual-rti-cuda":
        raise ValueError("bimanual-rti-cuda is not implemented; select reference-cpu explicitly")
    if qp_backend == "osqp-cuda":
        raise ValueError("osqp-cuda is not implemented; select reference-cpu explicitly")
    # Validate device syntax without querying device availability or allocating.
    torch.device(control_device)
    return {
        "mpc_backend_requested": mpc_backend,
        "mpc_backend_actual": "reference-cpu",
        "qp_backend_requested": qp_backend,
        "qp_backend_actual": "reference-cpu",
        "control_device_requested": control_device,
        "control_device_actual": "cpu",
    }
