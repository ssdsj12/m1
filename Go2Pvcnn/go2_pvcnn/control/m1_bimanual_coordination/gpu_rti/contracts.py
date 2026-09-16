"""Private, unpadded, batch-first CUDA trajectory contract."""

from dataclasses import dataclass

import torch


@dataclass
class GpuRtiTrajectory:
    action: torch.Tensor  # [B, 25, 43], float32 CUDA; never tree padding
    accepted: torch.Tensor  # [B], bool on the same CUDA device
    merit: torch.Tensor  # [B], float32 on the same CUDA device

    def __post_init__(self) -> None:
        for name in ("action", "accepted", "merit"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                raise ValueError(f"{name} must be a CUDA tensor")
        if self.action.ndim != 3 or self.action.shape[0] < 1 or self.action.shape[1:] != (25, 43):
            raise ValueError("action must have shape [B, 25, 43] with positive batch")
        if self.action.dtype != torch.float32:
            raise ValueError("action must be float32")
        batch = self.action.shape[0]
        for name, dtype in (("accepted", torch.bool), ("merit", torch.float32)):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != dtype or value.device != self.action.device:
                raise ValueError(f"{name} must have shape [B], dtype {dtype}, and the action device")

    @classmethod
    def zeros(cls, *, batch: int, device: str = "cuda:0") -> "GpuRtiTrajectory":
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GPU RTI requires a CUDA device")
        return cls(
            action=torch.zeros((batch, 25, 43), dtype=torch.float32, device=target),
            accepted=torch.zeros(batch, dtype=torch.bool, device=target),
            merit=torch.zeros(batch, dtype=torch.float32, device=target),
        )
