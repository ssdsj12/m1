"""Private fixed-address CUDA state mirror; CPU upload is an explicit boundary."""

import torch

from ..contracts import BimanualSnapshot, FullDynamicsState
from ..latent_contracts import pack_state_features


DYNAMICS_SHAPES = {
    "mass_matrix": (59, 59), "bias": (59,), "actuation_matrix": (59, 43),
    "wheel_contact_jacobian": (12, 59), "wheel_contact_bias": (12,),
}


def _workspace_device(batch, device):
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be a positive integer")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("workspace requires CUDA")
    # Resolve an unspecified CUDA index once, never in the steady-state path.
    return torch.device("cuda", torch.cuda.current_device() if target.index is None else target.index)


def _require(value, shape, device, dtype=torch.float32):
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"expected shape {shape}, dtype {dtype}, device {device}")
    if value.requires_grad:
        raise ValueError("private RTI workspaces do not accept autograd tensors")


class _FiniteRows:
    """Finite row reduction using only preallocated out= storage."""

    def __init__(self, batch, width, device):
        self.absolute = torch.empty((batch, width), dtype=torch.float32, device=device)
        self.elements = torch.empty((batch, width), dtype=torch.bool, device=device)
        self.rows = torch.empty(batch, dtype=torch.bool, device=device)

    def check(self, value):
        width = value.numel() // value.shape[0]
        # flatten() could copy noncontiguous sources. Callers pass contiguous staging.
        flat = value.view(value.shape[0], width)
        absolute = self.absolute[:, :width]
        elements = self.elements[:, :width]
        torch.abs(flat, out=absolute)
        torch.le(absolute, torch.finfo(torch.float32).max, out=elements)
        torch.all(elements, dim=1, out=self.rows)
        return self.rows


class GpuStateAdapter:
    """Latest accepted measurements plus a per-call validity mask.

    ``step`` accepts already resident float32 dynamics/features and exact int64
    nanosecond timestamps. Bad numeric or nonmonotonic rows preserve previous
    values but clear ``valid``. Consumers must gate on that mask. ``reset``
    clears row eligibility/timestamp without reallocating any storage.
    """

    def __init__(self, *, batch, device="cuda:0"):
        self.batch = batch
        self.device = _workspace_device(batch, device)
        shapes = {"features": (111,), **DYNAMICS_SHAPES}
        self._staging = {}
        for name, shape in shapes.items():
            setattr(self, name, torch.zeros((batch, *shape), dtype=torch.float32, device=self.device))
            self._staging[name] = torch.empty((batch, *shape), dtype=torch.float32, device=self.device)
        self.timestamps_ns = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self._timestamps = torch.empty_like(self.timestamps_ns)
        self.valid = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self._mask = torch.empty_like(self.valid)
        self.rejected_count = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self.reset_count = torch.zeros_like(self.rejected_count)
        self._finite = _FiniteRows(batch, 59 * 59, self.device)

    def dynamics_inputs(self):
        """Borrow published tensors; no CPU conversion or storage allocation."""
        return {name: getattr(self, name) for name in DYNAMICS_SHAPES}

    def step(self, features, timestamps_ns, **dynamics):
        if set(dynamics) != set(DYNAMICS_SHAPES):
            raise ValueError("exactly the five full dynamics fields are required")
        _require(features, (self.batch, 111), self.device)
        _require(timestamps_ns, (self.batch,), self.device, torch.int64)
        for name, shape in DYNAMICS_SHAPES.items():
            _require(dynamics[name], (self.batch, *shape), self.device)
        # Stage every borrowed source before publication, including aliases.
        self._staging["features"].copy_(features)
        for name in DYNAMICS_SHAPES:
            self._staging[name].copy_(dynamics[name])
        self._timestamps.copy_(timestamps_ns)
        torch.gt(self._timestamps, self.timestamps_ns, out=self.valid)
        torch.gt(self._timestamps, 0, out=self._mask)
        self.valid.logical_and_(self._mask)
        for value in self._staging.values():
            self.valid.logical_and_(self._finite.check(value))
        for name, value in self._staging.items():
            destination = getattr(self, name)
            mask = self.valid.view(self.batch, *((1,) * (value.ndim - 1)))
            torch.where(mask, value, destination, out=destination)
        torch.where(self.valid, self._timestamps, self.timestamps_ns, out=self.timestamps_ns)
        torch.logical_not(self.valid, out=self._mask)
        self.rejected_count.add_(self._mask)
        return self.valid

    def upload_cpu(self, snapshots, dynamics):
        """Legacy dataclass upload; allocations and host validation allowed here."""
        if len(snapshots) != self.batch or len(dynamics) != self.batch:
            raise ValueError("CPU upload must contain exactly batch rows")
        features, timestamps = [], []
        fields = {name: [] for name in DYNAMICS_SHAPES}
        for snapshot, state in zip(snapshots, dynamics):
            if not isinstance(snapshot, BimanualSnapshot) or not isinstance(state, FullDynamicsState):
                raise ValueError("CPU upload requires snapshot/full dynamics dataclasses")
            feature_fields = (snapshot.base_state, snapshot.m1_q, snapshot.m1_qd,
                snapshot.platform_q_qd, snapshot.left_arm.q, snapshot.left_arm.qd,
                snapshot.right_arm.q, snapshot.right_arm.qd, snapshot.left_hand.q,
                snapshot.left_hand.qd, snapshot.right_hand.q, snapshot.right_hand.qd,
                snapshot.box.pose_b, snapshot.box.twist_b)
            for value, width in zip(feature_fields, (13, 16, 16, 2, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6)):
                _require(value, (width,), torch.device("cpu"), torch.float64)
            if type(snapshot.timestamp_ns) is not int or not 0 < snapshot.timestamp_ns <= torch.iinfo(torch.int64).max:
                raise ValueError("timestamp_ns must be a positive int64")
            # Unlike the authoritative packer, this boundary stages invalid numeric
            # rows so GPU row masking can preserve valid neighbors atomically.
            if all(torch.isfinite(value).all().item() for value in feature_fields):
                features.append(pack_state_features(snapshot))
            else:
                features.append(torch.cat(feature_fields).float())
            timestamps.append(snapshot.timestamp_ns)
            for name, shape in DYNAMICS_SHAPES.items():
                value = getattr(state, name)
                _require(value, shape, torch.device("cpu"), torch.float64)
                fields[name].append(value)
        return self.step(torch.stack(features).to(self.device),
            torch.tensor(timestamps, dtype=torch.int64, device=self.device),
            **{name: torch.stack(values).to(self.device, torch.float32) for name, values in fields.items()})

    def reset(self, rows):
        _require(rows, (self.batch,), self.device, torch.bool)
        self._mask.copy_(rows)  # rows may alias valid
        self.valid.masked_fill_(self._mask, False)
        self.timestamps_ns.masked_fill_(self._mask, 0)
        self.reset_count.add_(self._mask)
