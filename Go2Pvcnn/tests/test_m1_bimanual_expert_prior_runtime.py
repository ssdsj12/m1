from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import (
    LoadedStudent,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MIXTURE_COMPONENTS,
    PRIOR_HORIZON,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FrozenO6FingertipPrior,
    O6FingertipPriorInput,
    PriorRuntimeCfg,
)


@dataclass(frozen=True)
class _Distribution:
    logits: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor


class _FixedModel(torch.nn.Module):
    def __init__(self, means: list[float], *, log_std: float = 0.0) -> None:
        super().__init__()
        self.means = means
        self.log_std = log_std
        self.seen: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> _Distribution:
        self.seen = value.detach().clone()
        return _Distribution(
            logits=torch.tensor([[0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
            mean=torch.stack(
                [torch.full((PRIOR_HORIZON, 5, 3), mean, dtype=torch.float32) for mean in self.means]
            ).unsqueeze(0),
            log_std=torch.full((1, MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3), self.log_std, dtype=torch.float32),
        )


class _NonfiniteModel(_FixedModel):
    def forward(self, value: torch.Tensor) -> _Distribution:
        distribution = super().forward(value)
        return _Distribution(distribution.logits, distribution.mean * float("nan"), distribution.log_std)


class _SlowModel(_FixedModel):
    def forward(self, value: torch.Tensor) -> _Distribution:
        import time

        time.sleep(0.01)
        return super().forward(value)


def _runtime(model: torch.nn.Module, **cfg: object) -> FrozenO6FingertipPrior:
    loaded = LoadedStudent(
        model=model,  # type: ignore[arg-type]
        metadata=object(),  # type: ignore[arg-type]
        metrics={"production_approved": True},
        latency={},
    )
    return FrozenO6FingertipPrior(loaded, cfg=PriorRuntimeCfg(**cfg))


def _sample(**overrides: object) -> O6FingertipPriorInput:
    values: dict[str, object] = {
        "fingertip_positions_b": torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10.0,
        "contact_jacobian": torch.ones((15, 6), dtype=torch.float64),
        "qd": torch.full((6,), 0.25, dtype=torch.float64),
        "contact_mask": torch.zeros(5, dtype=torch.bool),
        "phase": BimanualPhase.GRASP,
    }
    values.update(overrides)
    return O6FingertipPriorInput(**values)  # type: ignore[arg-type]


def test_runtime_selects_component_nearest_baseline_tip_velocity_and_packs_frozen_order():
    model = _FixedModel([0.0, 0.1, 0.8, -0.5], log_std=100.0)
    runtime = _runtime(model, precision_min=0.25, precision_max=4.0)
    sample = _sample()

    query = runtime.target(sample, baseline_qd=0.75 * torch.ones(6, dtype=torch.float64))

    assert query.target is not None
    assert query.target.component == 2
    assert torch.all(query.target.precision >= runtime.cfg.precision_min)
    assert torch.all(query.target.precision <= runtime.cfg.precision_max)
    assert query.target.mean_velocity.dtype == torch.float64
    assert query.target.precision.dtype == torch.float64
    assert model.seen is not None
    assert model.seen.dtype == torch.float32
    assert model.seen.shape == (1, 42)
    assert torch.equal(model.seen[0, :15], sample.fingertip_positions_b.reshape(-1).float())
    assert torch.equal(model.seen[0, 15:30], (sample.contact_jacobian @ sample.qd).float())
    assert torch.equal(model.seen[0, 30:35], sample.contact_mask.float())
    assert torch.equal(model.seen[0, 35:], torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32))


def test_safe_phase_and_nonfinite_output_disable_prior_without_target():
    safe = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5])).target(
        _sample(phase=BimanualPhase.HOLD_SAFE), torch.zeros(6, dtype=torch.float64)
    )
    invalid = _runtime(_NonfiniteModel([0.0, 0.1, 0.8, -0.5])).target(
        _sample(), torch.zeros(6, dtype=torch.float64)
    )

    assert safe.target is None
    assert safe.diagnostics.reason == "safe_phase"
    assert invalid.target is None
    assert invalid.diagnostics.reason == "nonfinite_prior"


def test_contacted_tip_rows_are_unregularized_and_timeout_discards_target():
    contact = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5])).target(
        _sample(contact_mask=torch.tensor([False, True, False, False, False])),
        torch.zeros(6, dtype=torch.float64),
    )
    timeout = _runtime(_SlowModel([0.0, 0.1, 0.8, -0.5]), inference_timeout_ms=0.01).target(
        _sample(), torch.zeros(6, dtype=torch.float64)
    )

    assert contact.target is not None
    assert torch.equal(contact.target.precision[3:6], torch.zeros(3, dtype=torch.float64))
    assert timeout.target is None
    assert timeout.diagnostics.reason == "timeout"
    assert timeout.diagnostics.inference_ms > 0.01


def test_input_contract_mismatch_is_a_disabled_diagnostic_not_an_exception():
    result = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5])).target(
        _sample(qd=torch.zeros(6, dtype=torch.float32)), torch.zeros(6, dtype=torch.float64)
    )

    assert result.target is None
    assert result.diagnostics.reason == "invalid_input"


def test_runtime_rejects_nonproduction_loaded_student_atomically():
    model = _FixedModel([0.0, 0.1, 0.8, -0.5])
    loaded = LoadedStudent(
        model=model,  # type: ignore[arg-type]
        metadata=object(),  # type: ignore[arg-type]
        metrics={"production_approved": False},
        latency={},
    )

    with pytest.raises(ValueError, match="production_approved"):
        FrozenO6FingertipPrior(loaded)
