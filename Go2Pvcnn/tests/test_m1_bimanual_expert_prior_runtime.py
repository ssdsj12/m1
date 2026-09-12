from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import LoadedStudent
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MIXTURE_COMPONENTS,
    PRIOR_HORIZON,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FrozenO6FingertipPrior,
    O6FingertipPriorInput,
    PriorRuntimeCfg,
    _test_only_prior,
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


class _NeverReturnsModel(_FixedModel):
    def forward(self, value: torch.Tensor) -> _Distribution:
        time.sleep(1.0)
        return super().forward(value)


def _runtime(model: torch.nn.Module, **cfg: object) -> FrozenO6FingertipPrior:
    return _test_only_prior(model, cfg=PriorRuntimeCfg(**cfg))


@pytest.fixture(autouse=True)
def _close_test_workers():
    runtimes: list[FrozenO6FingertipPrior] = []
    yield runtimes
    for runtime in runtimes:
        runtime.close()


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


def test_runtime_selects_component_nearest_baseline_tip_velocity_and_packs_frozen_order(_close_test_workers):
    model = _FixedModel([0.0, 0.1, 0.8, -0.5], log_std=100.0)
    runtime = _runtime(model, precision_min=0.25, precision_max=4.0)
    _close_test_workers.append(runtime)
    sample = _sample()

    query = runtime.target(sample, baseline_qd=0.75 * torch.ones(6, dtype=torch.float64))

    assert query.target is not None
    assert query.target.component == 2
    assert torch.all(query.target.precision >= runtime.cfg.precision_min)
    assert torch.all(query.target.precision <= runtime.cfg.precision_max)
    assert query.target.mean_velocity.dtype == torch.float64
    assert query.target.precision.dtype == torch.float64


def test_safe_phase_and_nonfinite_output_disable_prior_without_target(_close_test_workers):
    safe_runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    invalid_runtime = _runtime(_NonfiniteModel([0.0, 0.1, 0.8, -0.5]))
    _close_test_workers.extend((safe_runtime, invalid_runtime))
    safe = safe_runtime.target(
        _sample(phase=BimanualPhase.HOLD_SAFE), torch.zeros(6, dtype=torch.float64)
    )
    invalid = invalid_runtime.target(
        _sample(), torch.zeros(6, dtype=torch.float64)
    )

    assert safe.target is None
    assert safe.diagnostics.reason == "safe_phase"
    assert invalid.target is None
    assert invalid.diagnostics.reason == "nonfinite_prior"


def test_contacted_tip_rows_are_unregularized_and_timeout_discards_target(_close_test_workers):
    contact_runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    timeout_runtime = _runtime(_NeverReturnsModel([0.0, 0.1, 0.8, -0.5]), inference_timeout_ms=20.0)
    _close_test_workers.extend((contact_runtime, timeout_runtime))
    contact = contact_runtime.target(
        _sample(contact_mask=torch.tensor([False, True, False, False, False])),
        torch.zeros(6, dtype=torch.float64),
    )
    started = time.monotonic()
    timeout = timeout_runtime.target(
        _sample(), torch.zeros(6, dtype=torch.float64)
    )
    elapsed = time.monotonic() - started

    assert contact.target is not None
    assert torch.equal(contact.target.precision[3:6], torch.zeros(3, dtype=torch.float64))
    assert timeout.target is None
    assert timeout.diagnostics.reason == "timeout"
    assert timeout.diagnostics.inference_ms >= 20.0
    assert elapsed < 0.25
    bypass_started = time.monotonic()
    bypass = timeout_runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert bypass.target is None
    assert bypass.diagnostics.reason == "timeout"
    assert time.monotonic() - bypass_started < 0.05


def test_input_contract_mismatch_is_a_disabled_diagnostic_not_an_exception(_close_test_workers):
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    _close_test_workers.append(runtime)
    result = runtime.target(
        _sample(qd=torch.zeros(6, dtype=torch.float32)), torch.zeros(6, dtype=torch.float64)
    )

    assert result.target is None
    assert result.diagnostics.reason == "invalid_input"


@pytest.mark.parametrize("error", (RuntimeError, IndexError, AssertionError))
def test_worker_converts_every_model_exception_to_disabled_diagnostics(error, _close_test_workers):
    class _Raises(torch.nn.Module):
        def forward(self, value: torch.Tensor):
            raise error("boom")

    runtime = _runtime(_Raises())
    _close_test_workers.append(runtime)
    result = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert result.target is None
    assert result.diagnostics.reason == "prior_exception"
    assert result.diagnostics.inference_ms >= 0.0


@pytest.mark.parametrize("weight", (0.0, -1.0, float("inf"), float("nan")))
def test_runtime_requires_strictly_positive_finite_logit_weight(weight):
    with pytest.raises(ValueError, match="logit_weight"):
        PriorRuntimeCfg(logit_weight=weight)


def test_public_constructor_rejects_a_forged_loaded_student():
    forged = LoadedStudent(
        model=_FixedModel([0.0, 0.1, 0.8, -0.5]),
        metadata=object(),  # type: ignore[arg-type]
        metrics={"production_approved": True},
        latency={},
    )
    with pytest.raises(TypeError, match="from_artifact"):
        FrozenO6FingertipPrior(_worker=forged, cfg=PriorRuntimeCfg())  # type: ignore[arg-type]


def test_runtime_import_graph_never_loads_offline_modules():
    project = Path(__file__).resolve().parents[1]
    command = (
        "import sys; "
        "import go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime; "
        "root='go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.'; "
        "blocked=(root+'download',root+'preprocess',root+'storage',root+'dexmanipnet',root+'urdf_fk','huggingface','h5py','isaacgym'); "
        "leaked=[name for name in sys.modules if any(part in name.lower() for part in blocked)]; "
        "assert not leaked, leaked"
    )
    environment = {**os.environ, "PYTHONPATH": str(project)}
    assert subprocess.run([sys.executable, "-c", command], env=environment, check=False).returncode == 0
