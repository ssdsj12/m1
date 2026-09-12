from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import subprocess
import sys
import threading
import weakref

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import LoadedStudent
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MIXTURE_COMPONENTS,
    PRIOR_HORIZON,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FingertipPriorTarget,
    FrozenO6FingertipPrior,
    O6FingertipPriorInput,
    PriorRuntimeCfg,
)
import go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime as runtime_module


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
        while True:
            pass


class _UnpicklableModel(_FixedModel):
    def __init__(self, means: list[float]) -> None:
        super().__init__(means)
        self.callback = lambda: None


class _RaisesModel(torch.nn.Module):
    def __init__(self, error: type[BaseException]) -> None:
        super().__init__()
        self.error = error

    def forward(self, value: torch.Tensor):
        raise self.error("boom")


_TEST_MODEL: torch.nn.Module | None = None


def _runtime(model: torch.nn.Module, **cfg: object) -> FrozenO6FingertipPrior:
    global _TEST_MODEL
    _TEST_MODEL = model
    return FrozenO6FingertipPrior.from_artifact("test-only-artifact", cfg=PriorRuntimeCfg(**cfg))


@pytest.fixture(autouse=True)
def _close_test_workers(monkeypatch):
    runtimes: list[FrozenO6FingertipPrior] = []
    def _loaded(_: object) -> LoadedStudent:
        assert _TEST_MODEL is not None
        return LoadedStudent(model=_TEST_MODEL, metadata=object(), metrics={"production_approved": True}, latency={})  # type: ignore[arg-type]
    monkeypatch.setattr(runtime_module, "load_student_artifact", _loaded)
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


def test_contacted_tip_rows_are_unregularized_and_timeout_reaps_then_bypasses(_close_test_workers):
    contact_runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    timeout_runtime = _runtime(_NeverReturnsModel([0.0, 0.1, 0.8, -0.5]), inference_timeout_ms=20.0)
    _close_test_workers.extend((contact_runtime, timeout_runtime))
    contact = contact_runtime.target(
        _sample(contact_mask=torch.tensor([False, True, False, False, False])),
        torch.zeros(6, dtype=torch.float64),
    )
    process = timeout_runtime._worker.process
    timeout = timeout_runtime.target(
        _sample(), torch.zeros(6, dtype=torch.float64)
    )

    assert contact.target is not None
    assert torch.equal(contact.target.precision[3:6], torch.zeros(3, dtype=torch.float64))
    assert timeout.target is None
    assert timeout.diagnostics.reason == "timeout"
    assert timeout.diagnostics.inference_ms >= 20.0
    assert timeout_runtime._worker.process is process
    assert not process.is_alive()
    assert process.exitcode is not None
    bypass = timeout_runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert bypass.target is None
    assert bypass.diagnostics.reason == "timeout"


def test_input_contract_mismatch_is_a_disabled_diagnostic_not_an_exception(_close_test_workers):
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    _close_test_workers.append(runtime)
    result = runtime.target(
        _sample(qd=torch.zeros(6, dtype=torch.float32)), torch.zeros(6, dtype=torch.float64)
    )

    assert result.target is None
    assert result.diagnostics.reason == "invalid_input"


@pytest.mark.parametrize("error", (RuntimeError, IndexError, AssertionError, BaseException))
def test_worker_converts_every_model_exception_to_disabled_diagnostics(error, _close_test_workers):
    runtime = _runtime(_RaisesModel(error))
    _close_test_workers.append(runtime)
    result = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert result.target is None
    assert result.diagnostics.reason == "prior_exception"
    assert result.diagnostics.inference_ms >= 0.0


@pytest.mark.parametrize("weight", (0.0, -1.0, float("inf"), float("nan")))
def test_runtime_requires_strictly_positive_finite_logit_weight(weight):
    with pytest.raises(ValueError, match="logit_weight"):
        PriorRuntimeCfg(logit_weight=weight)


def test_runtime_worker_uses_spawn_and_context_close_reaps_child(_close_test_workers):
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    process = runtime._worker.process  # Private lifecycle evidence, not an injection seam.
    assert process._start_method == "spawn"
    with runtime:
        assert process.is_alive()
    assert not process.is_alive()
    assert process.exitcode is not None


def test_worker_start_failure_is_bounded_and_normalized_to_runtime_error():
    with pytest.raises(RuntimeError, match="initialization failed"):
        _runtime(_UnpicklableModel([0.0, 0.1, 0.8, -0.5]))


def test_finalizer_reaps_abandoned_worker():
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    process = runtime._worker.process
    reference = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert reference() is None
    assert not process.is_alive()
    assert process.exitcode is not None


def test_close_waits_for_the_same_lifecycle_lock_as_query(monkeypatch, _close_test_workers):
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]))
    _close_test_workers.append(runtime)
    entered = threading.Event()
    release = threading.Event()
    close_done = threading.Event()
    original_input = runtime._input

    def held_input(sample: O6FingertipPriorInput):
        entered.set()
        assert release.wait(timeout=1.0)
        return original_input(sample)

    monkeypatch.setattr(runtime, "_input", held_input)
    query_thread = threading.Thread(target=lambda: runtime.target(_sample(), torch.zeros(6, dtype=torch.float64)))
    close_thread = threading.Thread(target=lambda: (runtime.close(), close_done.set()))
    query_thread.start()
    assert entered.wait(timeout=1.0)
    close_thread.start()
    assert not close_done.wait(timeout=0.1)
    release.set()
    query_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)
    assert not query_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_done.is_set()
    assert not runtime._worker.process.is_alive()


def test_concurrent_query_honors_the_hard_deadline_while_lifecycle_lock_is_held(monkeypatch, _close_test_workers):
    runtime = _runtime(_FixedModel([0.0, 0.1, 0.8, -0.5]), inference_timeout_ms=20.0)
    _close_test_workers.append(runtime)
    entered = threading.Event()
    release = threading.Event()
    first = threading.Thread(target=lambda: runtime.target(_sample(), torch.zeros(6, dtype=torch.float64)))
    original_input = runtime._input

    def held_input(sample: O6FingertipPriorInput):
        entered.set()
        assert release.wait(timeout=1.0)
        return original_input(sample)

    monkeypatch.setattr(runtime, "_input", held_input)
    first.start()
    assert entered.wait(timeout=1.0)
    results: list[object] = []
    second = threading.Thread(target=lambda: results.append(runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))))
    second.start()
    second.join(timeout=0.5)
    second_finished_before_release = not second.is_alive()
    release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_finished_before_release
    assert results[0].diagnostics.reason == "timeout"  # type: ignore[union-attr]


def test_target_defensively_clones_caller_owned_tensors():
    mean = torch.ones(15, dtype=torch.float64)
    precision = torch.ones(15, dtype=torch.float64)
    target = FingertipPriorTarget(mean, precision, 0, 1.0)
    mean.zero_()
    precision.zero_()
    assert torch.equal(target.mean_velocity, torch.ones(15, dtype=torch.float64))
    assert torch.equal(target.precision, torch.ones(15, dtype=torch.float64))


def test_public_constructor_rejects_a_forged_loaded_student():
    forged = LoadedStudent(
        model=_FixedModel([0.0, 0.1, 0.8, -0.5]),
        metadata=object(),  # type: ignore[arg-type]
        metrics={"production_approved": True},
        latency={},
    )
    with pytest.raises(TypeError, match="from_artifact"):
        FrozenO6FingertipPrior(forged)  # type: ignore[call-arg]


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
