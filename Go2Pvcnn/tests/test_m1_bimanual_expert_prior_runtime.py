from __future__ import annotations

import gc
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import weakref

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import LoadedStudent, save_student_artifact
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    FINGER_ORDER, LEFT_REFLECTION, MIXTURE_COMPONENTS, MIXTURE_OUTPUT_AXIS_ORDER,
    MODEL_INPUT_FIELD_ORDER, PHASE_ORDER, PRIOR_DT, PRIOR_HORIZON, StudentArtifactMetadata,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import FingertipMixtureNet
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
    FingertipPriorTarget, FrozenO6FingertipPrior, O6FingertipPriorInput, PriorRuntimeCfg,
)
import go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime as runtime_module


def _metadata() -> StudentArtifactMetadata:
    return StudentArtifactMetadata(
        format_version=1, input_dim=42, mixture_components=4, horizon=20, dt=PRIOR_DT,
        finger_order=FINGER_ORDER, phase_order=PHASE_ORDER,
        mirror_matrix=torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
        dataset_aggregate_sha256="a" * 64, teacher_ensemble_manifest_sha256="b" * 64,
        teacher_seed=1701, distillation_seed=42, code_commit="c" * 40, weight_sha256="0" * 64,
        hidden=(42,), input_field_order=MODEL_INPUT_FIELD_ORDER, output_axis_order=MIXTURE_OUTPUT_AXIS_ORDER,
    )


def _production_model() -> FingertipMixtureNet:
    model = FingertipMixtureNet(hidden=(42,))
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.input[0].weight.copy_(torch.eye(42))
        model.input[0].bias.fill_(1.0)
        model.head.bias[:MIXTURE_COMPONENTS] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Component zero's first future node exposes the first fifteen packed features.
        model.head.weight[4:19, :15].copy_(torch.eye(15))
        width = PRIOR_HORIZON * 5 * 3
        for component in range(1, MIXTURE_COMPONENTS):
            model.head.bias[4 + component * width:4 + (component + 1) * width].fill_(100.0)
    return model


@pytest.fixture
def production_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "approved-artifact"
    save_student_artifact(
        root, model=_production_model(), metadata=_metadata(),
        metrics={"student_nll": 1.0, "teacher_nll": 1.0, "nll_delta_per_dim": 0.0,
                 "first_step_velocity_rmse": 0.8, "first_step_zero_rmse": 1.0, "first_step_improvement": 0.2,
                 "endpoint_rmse": 0.8, "teacher_endpoint_rmse": 0.8, "endpoint_zero_rmse": 1.0,
                 "endpoint_improvement": 0.2, "production_approved": True, "deterministic_repeat_verified": True},
        latency={"warmups": 100, "measurements": 1000, "p99_ms": 1.0},
        provenance={"nonproduction_synthetic": False, "dataset_aggregate_sha256": "a" * 64,
                    "teacher_ensemble_manifest_sha256": "b" * 64},
    )
    return root


@pytest.fixture(autouse=True)
def _close_test_workers():
    runtimes: list[FrozenO6FingertipPrior] = []
    yield runtimes
    for runtime in runtimes: runtime.close()


def _runtime(path: Path, **cfg: object) -> FrozenO6FingertipPrior:
    return FrozenO6FingertipPrior.from_artifact(path, cfg=PriorRuntimeCfg(**cfg))


def _sample(**overrides: object) -> O6FingertipPriorInput:
    values: dict[str, object] = {
        "fingertip_positions_b": torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10.0,
        "contact_jacobian": torch.ones((15, 6), dtype=torch.float64),
        "qd": torch.full((6,), 0.25, dtype=torch.float64),
        "contact_mask": torch.zeros(5, dtype=torch.bool), "phase": BimanualPhase.GRASP,
    }
    values.update(overrides)
    return O6FingertipPriorInput(**values)  # type: ignore[arg-type]


class _EmptyResponses:
    def __init__(self) -> None: self._deadline = threading.Event()
    def get(self, *, timeout: float):
        assert not self._deadline.wait(timeout=timeout)
        raise queue.Empty


class _BaseExceptionResponses:
    def get(self, *, timeout: float): raise BaseException("worker transport failure")


class _StartFailureProcess:
    def start(self): raise RuntimeError("spawn failed")
    def is_alive(self): return False


class _StartFailureContext:
    def Queue(self, size): return queue.Queue(size)
    def Process(self, **kwargs): return _StartFailureProcess()


def test_runtime_packs_exact_float32_42_feature_order_through_artifact_worker(production_artifact, _close_test_workers):
    runtime = _runtime(production_artifact)
    _close_test_workers.append(runtime)
    sample = _sample(contact_mask=torch.tensor([False, True, False, True, False]))
    query = runtime.target(sample, baseline_qd=torch.zeros(6, dtype=torch.float64))
    assert query.target is not None and query.target.component == 0
    packed = torch.cat((sample.fingertip_positions_b.reshape(-1), sample.contact_jacobian @ sample.qd,
                        sample.contact_mask.to(torch.float64),
                        torch.nn.functional.one_hot(torch.tensor(2), num_classes=len(PHASE_ORDER)).to(torch.float64))).float()
    expected = torch.nn.functional.silu(torch.nn.functional.silu(packed[:15] + 1.0)).double()
    assert torch.allclose(query.target.mean_velocity, expected)
    assert torch.equal(query.target.precision[3:6], torch.zeros(3, dtype=torch.float64))


def test_safe_phase_and_invalid_input_disable_prior_without_target(production_artifact, _close_test_workers):
    runtime = _runtime(production_artifact)
    _close_test_workers.append(runtime)
    safe = runtime.target(_sample(phase=BimanualPhase.HOLD_SAFE), torch.zeros(6, dtype=torch.float64))
    invalid = runtime.target(_sample(qd=torch.zeros(6, dtype=torch.float32)), torch.zeros(6, dtype=torch.float64))
    assert safe.target is None and safe.diagnostics.reason == "safe_phase"
    assert invalid.target is None and invalid.diagnostics.reason == "invalid_input"


def test_timeout_returns_within_wall_clock_budget_bypasses_and_eventually_reaps(production_artifact, _close_test_workers):
    runtime = _runtime(production_artifact, inference_timeout_ms=20.0)
    _close_test_workers.append(runtime)
    worker, process = runtime._worker, runtime._worker.process
    worker.responses = _EmptyResponses()
    started = time.monotonic()
    timeout = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    wall_seconds = time.monotonic() - started
    assert timeout.target is None and timeout.diagnostics.reason == "timeout"
    assert timeout.diagnostics.inference_ms >= 20.0
    assert wall_seconds < 0.06
    bypass_started = time.monotonic()
    bypass = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert bypass.target is None and bypass.diagnostics.reason == "timeout"
    assert time.monotonic() - bypass_started < 0.01
    assert worker.reaped.wait(timeout=2.0)
    assert not process.is_alive() and process.exitcode is not None


def test_target_contains_cleanup_transport_base_exception(production_artifact, _close_test_workers):
    runtime = _runtime(production_artifact)
    _close_test_workers.append(runtime)
    worker, process = runtime._worker, runtime._worker.process
    runtime._worker.responses = _BaseExceptionResponses()
    result = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert result.target is None and result.diagnostics.reason == "prior_exception"
    assert result.diagnostics.inference_ms >= 0.0


def test_timeout_cleanup_error_cannot_escape_target(production_artifact, monkeypatch, _close_test_workers):
    runtime = _runtime(production_artifact, inference_timeout_ms=20.0)
    _close_test_workers.append(runtime)
    runtime._worker.responses = _EmptyResponses()
    def failed_enqueue(worker): raise RuntimeError("reaper unavailable")
    monkeypatch.setattr(runtime_module, "_enqueue_reap", failed_enqueue, raising=False)
    result = runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))
    assert result.target is None
    assert result.diagnostics.reason == "timeout_cleanup_error"
    assert result.diagnostics.inference_ms >= 20.0


def test_runtime_worker_uses_spawn_and_context_close_reaps_child(production_artifact, _close_test_workers):
    runtime = _runtime(production_artifact)
    _close_test_workers.append(runtime)
    process = runtime._worker.process
    assert process._start_method == "spawn"
    with runtime: assert process.is_alive()
    assert not process.is_alive() and process.exitcode is not None


def test_spawn_start_failure_is_bounded_and_normalized(production_artifact, monkeypatch):
    monkeypatch.setattr(runtime_module.mp, "get_context", lambda method: _StartFailureContext())
    with pytest.raises(RuntimeError, match="initialization failed"):
        FrozenO6FingertipPrior.from_artifact(production_artifact)


def test_finalizer_reaps_abandoned_worker(production_artifact):
    runtime = _runtime(production_artifact)
    process, reference = runtime._worker.process, weakref.ref(runtime)
    del runtime
    gc.collect()
    process.join(timeout=2.0)
    assert reference() is None
    assert not process.is_alive() and process.exitcode is not None


def test_close_waits_for_the_same_lifecycle_lock_as_query(production_artifact, monkeypatch, _close_test_workers):
    runtime = _runtime(production_artifact)
    _close_test_workers.append(runtime)
    worker, process = runtime._worker, runtime._worker.process
    entered, release, close_done = threading.Event(), threading.Event(), threading.Event()
    original_input = runtime._input
    def held_input(sample: O6FingertipPriorInput):
        entered.set(); assert release.wait(timeout=1.0); return original_input(sample)
    monkeypatch.setattr(runtime, "_input", held_input)
    query_thread = threading.Thread(target=lambda: runtime.target(_sample(), torch.zeros(6, dtype=torch.float64)))
    close_thread = threading.Thread(target=lambda: (runtime.close(), close_done.set()))
    query_thread.start(); assert entered.wait(timeout=1.0); close_thread.start()
    assert not close_done.wait(timeout=0.1)
    release.set(); query_thread.join(timeout=1.0); close_thread.join(timeout=1.0)
    assert not query_thread.is_alive() and not close_thread.is_alive()
    assert close_done.is_set() and worker.reaped.wait(timeout=2.0) and not process.is_alive()


def test_concurrent_query_honors_lock_deadline(production_artifact, monkeypatch, _close_test_workers):
    runtime = _runtime(production_artifact, inference_timeout_ms=20.0)
    _close_test_workers.append(runtime)
    entered, release = threading.Event(), threading.Event()
    original_input = runtime._input
    def held_input(sample: O6FingertipPriorInput):
        entered.set(); assert release.wait(timeout=1.0); return original_input(sample)
    monkeypatch.setattr(runtime, "_input", held_input)
    first = threading.Thread(target=lambda: runtime.target(_sample(), torch.zeros(6, dtype=torch.float64)))
    first.start(); assert entered.wait(timeout=1.0)
    results: list[object] = []
    second = threading.Thread(target=lambda: results.append(runtime.target(_sample(), torch.zeros(6, dtype=torch.float64))))
    second.start(); second.join(timeout=0.5); finished_before_release = not second.is_alive()
    release.set(); first.join(timeout=1.0); second.join(timeout=1.0)
    assert not first.is_alive() and not second.is_alive() and finished_before_release
    assert results[0].diagnostics.reason == "timeout"  # type: ignore[union-attr]


def test_target_defensively_clones_caller_owned_tensors():
    mean, precision = torch.ones(15, dtype=torch.float64), torch.ones(15, dtype=torch.float64)
    target = FingertipPriorTarget(mean, precision, 0, 1.0)
    mean.zero_(); precision.zero_()
    assert torch.equal(target.mean_velocity, torch.ones(15, dtype=torch.float64))
    assert torch.equal(target.precision, torch.ones(15, dtype=torch.float64))


def test_public_constructor_and_private_factory_forgery_reject(production_artifact):
    forged = LoadedStudent(model=_production_model(), metadata=_metadata(), metrics={"production_approved": True}, latency={})
    with pytest.raises(TypeError, match="from_artifact"):
        FrozenO6FingertipPrior(forged)  # type: ignore[call-arg]
    assert not hasattr(runtime_module, "_new_prior")
    runtime = FrozenO6FingertipPrior.from_artifact(production_artifact)
    runtime.close()


@pytest.mark.parametrize("weight", (0.0, -1.0, float("inf"), float("nan")))
def test_runtime_requires_strictly_positive_finite_logit_weight(weight):
    with pytest.raises(ValueError, match="logit_weight"):
        PriorRuntimeCfg(logit_weight=weight)


def test_runtime_import_graph_never_loads_offline_modules():
    project = Path(__file__).resolve().parents[1]
    command = (
        "import sys; import go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime; "
        "root='go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.'; "
        "blocked=(root+'download',root+'preprocess',root+'storage',root+'dexmanipnet',root+'urdf_fk','huggingface','h5py','isaacgym'); "
        "leaked=[name for name in sys.modules if any(part in name.lower() for part in blocked)]; assert not leaked, leaked"
    )
    environment = {**os.environ, "PYTHONPATH": str(project)}
    assert subprocess.run([sys.executable, "-c", command], env=environment, check=False).returncode == 0
