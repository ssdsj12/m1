from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    MixtureDistribution,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import FingertipMixtureNet


SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_distill_fingertip_prior.py"
EVAL_SCRIPT = Path(__file__).parents[1] / "scripts" / "m1_dual_panda_o6_eval_fingertip_prior.py"


def _module():
    spec = importlib.util.spec_from_file_location("task7_distill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _eval_module():
    spec = importlib.util.spec_from_file_location("task7_eval", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _teacher() -> MixtureDistribution:
    logits = torch.tensor([[0.4, -0.7, 1.2, -0.2]], dtype=torch.float32)
    offsets = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    return MixtureDistribution(
        logits=logits,
        mean=offsets.expand(1, 4, 20, 5, 3).clone(),
        log_std=torch.full((1, 4, 20, 5, 3), -1.5, dtype=torch.float32),
    )


def _permuted(distribution: MixtureDistribution, permutation: torch.Tensor) -> MixtureDistribution:
    return MixtureDistribution(
        logits=distribution.logits[:, permutation],
        mean=distribution.mean[:, permutation],
        log_std=distribution.log_std[:, permutation],
    )


def test_distillation_scores_fixed_teacher_samples_without_component_alignment():
    module = _module()
    teacher = _teacher()
    permuted = _permuted(teacher, torch.tensor([2, 0, 3, 1]))
    samples = module.sample_mixture(teacher, samples_per_state=8, seed=42)

    assert samples.shape == (1, 8, 20, 5, 3)
    assert not torch.equal(teacher.logits, permuted.logits)
    torch.testing.assert_close(
        module.distribution_distillation_loss(teacher, samples),
        module.distribution_distillation_loss(permuted, samples),
    )


def test_distillation_loss_rejects_samples_with_wrong_frozen_shape():
    module = _module()
    with pytest.raises(ValueError, match="teacher_samples"):
        module.distribution_distillation_loss(_teacher(), torch.zeros(1, 8, 20, 3, 5))


def test_ensemble_teacher_aggregation_keeps_each_member_four_component_contract():
    module = _module()
    torch.manual_seed(3)
    members = (FingertipMixtureNet(hidden=(16, 16)), FingertipMixtureNet(hidden=(16, 16)))
    inputs = torch.zeros(2, 42, dtype=torch.float32)
    target = torch.zeros(2, 20, 5, 3, dtype=torch.float32)

    log_prob = module._ensemble_log_prob(members, inputs, target)
    samples = module._sample_ensemble(members, inputs, samples_per_state=3, seed=7)

    assert log_prob.shape == (2,)
    assert samples.shape == (2, 3, 20, 5, 3)
    assert torch.isfinite(log_prob).all() and torch.isfinite(samples).all()


class _FixedMixture(nn.Module):
    def __init__(self, logits: list[float], means: list[float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.register_buffer("fixed_means", torch.tensor(means, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> MixtureDistribution:
        batch = inputs.shape[0]
        logits = self.fixed_logits.expand(batch, -1)
        means = self.fixed_means.reshape(1, 4, 1, 1, 1).expand(batch, 4, 20, 5, 3)
        return MixtureDistribution(logits, means, torch.full_like(means, -7.0))


def test_ensemble_sampling_is_invariant_to_each_members_uniform_logit_shift():
    module = _module()
    inputs = torch.zeros(1, 42, dtype=torch.float32)
    original = (
        _FixedMixture([0.0, 1.0, -1.0, -2.0], [0.0, 1.0, 2.0, 3.0]),
        _FixedMixture([2.0, -2.0, 0.5, 0.0], [10.0, 11.0, 12.0, 13.0]),
    )
    shifted = (
        _FixedMixture([17.0, 18.0, 16.0, 15.0], [0.0, 1.0, 2.0, 3.0]),
        _FixedMixture([-9.0, -13.0, -10.5, -11.0], [10.0, 11.0, 12.0, 13.0]),
    )

    first = module._sample_ensemble(original, inputs, samples_per_state=64, seed=91)
    second = module._sample_ensemble(shifted, inputs, samples_per_state=64, seed=91)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    expected = torch.cat(
        [member(inputs).logits.softmax(-1) / len(original) for member in original], dim=-1
    )
    torch.testing.assert_close(module._ensemble_component_probabilities(original, inputs), expected)

    target = torch.zeros(1, 20, 5, 3, dtype=torch.float32)
    torch.testing.assert_close(
        module._ensemble_log_prob(original, inputs, target),
        module._ensemble_log_prob(shifted, inputs, target),
    )


def test_teacher_sample_store_is_atomic_sha_pinned_and_batch_bounded(tmp_path: Path, monkeypatch):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(9, 42, dtype=torch.float32),
        torch.zeros(9, 20, 5, 3, dtype=torch.float32),
        tuple(f"g{index}" for index in range(9)),
    )
    members = (_FixedMixture([0.0] * 4, [0.0] * 4), _FixedMixture([0.0] * 4, [1.0] * 4))
    seen: list[int] = []
    original = module._sample_ensemble

    def bounded(models, inputs, **kwargs):
        seen.append(inputs.shape[0])
        return original(models, inputs, **kwargs)

    monkeypatch.setattr(module, "_sample_ensemble", bounded)
    store = module._prepare_teacher_sample_store(
        tmp_path / "samples", members, dataset, samples_per_state=8, seed=42,
        dataset_sha256="a" * 64, ensemble_sha256="b" * 64, chunk_size=2,
    )

    assert max(seen) <= 2
    assert isinstance(store.samples, np.memmap)
    manifest = json.loads((store.root / "manifest.json").read_text())
    assert manifest["sample_count"] == 9 and manifest["samples_per_state"] == 8
    assert manifest["dataset_aggregate_sha256"] == "a" * 64
    assert manifest["teacher_ensemble_manifest_sha256"] == "b" * 64
    assert module.sha256_file(store.root / "samples.npy") == manifest["samples_sha256"]

    with pytest.raises(ValueError, match="identity"):
        module._prepare_teacher_sample_store(
            store.root, members, dataset, samples_per_state=8, seed=42,
            dataset_sha256="a" * 64, ensemble_sha256="b" * 64, chunk_size=3,
        )


def test_teacher_sample_store_failure_never_publishes_or_pollutes(tmp_path: Path, monkeypatch):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(3, 42, dtype=torch.float32),
        torch.zeros(3, 20, 5, 3, dtype=torch.float32),
        ("g0", "g1", "g2"),
    )
    calls = 0

    def explode(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_sample_ensemble", explode)
    with pytest.raises(RuntimeError, match="interrupted"):
        module._prepare_teacher_sample_store(
            tmp_path / "samples", (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
            samples_per_state=8, seed=42, dataset_sha256="a" * 64,
            ensemble_sha256="b" * 64, chunk_size=2,
        )
    assert calls == 1
    assert not (tmp_path / "samples").exists()
    assert not list(tmp_path.glob(".samples.stage-*"))


def test_teacher_sample_store_rejects_symlink_and_coordinated_nonfinite_tampering(tmp_path: Path):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(2, 42), torch.zeros(2, 20, 5, 3), ("g0", "g1")
    )
    store = module._prepare_teacher_sample_store(
        tmp_path / "samples", (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
        samples_per_state=2, seed=42, dataset_sha256="a" * 64,
        ensemble_sha256="b" * 64,
    )
    link = tmp_path / "linked"
    link.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._prepare_teacher_sample_store(
            link, (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
            samples_per_state=2, seed=42, dataset_sha256="a" * 64,
            ensemble_sha256="b" * 64,
        )

    values = np.load(store.root / "samples.npy", mmap_mode="r+")
    values[0, 0, 0, 0, 0] = np.nan
    values.flush()
    del values
    manifest_path = store.root / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["samples_sha256"] = module.sha256_file(store.root / "samples.npy")
    body = dict(document)
    body.pop("manifest_sha256")
    document["manifest_sha256"] = module.sha256(module._canonical_json(body)).hexdigest()
    manifest_path.write_bytes(module._canonical_json(document))
    with pytest.raises(ValueError, match="finite"):
        module._load_teacher_sample_store(store.root, {
            key: document[key] for key in (
                "format_version", "sampling_algorithm", "sample_count", "samples_per_state",
                "seed", "dataset_aggregate_sha256", "teacher_ensemble_manifest_sha256",
                "chunk_size", "shape", "dtype",
            )
        })


def test_teacher_sample_store_rejects_coordinated_legacy_manifest_fields(tmp_path: Path):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(2, 42), torch.zeros(2, 20, 5, 3), ("g0", "g1")
    )
    store = module._prepare_teacher_sample_store(
        tmp_path / "samples", (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
        samples_per_state=2, seed=42, dataset_sha256="a" * 64,
        ensemble_sha256="b" * 64,
    )
    path = store.root / "manifest.json"
    document = json.loads(path.read_text())
    document["legacy_sampler"] = "unverified"
    body = dict(document)
    body.pop("manifest_sha256")
    document["manifest_sha256"] = module.sha256(module._canonical_json(body)).hexdigest()
    path.write_bytes(module._canonical_json(document))

    with pytest.raises(ValueError, match="schema"):
        module._prepare_teacher_sample_store(
            store.root, (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
            samples_per_state=2, seed=42, dataset_sha256="a" * 64,
            ensemble_sha256="b" * 64,
        )


def test_student_resume_matches_uninterrupted_epoch_boundary_training(tmp_path: Path):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(6, 42, dtype=torch.float32),
        torch.linspace(-0.1, 0.1, 6 * 20 * 5 * 3, dtype=torch.float32).reshape(6, 20, 5, 3),
        tuple(f"g{index}" for index in range(6)),
    )
    members = (_FixedMixture([0.0] * 4, [0.0, 0.1, -0.1, 0.2]),)
    store = module._prepare_teacher_sample_store(
        tmp_path / "teacher", members, dataset, samples_per_state=2, seed=42,
        dataset_sha256="a" * 64, ensemble_sha256="b" * 64, chunk_size=2,
    )
    identity = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=2,
        batch_size=3, learning_rate=1e-3, samples_per_state=2, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    resumed_workspace = module._StudentResumeWorkspace(tmp_path / ".student-a.resume-v1", identity)
    with pytest.raises(RuntimeError, match="test interruption"):
        module._train_student(
            dataset, store, hidden=(8,), epochs=2, batch_size=3, learning_rate=1e-3,
            seed=42, device=torch.device("cpu"), workspace=resumed_workspace,
            interrupt_after_epoch=1,
        )
    resumed = module._train_student(
        dataset, store, hidden=(8,), epochs=2, batch_size=3, learning_rate=1e-3,
        seed=42, device=torch.device("cpu"), workspace=resumed_workspace,
    )

    clean_identity = dict(identity)
    clean_workspace = module._StudentResumeWorkspace(tmp_path / ".student-b.resume-v1", clean_identity)
    clean = module._train_student(
        dataset, store, hidden=(8,), epochs=2, batch_size=3, learning_rate=1e-3,
        seed=42, device=torch.device("cpu"), workspace=clean_workspace,
    )
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, clean.state_dict()[key], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU0 is unavailable")
def test_student_cuda0_real_shape_resumes_after_epoch_interruption(tmp_path: Path):
    # CUDA cannot be de-initialized, so isolate this integration check from CPU preflight tests.
    code = r'''
import importlib.util
from pathlib import Path
import sys
import torch

script, root = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("cuda_distill_check", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._configure_cuda_determinism("cuda:0")
device = torch.device("cuda:0")
dataset = module.GroupShardDataset(
    torch.zeros(4, 42), torch.zeros(4, 20, 5, 3), tuple(f"g{i}" for i in range(4))
)
member = module.FingertipMixtureNet(hidden=(8,))
store = module._prepare_teacher_sample_store(
    root / "teacher", (member,), dataset, samples_per_state=2, seed=42,
    dataset_sha256="a" * 64, ensemble_sha256="b" * 64, chunk_size=2,
)
identity = module._distillation_identity(
    aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=2,
    batch_size=2, learning_rate=1e-3, samples_per_state=2, seed=42,
    device=device, synthetic_smoke=False,
)
assert identity["cublas_workspace_config"] == ":4096:8"
assert identity["device"]["type"] == "cuda"
assert set(identity["device"]) == {"type", "uuid", "name", "compute_capability"}
workspace = module._StudentResumeWorkspace(root / ".cuda.resume-v1", identity)
try:
    module._train_student(
        dataset, store, hidden=(8,), epochs=2, batch_size=2, learning_rate=1e-3,
        seed=42, device=device, workspace=workspace, interrupt_after_epoch=1,
    )
except RuntimeError as error:
    assert "test interruption" in str(error)
else:
    raise AssertionError("expected epoch interruption")
model = module._train_student(
    dataset, store, hidden=(8,), epochs=2, batch_size=2, learning_rate=1e-3,
    seed=42, device=device, workspace=workspace,
)
distribution = model(torch.zeros(1, 42, device=device))
assert distribution.mean.shape == (1, 4, 20, 5, 3)
assert distribution.mean.device == device and torch.isfinite(distribution.mean).all()
'''
    completed = __import__("subprocess").run(
        [sys.executable, "-c", code, str(SCRIPT), str(tmp_path)],
        check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    assert completed.returncode == 0, completed.stderr


def test_student_resume_rejects_changed_semantic_identity_and_symlink(tmp_path: Path):
    module = _module()
    identity = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=2,
        batch_size=3, learning_rate=1e-3, samples_per_state=8, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    path = tmp_path / ".student.resume-v1"
    module._StudentResumeWorkspace(path, identity)
    changed = dict(identity)
    changed["samples_per_state"] = 7
    with pytest.raises(ValueError, match="identity"):
        module._StudentResumeWorkspace(path, changed)

    linked = tmp_path / ".linked.resume-v1"
    linked.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._StudentResumeWorkspace(linked, identity)

    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.symlink_to(tmp_path / "real-parent", target_is_directory=True)
    (tmp_path / "real-parent").mkdir()
    with pytest.raises(ValueError, match="symlink"):
        module._StudentResumeWorkspace(unsafe_parent / ".student.resume-v1", identity)


def test_student_resume_rejects_a_symlinked_checkpoint_directory(tmp_path: Path):
    module = _module()
    identity = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=2,
        batch_size=3, learning_rate=1e-3, samples_per_state=8, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    path = tmp_path / ".student.resume-v1"
    module._StudentResumeWorkspace(path, identity)
    (path / "checkpoints").rmdir()
    outside = tmp_path / "outside-checkpoints"
    outside.mkdir()
    (path / "checkpoints").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module._StudentResumeWorkspace(path, identity)


def test_student_resume_initialization_is_atomic_on_progress_write_failure(tmp_path: Path, monkeypatch):
    module = _module()
    identity = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=2,
        batch_size=3, learning_rate=1e-3, samples_per_state=8, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    path = tmp_path / ".student.resume-v1"

    def fail(*args, **kwargs):
        raise OSError("simulated durable write failure")

    monkeypatch.setattr(module, "_atomic_bytes", fail)
    with pytest.raises(OSError, match="simulated"):
        module._StudentResumeWorkspace(path, identity)
    assert not path.exists()
    assert not list(tmp_path.glob("..student.resume-v1.stage-*"))


def test_student_resume_rejects_coordinated_legacy_progress_and_checkpoint_fields(tmp_path: Path):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(3, 42), torch.zeros(3, 20, 5, 3), ("g0", "g1", "g2")
    )
    store = module._prepare_teacher_sample_store(
        tmp_path / "teacher", (_FixedMixture([0.0] * 4, [0.0] * 4),), dataset,
        samples_per_state=2, seed=42, dataset_sha256="a" * 64,
        ensemble_sha256="b" * 64,
    )
    identity = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=1,
        batch_size=3, learning_rate=1e-3, samples_per_state=2, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    progress_path = tmp_path / ".student.resume-v1"
    workspace = module._StudentResumeWorkspace(progress_path, identity)
    module._train_student(
        dataset, store, hidden=(8,), epochs=1, batch_size=3, learning_rate=1e-3,
        seed=42, device=torch.device("cpu"), workspace=workspace,
    )

    progress_file = progress_path / "progress.json"
    progress = json.loads(progress_file.read_text())
    progress["legacy_epoch"] = 1
    body = dict(progress)
    body.pop("progress_sha256")
    progress["progress_sha256"] = module.sha256(module._canonical_json(body)).hexdigest()
    progress_file.write_bytes(module._canonical_json(progress))
    with pytest.raises(ValueError, match="schema"):
        module._StudentResumeWorkspace(progress_path, identity)

    del progress["legacy_epoch"]
    checkpoint_path = progress_path / progress["checkpoint"]["path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["legacy_rng_state"] = torch.zeros(1)
    torch.save(checkpoint, checkpoint_path)
    progress["checkpoint"]["sha256"] = module.sha256_file(checkpoint_path)
    body = dict(progress)
    body.pop("progress_sha256")
    progress["progress_sha256"] = module.sha256(module._canonical_json(body)).hexdigest()
    progress_file.write_bytes(module._canonical_json(progress))
    workspace = module._StudentResumeWorkspace(progress_path, identity)
    with pytest.raises(ValueError, match="state"):
        workspace.load(hidden=(8,))


def test_distillation_module_does_not_import_the_expert_cli_as_a_library():
    sys.modules.pop("m1_dual_panda_o6_train_fingertip_expert", None)
    _module()
    assert "m1_dual_panda_o6_train_fingertip_expert" not in sys.modules


def test_distillation_parser_and_identity_bind_cuda_and_all_training_semantics(monkeypatch):
    module = _module()
    defaults = module.build_parser().parse_args(["--output-dir", "x"])
    assert defaults.device == module.DISTILLATION_TRAINING_DEFAULTS["device"]
    assert defaults.epochs == module.DISTILLATION_TRAINING_DEFAULTS["epochs"]
    assert defaults.batch_size == module.DISTILLATION_TRAINING_DEFAULTS["batch_size"]
    assert defaults.learning_rate == module.DISTILLATION_TRAINING_DEFAULTS["learning_rate"]
    assert defaults.samples_per_state == module.DISTILLATION_TRAINING_DEFAULTS["samples_per_state"]
    first = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=200,
        batch_size=128, learning_rate=1e-3, samples_per_state=8, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    assert first["samples_per_state"] == 8
    assert first["training_semantics"]["source_sha256"].keys() == {
        "trainer", "model", "contracts", "artifact"
    }
    changed = module._distillation_identity(
        aggregate_sha="a" * 64, ensemble_sha="b" * 64, hidden=(8,), epochs=200,
        batch_size=128, learning_rate=1e-3, samples_per_state=7, seed=42,
        device=torch.device("cpu"), synthetic_smoke=False,
    )
    assert module._identity_sha256(first) != module._identity_sha256(changed)


def test_distillation_cuda_preflight_rejects_before_creating_output(tmp_path: Path):
    output = tmp_path / "must-not-exist"
    completed = __import__("subprocess").run(
        [
            sys.executable, str(SCRIPT), "--synthetic-smoke", "--epochs", "1",
            "--device", "cuda:0", "--output-dir", str(output),
        ],
        check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1]), "CUBLAS_WORKSPACE_CONFIG": "invalid"},
    )

    assert completed.returncode != 0
    assert "CUBLAS_WORKSPACE_CONFIG" in completed.stderr
    assert not output.exists()
    assert not output.with_name(f".{output.name}.resume-v1").exists()


def test_real_gate_failure_returns_nonzero_without_final_artifact_and_hashes_diagnostic(
    tmp_path: Path, monkeypatch,
):
    module = _module()
    dataset = module.GroupShardDataset(
        torch.zeros(2, 42), torch.ones(2, 20, 5, 3), ("g0", "g1")
    )
    aggregate = "a" * 64
    monkeypatch.setattr(module, "_resolve_manifest", lambda value: tmp_path / "aggregate_manifest.json")
    monkeypatch.setattr(module, "verify_aggregate_manifest", lambda root: {"aggregate_sha256": aggregate})
    monkeypatch.setattr(module, "_load_group_split", lambda path, split: (dataset, {"aggregate_sha256": aggregate}))
    ensemble = module._Ensemble(
        (_FixedMixture([0.0] * 4, [0.0] * 4),), "b" * 64, aggregate, (8,), 42, False
    )
    monkeypatch.setattr(module, "_load_ensemble", lambda *args, **kwargs: ensemble)
    monkeypatch.setattr(module, "_prepare_teacher_sample_store", lambda *args, **kwargs: object())

    def deterministic_model(*args, **kwargs):
        torch.manual_seed(4)
        return FingertipMixtureNet(hidden=(8,))

    monkeypatch.setattr(module, "_train_student", deterministic_model)
    monkeypatch.setattr(module, "_metrics", lambda *args, **kwargs: {
        "student_nll": 400.0, "teacher_nll": 300.0, "nll_delta_per_dim": 1 / 3,
        "first_step_velocity_rmse": 1.0, "first_step_zero_rmse": 1.0,
        "first_step_improvement": 0.0, "endpoint_rmse": 1.0,
        "teacher_endpoint_rmse": 0.5, "endpoint_zero_rmse": 1.0,
        "endpoint_improvement": 0.0,
    })
    monkeypatch.setattr(module, "_latency", lambda model: {"warmups": 100, "measurements": 1000, "p99_ms": 1.0})
    monkeypatch.setattr(module, "save_student_artifact", lambda *args, **kwargs: pytest.fail("failed real gate published an artifact"))
    output = tmp_path / "student"

    assert module.main([
        "--dataset-manifest", str(tmp_path / "aggregate_manifest.json"),
        "--ensemble-dir", str(tmp_path / "expert"), "--output-dir", str(output),
        "--epochs", "1", "--hidden", "8",
    ]) != 0
    assert not output.exists()
    report_path = tmp_path / ".student.resume-v1" / "diagnostics" / "nondeployable.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    declared = body.pop("report_sha256")
    assert declared == module.sha256(module._canonical_json(body)).hexdigest()


def test_student_objective_has_nonzero_temporal_acceleration_and_jerk_terms():
    module = _module()
    torch.manual_seed(9)
    model = FingertipMixtureNet(hidden=(16, 16))
    inputs = torch.zeros(2, 42, dtype=torch.float32)
    target = torch.zeros(2, 20, 5, 3, dtype=torch.float32)
    samples = torch.zeros(2, 2, 20, 5, 3, dtype=torch.float32)

    total, label, sampled, acceleration, jerk = module.student_distillation_objective(
        model(inputs), target, samples
    )

    assert acceleration > 0.0 and jerk > 0.0
    torch.testing.assert_close(
        total,
        0.5 * label + 0.5 * sampled
        + module.DISTILLATION_CONFIG["acceleration_weight"] * acceleration
        + module.DISTILLATION_CONFIG["jerk_weight"] * jerk,
    )


def test_comparison_gate_rejects_negative_or_unversioned_report_values():
    result = _eval_module().accept_prior_comparison(
        {
            "nll_improvement_fraction": 0.2,
            "prior_off_jerk_p95": -1.0,
            "prior_on_jerk_p95": -2.0,
            "prior_off_task_success": 0.9,
            "prior_on_task_success": 0.9,
            "prior_off_safety_rejections": 0,
            "prior_on_safety_rejections": 0,
        }
    )

    assert not result.accepted
    assert result.reason == "invalid_metrics"


def test_comparison_gate_requires_matching_evaluation_provenance_not_only_counts():
    module = _eval_module()
    provenance = {
        "evaluation_manifest_sha256": "a" * 64,
        "scenario_definition_sha256": "b" * 64,
        "trial_set_sha256": "c" * 64,
        "safety_definition_sha256": "d" * 64,
        "controller_contract_sha256": "e" * 64,
    }
    metrics = {
        "comparison_format_version": 1, "provenance_format_version": 1, "trial_count": 3,
        "nll_improvement_fraction": 0.2, "prior_off_jerk_p95": 1.0, "prior_on_jerk_p95": 0.9,
        "prior_off_task_success": 1.0, "prior_on_task_success": 1.0,
        "prior_off_safety_rejections": 0, "prior_on_safety_rejections": 0,
        "prior_off_provenance": {**provenance, "prior_mode": "disabled", "prior_config_sha256": "0" * 64, "student_artifact_sha256": "0" * 64}, "prior_on_provenance": {**provenance, "trial_set_sha256": "f" * 64, "prior_mode": "enabled", "prior_config_sha256": "1" * 64, "student_artifact_sha256": "2" * 64},
    }
    assert module.accept_prior_comparison(metrics).reason == "provenance_mismatch"


def test_comparison_gate_requires_explicit_disabled_and_enabled_prior_identity():
    module = _eval_module()
    shared = {"evaluation_manifest_sha256": "a" * 64, "scenario_definition_sha256": "b" * 64, "trial_set_sha256": "c" * 64, "safety_definition_sha256": "d" * 64, "controller_contract_sha256": "e" * 64}
    metric = {"comparison_format_version": 1, "provenance_format_version": 1, "trial_count": 2, "nll_improvement_fraction": .2, "prior_off_jerk_p95": 1., "prior_on_jerk_p95": .9, "prior_off_task_success": 1., "prior_on_task_success": 1., "prior_off_safety_rejections": 0, "prior_on_safety_rejections": 0, "prior_off_provenance": {**shared, "prior_mode": "disabled", "prior_config_sha256": "0" * 64, "student_artifact_sha256": "0" * 64}, "prior_on_provenance": {**shared, "prior_mode": "enabled", "prior_config_sha256": "f" * 64, "student_artifact_sha256": "1" * 64}}
    assert module.accept_prior_comparison(metric).accepted
    metric["prior_on_provenance"].pop("prior_config_sha256")
    assert module.accept_prior_comparison(metric).reason == "invalid_metrics"


def test_evaluator_cli_accepts_valid_disabled_to_enabled_reports(tmp_path: Path):
    import json, subprocess, os
    shared = {"evaluation_manifest_sha256": "a" * 64, "scenario_definition_sha256": "b" * 64, "trial_set_sha256": "c" * 64, "safety_definition_sha256": "d" * 64, "controller_contract_sha256": "e" * 64}
    off = {"report_format_version": 1, "executed_fingertip_nll": 2.0, "jerk_p95": 1.0, "task_success": 1.0, "safety_rejections": 0, "trial_count": 2, "provenance": {**shared, "prior_mode": "disabled", "prior_config_sha256": "0" * 64, "student_artifact_sha256": "0" * 64}}
    on = {**off, "executed_fingertip_nll": 1.0, "jerk_p95": .9, "provenance": {**shared, "prior_mode": "enabled", "prior_config_sha256": "f" * 64, "student_artifact_sha256": "1" * 64}}
    off_path, on_path, output = tmp_path / "off.json", tmp_path / "on.json", tmp_path / "result.json"
    off_path.write_text(json.dumps(off)); on_path.write_text(json.dumps(on))
    done = subprocess.run([sys.executable, str(EVAL_SCRIPT), "--prior-off-report", str(off_path), "--prior-on-report", str(on_path), "--output", str(output)], env={**os.environ, "PYTHONPATH": str(EVAL_SCRIPT.parents[1])}, capture_output=True, text=True)
    assert done.returncode == 0 and json.loads(output.read_text())["acceptance"]["accepted"] is True


def test_distill_help_is_offline_and_does_not_create_output(tmp_path: Path):
    import subprocess
    import sys
    import os

    output = tmp_path / "must-stay-missing"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1])},
    )

    assert completed.returncode == 0
    assert "offline" in completed.stdout.lower()
    assert not output.exists()
