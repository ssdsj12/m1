"""Offline distillation of a compact, SHA-pinned fingertip mixture student."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import save_student_artifact
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    FINGER_ORDER, LEFT_REFLECTION, MODEL_INPUT_FIELD_ORDER, MIXTURE_OUTPUT_AXIS_ORDER,
    PHASE_ORDER, PRIOR_DT, MixtureDistribution, StudentArtifactMetadata,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.download import sha256_file
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.model import (
    FingertipMixtureNet, mixture_log_prob, mixture_nll, temporal_regularizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import verify_aggregate_manifest

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)
from m1_dual_panda_o6_train_fingertip_expert import (
    GroupShardDataset, _canonical_json, _ensemble_manifest_sha256, _load_group_split,
    _resolve_manifest, _set_seed,
)

DISTILLATION_CONFIG = {"label_nll_weight": 0.5, "teacher_sample_weight": 0.5, "acceleration_weight": 1e-5, "jerk_weight": 1e-7}


def sample_mixture(
    distribution: MixtureDistribution, *, samples_per_state: int, seed: int
) -> torch.Tensor:
    """Draw canonical fixed-seed samples, preserving no component index correspondence."""

    if distribution.logits.ndim != 2:
        raise ValueError("teacher distribution must have a batch dimension")
    if type(samples_per_state) is not int or samples_per_state <= 0 or type(seed) is not int:
        raise ValueError("samples_per_state and seed must be valid integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = distribution.logits.detach().cpu().softmax(-1)
    component = torch.multinomial(probabilities, samples_per_state, replacement=True, generator=generator)
    batch, components = component.shape[0], distribution.logits.shape[-1]
    mean = distribution.mean.detach().cpu().unsqueeze(1).expand(batch, samples_per_state, components, 20, 5, 3)
    log_std = distribution.log_std.detach().cpu().unsqueeze(1).expand_as(mean)
    index = component[:, :, None, None, None, None].expand(batch, samples_per_state, 1, 20, 5, 3)
    selected_mean = mean.gather(2, index).squeeze(2)
    selected_log_std = log_std.gather(2, index).squeeze(2)
    return selected_mean + selected_log_std.exp() * torch.randn(selected_mean.shape, generator=generator, dtype=torch.float32)


def distribution_distillation_loss(student: MixtureDistribution, teacher_samples: torch.Tensor) -> torch.Tensor:
    """Score fixed teacher samples under the student distribution, never aligning components."""

    if not isinstance(teacher_samples, torch.Tensor) or teacher_samples.dtype != torch.float32:
        raise TypeError("teacher_samples must be a float32 tensor")
    if student.logits.ndim != 2 or teacher_samples.ndim != 5:
        raise ValueError("teacher_samples must have shape (batch, samples, 20, 5, 3)")
    batch, samples = teacher_samples.shape[:2]
    if teacher_samples.shape != (student.logits.shape[0], samples, 20, 5, 3) or samples <= 0:
        raise ValueError("teacher_samples must have shape (batch, samples, 20, 5, 3)")
    if teacher_samples.device != student.mean.device or not torch.isfinite(teacher_samples).all().item():
        raise ValueError("teacher_samples must be finite and on the student device")
    components = student.logits.shape[-1]
    expanded = MixtureDistribution(
        logits=student.logits.unsqueeze(1).expand(batch, samples, components),
        mean=student.mean.unsqueeze(1).expand(batch, samples, components, 20, 5, 3),
        log_std=student.log_std.unsqueeze(1).expand(batch, samples, components, 20, 5, 3),
    )
    return -mixture_log_prob(expanded, teacher_samples).mean()


def student_distillation_objective(student: MixtureDistribution, target: torch.Tensor, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label, sampled = mixture_nll(student, target), distribution_distillation_loss(student, samples)
    acceleration, jerk = temporal_regularizer(student, dt=PRIOR_DT)
    total = DISTILLATION_CONFIG["label_nll_weight"] * label + DISTILLATION_CONFIG["teacher_sample_weight"] * sampled + DISTILLATION_CONFIG["acceleration_weight"] * acceleration + DISTILLATION_CONFIG["jerk_weight"] * jerk
    return total, label, sampled, acceleration, jerk


@dataclass(frozen=True)
class _Ensemble:
    models: tuple[FingertipMixtureNet, ...]
    manifest_sha256: str
    dataset_sha256: str
    hidden: tuple[int, ...]
    teacher_seed: int
    synthetic: bool


def _regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")


def _load_ensemble(path: str | Path, *, expected_dataset_sha: str, allow_synthetic: bool) -> _Ensemble:
    root = Path(path).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("ensemble directory must be a regular directory")
    manifest_path = root / "ensemble_manifest.json"
    _regular(manifest_path, "ensemble manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ensemble manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("ensemble manifest is invalid")
    declared = manifest.get("ensemble_manifest_sha256")
    body = dict(manifest)
    body.pop("ensemble_manifest_sha256", None)
    if type(declared) is not str or declared != _ensemble_manifest_sha256(body):
        raise ValueError("ensemble manifest SHA-256 mismatch")
    if manifest.get("format_version") != 1 or manifest.get("dataset_aggregate_sha256") != expected_dataset_sha:
        raise ValueError("ensemble does not match verified dataset aggregate")
    synthetic = manifest.get("synthetic_smoke") is True
    if not allow_synthetic and (synthetic or manifest.get("production_deployable") is not True):
        raise ValueError("ensemble is not deployable")
    hidden_value = manifest.get("hidden")
    records = manifest.get("members")
    if type(hidden_value) is not list or not hidden_value or any(type(width) is not int or width <= 0 for width in hidden_value):
        raise ValueError("ensemble architecture is invalid")
    if type(records) is not list or len(records) < 2:
        raise ValueError("ensemble members are invalid")
    hidden = tuple(hidden_value)
    models: list[FingertipMixtureNet] = []
    for index, record in enumerate(records):
        expected_path = f"checkpoints/member-{index:02d}-best.pt"
        if (
            type(record) is not dict or record.get("member_index") != index or type(record.get("seed")) is not int
            or record.get("checkpoint") != expected_path or type(record.get("checkpoint_sha256")) is not str
        ):
            raise ValueError("ensemble member record is invalid")
        checkpoint = root / expected_path
        _regular(checkpoint, "ensemble checkpoint")
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("ensemble checkpoint SHA-256 mismatch")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise ValueError("ensemble checkpoint could not be safely loaded") from error
        if not isinstance(state, dict) or state.get("format_version") != 1 or state.get("member_index") != index or state.get("seed") != record["seed"] or state.get("hidden") != hidden or state.get("dataset_aggregate_sha256") != expected_dataset_sha or not isinstance(state.get("model_state"), dict):
            raise ValueError("ensemble checkpoint state is invalid")
        model = FingertipMixtureNet(hidden=hidden)
        expected_state = model.state_dict()
        weights = state["model_state"]
        if set(weights) != set(expected_state):
            raise ValueError("ensemble checkpoint model state keys are invalid")
        for key, reference in expected_state.items():
            value = weights[key]
            if not isinstance(value, torch.Tensor) or value.dtype != reference.dtype or value.shape != reference.shape or not torch.isfinite(value).all().item():
                raise ValueError("ensemble checkpoint model state is invalid")
        model.load_state_dict(weights, strict=True)
        model.eval()
        models.append(model)
    return _Ensemble(tuple(models), declared, expected_dataset_sha, hidden, int(records[0]["seed"]), synthetic)


def _ensemble_outputs(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor) -> tuple[MixtureDistribution, ...]:
    distributions = [model(inputs) for model in models]
    if not distributions:
        raise ValueError("ensemble must contain a member")
    return tuple(distributions)


def _ensemble_log_prob(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact member-average density while each member remains a frozen 4-mixture."""

    values = torch.stack([mixture_log_prob(distribution, target) for distribution in _ensemble_outputs(models, inputs)], dim=1)
    return torch.logsumexp(values, dim=1) - math.log(values.shape[1])


def _ensemble_mean(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor) -> torch.Tensor:
    distributions = _ensemble_outputs(models, inputs)
    count = len(distributions)
    means = torch.stack([distribution.mean for distribution in distributions], dim=1)
    weights = torch.stack([distribution.logits.softmax(-1) for distribution in distributions], dim=1) / count
    return (weights[..., None, None, None] * means).sum(dim=(1, 2))


def _sample_ensemble(models: Sequence[FingertipMixtureNet], inputs: torch.Tensor, *, samples_per_state: int, seed: int) -> torch.Tensor:
    distributions = _ensemble_outputs(models, inputs)
    if type(samples_per_state) is not int or samples_per_state <= 0 or type(seed) is not int:
        raise ValueError("samples_per_state and seed must be valid integers")
    logits = torch.stack([distribution.logits.detach().cpu() for distribution in distributions], dim=1)
    means = torch.stack([distribution.mean.detach().cpu() for distribution in distributions], dim=1)
    log_stds = torch.stack([distribution.log_std.detach().cpu() for distribution in distributions], dim=1)
    batch = logits.shape[0]
    flat_logits = logits.reshape(batch, -1)
    flat_means = means.reshape(batch, -1, 20, 5, 3)
    flat_log_stds = log_stds.reshape_as(flat_means)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.multinomial(flat_logits.softmax(-1), samples_per_state, replacement=True, generator=generator)
    index = selected[:, :, None, None, None, None].expand(batch, samples_per_state, 1, 20, 5, 3)
    means_by_sample = flat_means.unsqueeze(1).expand(batch, samples_per_state, flat_means.shape[1], 20, 5, 3)
    stds_by_sample = flat_log_stds.unsqueeze(1).expand_as(means_by_sample)
    return means_by_sample.gather(2, index).squeeze(2) + stds_by_sample.gather(2, index).squeeze(2).exp() * torch.randn((batch, samples_per_state, 20, 5, 3), generator=generator, dtype=torch.float32)


def _teacher_samples(models: Sequence[FingertipMixtureNet], dataset: GroupShardDataset, *, samples_per_state: int, seed: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            chunks.append(_sample_ensemble(models, dataset.inputs[start:start + 128], samples_per_state=samples_per_state, seed=seed + start))
    return torch.cat(chunks, dim=0)


def _predictive_mean(distribution: MixtureDistribution) -> torch.Tensor:
    return (distribution.logits.softmax(-1)[..., None, None, None] * distribution.mean).sum(dim=1)


def _metrics(model: FingertipMixtureNet, ensemble: _Ensemble, dataset: GroupShardDataset) -> dict[str, float]:
    first_sq = endpoint_sq = teacher_endpoint_sq = zero_first_sq = zero_endpoint_sq = student_sum = teacher_sum = 0.0
    samples = 0
    with torch.no_grad():
        for start in range(0, len(dataset), 128):
            inputs, target = dataset.inputs[start:start + 128], dataset.targets[start:start + 128]
            student = model(inputs)
            teacher_log_prob = _ensemble_log_prob(ensemble.models, inputs, target)
            student_sum += float((-mixture_log_prob(student, target)).sum().item())
            teacher_sum += float((-teacher_log_prob).sum().item())
            mean = _predictive_mean(student)
            first_sq += float((mean[:, 0] - target[:, 0]).square().sum().item())
            zero_first_sq += float(target[:, 0].square().sum().item())
            endpoint = mean.sum(dim=1) * PRIOR_DT
            teacher_endpoint_mean = _ensemble_mean(ensemble.models, inputs).sum(dim=1) * PRIOR_DT
            target_endpoint = target.sum(dim=1) * PRIOR_DT
            endpoint_sq += float((endpoint - target_endpoint).square().sum().item())
            teacher_endpoint_sq += float((teacher_endpoint_mean - target_endpoint).square().sum().item())
            zero_endpoint_sq += float(target_endpoint.square().sum().item())
            samples += target.shape[0]
    if samples == 0 or zero_first_sq <= 0.0 or zero_endpoint_sq <= 0.0:
        raise ValueError("evaluation set cannot produce zero baselines")
    dimensions = samples * 20 * 5 * 3
    first_count = samples * 5 * 3
    student_nll, teacher_nll = student_sum / samples, teacher_sum / samples
    endpoint_rmse = math.sqrt(endpoint_sq / first_count)
    teacher_endpoint_rmse = math.sqrt(teacher_endpoint_sq / first_count)
    zero_endpoint = math.sqrt(zero_endpoint_sq / first_count)
    first_rmse, zero_first = math.sqrt(first_sq / first_count), math.sqrt(zero_first_sq / first_count)
    return {
        "student_nll": student_nll,
        "teacher_nll": teacher_nll,
        "nll_delta_per_dim": (student_nll - teacher_nll) / 300.0,
        "first_step_velocity_rmse": first_rmse,
        "first_step_zero_rmse": zero_first,
        "first_step_improvement": (zero_first - first_rmse) / zero_first,
        "endpoint_rmse": endpoint_rmse,
        "teacher_endpoint_rmse": teacher_endpoint_rmse,
        "endpoint_zero_rmse": zero_endpoint,
        "endpoint_improvement": (zero_endpoint - endpoint_rmse) / zero_endpoint,
    }


def _latency(model: FingertipMixtureNet) -> dict[str, object]:
    model.eval()
    inputs = torch.zeros(1, 42, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(100):
            model(inputs)
        timings: list[int] = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            model(inputs)
            timings.append(time.perf_counter_ns() - start)
    return {"warmups": 100, "measurements": 1000, "p99_ms": float(np.quantile(np.asarray(timings, dtype=np.float64), 0.99, method="higher") / 1_000_000.0)}


def _git_commit() -> str:
    completed = subprocess.run(("git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"), check=True, capture_output=True, text=True)
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve code commit")
    return value


def _train_student(train: GroupShardDataset, ensemble: _Ensemble, *, hidden: tuple[int, ...], epochs: int, batch_size: int, learning_rate: float, samples_per_state: int, seed: int) -> FingertipMixtureNet:
    _set_seed(seed)
    teacher_samples = _teacher_samples(ensemble.models, train, samples_per_state=samples_per_state, seed=seed)
    model = FingertipMixtureNet(hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(train.inputs, train.targets, teacher_samples)
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        model.train()
        for inputs, target, samples in loader:
            loss, _, _, _, _ = student_distillation_objective(model(inputs), target, samples)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("student distillation loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _synthetic_ensemble(stage: Path, *, epochs: int) -> tuple[Path, Path]:
    teacher = stage / "synthetic-teacher"
    command = [sys.executable, str(Path(__file__).with_name("m1_dual_panda_o6_train_fingertip_expert.py")), "--synthetic-smoke", "--output-dir", str(teacher), "--epochs", str(epochs), "--hidden", "16,16", "--member-seeds", "1701,2718"]
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})
    return teacher / "synthetic-shards" / "aggregate_manifest.json", teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline compact fingertip-prior distillation; no network or downloads.")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--ensemble-dir")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples-per-state", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", default="64,64")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0 or args.samples_per_state <= 0:
        raise ValueError("distillation values are invalid")
    if args.synthetic_smoke and (args.dataset_manifest or args.ensemble_dir):
        raise ValueError("synthetic smoke cannot consume caller-provided inputs")
    if not args.synthetic_smoke and (not args.dataset_manifest or not args.ensemble_dir):
        raise ValueError("real distillation requires dataset manifest and deployable ensemble")
    try:
        hidden = tuple(int(value) for value in args.hidden.split(","))
    except ValueError as error:
        raise ValueError("hidden must be comma-separated positive integer widths") from error
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden widths are invalid")
    destination = Path(args.output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.distill-", dir=destination.parent))
    try:
        if args.synthetic_smoke:
            manifest_path, ensemble_dir = _synthetic_ensemble(stage, epochs=min(args.epochs, 2))
        else:
            manifest_path, ensemble_dir = _resolve_manifest(args.dataset_manifest), Path(args.ensemble_dir).resolve()
        document = verify_aggregate_manifest(manifest_path.parent)
        train, train_doc = _load_group_split(manifest_path, "train")
        test, test_doc = _load_group_split(manifest_path, "test")
        if train_doc["aggregate_sha256"] != document["aggregate_sha256"] or test_doc["aggregate_sha256"] != document["aggregate_sha256"]:
            raise ValueError("dataset aggregate changed during verified loading")
        ensemble = _load_ensemble(ensemble_dir, expected_dataset_sha=document["aggregate_sha256"], allow_synthetic=bool(args.synthetic_smoke))
        model = _train_student(train, ensemble, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, samples_per_state=args.samples_per_state, seed=args.seed)
        metrics = _metrics(model, ensemble, test)
        latency = _latency(model)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("student metrics are non-finite")
        production_approved = (not args.synthetic_smoke and not ensemble.synthetic and metrics["nll_delta_per_dim"] <= 0.05 and metrics["endpoint_rmse"] <= 1.05 * metrics["teacher_endpoint_rmse"] and metrics["first_step_improvement"] >= 0.10 and metrics["endpoint_improvement"] >= 0.10 and latency["p99_ms"] < 2.0)
        metrics["production_approved"] = production_approved
        metrics["deterministic_repeat_verified"] = True
        metadata = StudentArtifactMetadata(
            format_version=1, input_dim=42, mixture_components=4, horizon=20, dt=PRIOR_DT,
            finger_order=FINGER_ORDER, phase_order=PHASE_ORDER,
            mirror_matrix=torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
            dataset_aggregate_sha256=document["aggregate_sha256"], teacher_ensemble_manifest_sha256=ensemble.manifest_sha256,
            teacher_seed=ensemble.teacher_seed, distillation_seed=args.seed, code_commit=_git_commit(), weight_sha256="0" * 64,
            hidden=hidden, input_field_order=MODEL_INPUT_FIELD_ORDER, output_axis_order=MIXTURE_OUTPUT_AXIS_ORDER,
        )
        provenance = {"nonproduction_synthetic": bool(args.synthetic_smoke), "dataset_aggregate_sha256": document["aggregate_sha256"], "teacher_ensemble_manifest_sha256": ensemble.manifest_sha256}
        first = stage / "student"
        pinned = save_student_artifact(first, model=model, metadata=metadata, metrics=metrics, latency=latency, provenance=provenance)
        repeated_model = _train_student(train, ensemble, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, samples_per_state=args.samples_per_state, seed=args.seed)
        repeated_metrics = _metrics(repeated_model, ensemble, test)
        repeated_metrics["production_approved"] = production_approved
        repeated_metrics["deterministic_repeat_verified"] = True
        repeat = save_student_artifact(stage / "repeat", model=repeated_model, metadata=metadata, metrics=repeated_metrics, latency=_latency(repeated_model), provenance=provenance)
        first_metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
        repeat_metadata = json.loads((stage / "repeat" / "metadata.json").read_text(encoding="utf-8"))
        if pinned.weight_sha256 != repeat.weight_sha256 or first_metadata["reproducibility_fingerprint"] != repeat_metadata["reproducibility_fingerprint"]:
            raise RuntimeError("independent deterministic student artifact fingerprint mismatch")
        shutil.rmtree(stage / "repeat")
        os.replace(first, destination)
        print(json.dumps({"artifact": str(destination), "weight_sha256": pinned.weight_sha256, "metrics": metrics, "latency": latency, "production_approved": production_approved}, sort_keys=True))
        return 0
    except BaseException:
        if destination.exists() and destination.parent == stage.parent:
            # The destination is only created by the successful final replace; never remove a caller path.
            pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
