from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"
VERIFY = ROOT / "scripts/m1_dual_panda_o6_verify.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("m1_dual_panda_o6_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_layer_verifier_freezes_test_groups_gpu0_and_exact_formal_matrix():
    source = VERIFY.read_text(encoding="utf-8")

    for token in (
        'choices=("cpu", "qp", "smoke", "formal", "all")',
        "CPU_TESTS",
        "PURE_QP_TESTS",
        '"--device"',
        '"cuda:0"',
        '"--seeds"',
        '"42"',
        '"43"',
        '"44"',
        '"--trials-per-seed"',
        '"10"',
        '"--jsonl"',
        '"--manifest"',
        '"formal_aggregate.manifest.json"',
    ):
        assert token in source
    assert ".learn(" not in source
    assert "scripts/train.py" not in source


def test_probe_exposes_formal_jsonl_and_sha_pinned_manifest():
    source = PROBE.read_text(encoding="utf-8")
    for token in (
        "--jsonl",
        "--manifest",
        "_atomic_jsonl",
        "_artifact_manifest",
        "trials_jsonl",
        "aggregate_report",
    ):
        assert token in source


def test_formal_jsonl_and_manifest_pin_exact_artifact_bytes(tmp_path):
    probe = _load_probe_module()
    rows = [{"seed": 42, "trial_index": 0}, {"seed": 43, "trial_index": 0}]
    report = {
        "passed": True,
        "acceptance_mode": "formal",
        "aggregate": {"accepted": True, "observed_trial_count": 2},
        "metadata": {
            "asset_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "git_ref": "c" * 40,
            "isaac_version": "5.1.0.0",
            "command": "probe --seeds 42 43",
        },
    }
    report_path = tmp_path / "report.json"
    jsonl_path = tmp_path / "trials.jsonl"
    manifest_path = tmp_path / "aggregate.manifest.json"
    probe._atomic_json(report_path, report)
    probe._atomic_jsonl(jsonl_path, rows)
    manifest = probe._artifact_manifest(
        report, report_path=report_path, jsonl_path=jsonl_path
    )
    probe._atomic_json(manifest_path, manifest)

    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == rows
    assert manifest["status"] == "passed"
    assert manifest["pins"]["trials_jsonl"]["sha256"] == probe._sha256(jsonl_path)
    assert manifest["pins"]["aggregate_report"]["sha256"] == probe._sha256(report_path)


def test_probe_summary_has_prior_atomic_fields_and_disabled_nulls():
    probe = _load_probe_module()
    summary = probe.make_probe_summary(configured=False)
    assert {
        "prior_enabled",
        "prior_disabled_reason",
        "prior_component",
        "prior_probability",
        "prior_precision_min",
        "prior_precision_max",
        "prior_soft_cost",
        "prior_inference_p99_ms",
        "prior_baseline_tip_velocity_delta_norm",
        "prior_qp_rejected_count",
        "prior_fallback_count",
    } <= summary.keys()
    assert summary["prior_enabled"] is False
    assert summary["prior_disabled_reason"] == "not_configured"
    for key in (
        "prior_component",
        "prior_probability",
        "prior_precision_min",
        "prior_precision_max",
        "prior_soft_cost",
        "prior_inference_p99_ms",
        "prior_baseline_tip_velocity_delta_norm",
        "prior_qp_rejected_count",
        "prior_fallback_count",
        "prior_variance_min",
        "prior_variance_mean",
        "prior_variance_max",
    ):
        assert summary[key] is None


def test_probe_parser_default_and_no_observation_counts_are_real_not_fabricated():
    probe = _load_probe_module()
    assert probe._parser().parse_args([]).fingertip_prior_artifact is None
    configured = probe.make_probe_summary(configured=True)
    final = probe.finalize_probe_prior_summary(configured)
    assert final["prior_qp_rejected_count"] is None
    assert final["prior_fallback_count"] is None
    assert final["prior_disabled_reason"] == "no_relevant_observation"


def test_probe_prior_diagnostics_are_side_isolated_and_p99_uses_enabled_samples_only():
    probe = _load_probe_module()
    left = probe.make_probe_summary(configured=True)
    right = probe.make_probe_summary(configured=True)
    probe.accumulate_probe_prior_diagnostics(
        left,
        SimpleNamespace(
            prior_configured=True,
            prior_enabled=True,
            prior_fallback_reason=None,
            prior_component=2,
            prior_probability=0.75,
            prior_precision_min=1.0,
            prior_precision_max=4.0,
            prior_variance_min=0.25,
            prior_variance_mean=0.625,
            prior_variance_max=1.0,
            prior_cost=0.5,
            prior_inference_ms=2.0,
            regularized_tip_velocity_delta_norm=0.25,
            prior_qp_accepted=True,
        ),
    )
    probe.accumulate_probe_prior_diagnostics(
        left,
        SimpleNamespace(
            prior_configured=True,
            prior_enabled=False,
            prior_fallback_reason="prior_qp_rejected",
            prior_component=None,
            prior_probability=None,
            prior_precision_min=None,
            prior_precision_max=None,
            prior_cost=None,
            prior_inference_ms=99.0,
            regularized_tip_velocity_delta_norm=None,
            prior_qp_accepted=False,
        ),
    )
    assert probe.finalize_probe_prior_summary(left)["prior_inference_p99_ms"] == pytest.approx(2.0)
    assert probe.finalize_probe_prior_summary(left)["prior_qp_rejected_count"] == 1
    assert probe.finalize_probe_prior_summary(left)["prior_variance_mean"] == pytest.approx(0.625)
    assert probe.finalize_probe_prior_summary(right)["prior_enabled"] is False
    assert probe.finalize_probe_prior_summary(right)["prior_disabled_reason"] == "no_relevant_observation"


def test_artifact_validation_is_safe_and_does_not_create_a_worker(tmp_path):
    from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import (
        validate_student_artifact,
    )

    with pytest.raises(ValueError):
        validate_student_artifact(tmp_path / "missing")


def test_probe_closes_wrapper_when_reset_raises():
    probe = _load_probe_module()

    class ExplodingWrapper:
        instance = None

        def __init__(self, _env, **_kwargs):
            type(self).instance = self
            self.closed = []

        def reset(self, *, seed):
            raise RuntimeError(f"reset {seed}")

        def close(self, *, close_env):
            self.closed.append(close_env)

    with pytest.raises(RuntimeError, match="reset 3"):
        probe._run_trial(
            object(), ExplodingWrapper, seed=3, trial_index=0, steps=1,
            mode="teacher", latent_artifact=None, fingertip_prior_artifact=None,
            runtime_factory=object,
        )
    assert ExplodingWrapper.instance.closed == [False]


def test_wrapper_closes_first_prior_when_second_artifact_construction_fails(monkeypatch):
    # The failure must happen before the wrapper reaches any Isaac-dependent
    # adapter code, so a minimal asset boundary is sufficient for this fake env.
    assets = ModuleType("go2_pvcnn.assets")
    assets.M1_FOOT_BODY_NAMES = ()
    asset_layout = ModuleType("go2_pvcnn.assets.m1_dual_panda_o6")
    for name, value in {
        "LEFT_O6_ACTIVE_JOINT_NAMES": (),
        "LEFT_O6_FINGERTIP_BODY_NAMES": (),
        "LEFT_O6_PALM_BODY_NAME": "left_palm",
        "LEFT_PANDA_ACTIVE_JOINT_NAMES": (),
        "LEFT_PANDA_WRIST_BODY_NAME": "left_wrist",
        "M1_BASE_ACTIVE_JOINT_NAMES": (),
        "M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES": (),
        "M1_DUAL_PANDA_O6_BASE_BODY_NAME": "base",
        "M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME": "platform",
        "O6_MIMIC_MAP": {},
        "RIGHT_O6_ACTIVE_JOINT_NAMES": (),
        "RIGHT_O6_FINGERTIP_BODY_NAMES": (),
        "RIGHT_O6_PALM_BODY_NAME": "right_palm",
        "RIGHT_PANDA_ACTIVE_JOINT_NAMES": (),
        "RIGHT_PANDA_WRIST_BODY_NAME": "right_wrist",
        "resolve_active_joint_ids": lambda _names: (),
    }.items():
        setattr(asset_layout, name, value)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets", assets)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets.m1_dual_panda_o6", asset_layout)
    wrapper_spec = importlib.util.spec_from_file_location(
        "m1_dual_panda_o6_wrapper",
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py",
    )
    assert wrapper_spec is not None and wrapper_spec.loader is not None
    wrapper_module = importlib.util.module_from_spec(wrapper_spec)
    monkeypatch.setitem(sys.modules, "m1_dual_panda_o6_wrapper", wrapper_module)
    wrapper_spec.loader.exec_module(wrapper_module)

    class FirstPrior:
        closed = False

        def close(self):
            self.closed = True

        def target(self, *_args):
            raise AssertionError("fake prior must not be queried during startup")

    first = FirstPrior()
    calls = 0

    def construct(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise ValueError("strict artifact failure")

    monkeypatch.setattr(wrapper_module, "_construct_fingertip_prior", construct)

    class FakeEnv:
        @property
        def unwrapped(self):
            raise AssertionError("artifact failure must precede environment startup")

    with pytest.raises(ValueError, match="strict artifact failure"):
        wrapper_module.M1DualPandaO6BimanualWrapper(
            FakeEnv(), fingertip_prior_artifact="approved-artifact"
        )
    assert first.closed is True

    left, right = FirstPrior(), FirstPrior()
    constructed = [left, right]
    monkeypatch.setattr(
        wrapper_module, "_construct_fingertip_prior", lambda _path: constructed.pop(0)
    )
    monkeypatch.setattr(
        wrapper_module, "M1DualPandaO6SnapshotAdapter", lambda *_args, **_kwargs: object()
    )

    class ReadyEnv:
        close_calls = 0

        @property
        def unwrapped(self):
            return SimpleNamespace(num_envs=1)

        def close(self):
            self.close_calls += 1

    env = ReadyEnv()
    wrapper = wrapper_module.M1DualPandaO6BimanualWrapper(
        env, fingertip_prior_artifact="approved-artifact"
    )
    assert wrapper._fingertip_priors == (left, right)
    assert left is not right
    wrapper.close(close_env=False)
    wrapper.close(close_env=False)
    assert left.closed is True and right.closed is True
    assert env.close_calls == 0

    original_sync = wrapper_module.M1DualPandaO6BimanualWrapper._sync_legacy_aliases
    alias_priors = [FirstPrior(), FirstPrior()]
    alias_queue = list(alias_priors)
    monkeypatch.setattr(
        wrapper_module, "_construct_fingertip_prior", lambda _path: alias_queue.pop(0)
    )
    monkeypatch.setattr(
        wrapper_module.M1DualPandaO6BimanualWrapper,
        "_sync_legacy_aliases",
        lambda _self: (_ for _ in ()).throw(RuntimeError("late aliases")),
    )
    with pytest.raises(RuntimeError, match="late aliases"):
        wrapper_module.M1DualPandaO6BimanualWrapper(
            ReadyEnv(), fingertip_prior_artifact="approved-artifact"
        )
    assert all(prior.closed for prior in alias_priors)

    monkeypatch.setattr(
        wrapper_module.M1DualPandaO6BimanualWrapper,
        "_sync_legacy_aliases",
        original_sync,
    )
    latent_priors = [FirstPrior(), FirstPrior()]
    latent_queue = list(latent_priors)
    monkeypatch.setattr(
        wrapper_module, "_construct_fingertip_prior", lambda _path: latent_queue.pop(0)
    )
    monkeypatch.setattr(
        wrapper_module.LatentRuntime,
        "from_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("latent load")),
    )
    with pytest.raises(RuntimeError, match="latent load"):
        wrapper_module.M1DualPandaO6BimanualWrapper(
            ReadyEnv(),
            mode="latent",
            latent_artifact="latent",
            fingertip_prior_artifact="approved-artifact",
        )
    assert all(prior.closed for prior in latent_priors)
