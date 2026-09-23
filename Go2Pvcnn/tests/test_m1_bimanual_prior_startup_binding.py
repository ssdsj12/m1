from __future__ import annotations

from hashlib import sha256
import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_m1_bimanual_expert_prior_runtime import production_artifact, _sample
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior import runtime
from go2_pvcnn.control.m1_bimanual_coordination.runtime import BimanualRuntime


ROOT = Path(__file__).resolve().parents[1]


def _binding_type():
    from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime_binding import (
        PreparedO6FingertipPriors,
    )
    return PreparedO6FingertipPriors


def _pin(path):
    return sha256((path / "metadata.json").read_bytes()).hexdigest()


def _load_entrypoint(name):
    path = ROOT / f"scripts/m1_dual_panda_o6_bimanual_{name}.py"
    spec = importlib.util.spec_from_file_location(f"startup_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_wrapper(monkeypatch):
    assets = ModuleType("go2_pvcnn.assets")
    assets.M1_FOOT_BODY_NAMES = ()
    layout = ModuleType("go2_pvcnn.assets.m1_dual_panda_o6")
    for name in (
        "LEFT_O6_ACTIVE_JOINT_NAMES", "LEFT_O6_FINGERTIP_BODY_NAMES",
        "LEFT_PANDA_ACTIVE_JOINT_NAMES", "M1_BASE_ACTIVE_JOINT_NAMES",
        "M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES", "RIGHT_O6_ACTIVE_JOINT_NAMES",
        "RIGHT_O6_FINGERTIP_BODY_NAMES", "RIGHT_PANDA_ACTIVE_JOINT_NAMES",
    ):
        setattr(layout, name, ())
    for name in (
        "LEFT_O6_PALM_BODY_NAME", "LEFT_PANDA_WRIST_BODY_NAME",
        "M1_DUAL_PANDA_O6_BASE_BODY_NAME", "M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME",
        "RIGHT_O6_PALM_BODY_NAME", "RIGHT_PANDA_WRIST_BODY_NAME",
    ):
        setattr(layout, name, name)
    layout.O6_MIMIC_MAP = {}
    layout.resolve_active_joint_ids = lambda _names: ()
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets", assets)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets.m1_dual_panda_o6", layout)
    spec = importlib.util.spec_from_file_location(
        "startup_wrapper", ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "startup_wrapper", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "M1DualPandaO6SnapshotAdapter", lambda *_a, **_k: object())
    return module


def test_bound_workers_survive_mutable_artifact_replacement(production_artifact, monkeypatch):
    pin = _pin(production_artifact)
    with _binding_type().from_artifact(
        production_artifact, expected_metadata_sha256=pin
    ) as binding:
        original = (production_artifact / "metadata.json").read_bytes()
        (production_artifact / "metadata.json").write_bytes(b"replaced after validation")
        monkeypatch.setattr(
            runtime, "load_student_artifact",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("mutable path reread")),
        )
        priors = binding.for_metadata_pin(pin)
        results = [prior.target(_sample(), torch.zeros(6, dtype=torch.float64)) for prior in priors]
        assert all(result.target is not None for result in results)
        assert torch.equal(results[0].target.mean_velocity, results[1].target.mean_velocity)
        assert sha256(original).hexdigest() == pin
    assert all(not prior._worker.process.is_alive() for prior in priors)


def test_replacement_between_left_and_right_binding_rejects_and_reaps_first(production_artifact, monkeypatch):
    pin = _pin(production_artifact)
    factory = runtime.FrozenO6FingertipPrior.from_artifact
    created = []

    def replace_after_first(path, **kwargs):
        prior = factory(path, **kwargs)
        created.append(prior)
        (path / "metadata.json").write_bytes(b"replacement before second binding")
        return prior

    monkeypatch.setattr(runtime.FrozenO6FingertipPrior, "from_artifact", replace_after_first)
    with pytest.raises(ValueError, match="expected metadata SHA-256 pin"):
        _binding_type().from_artifact(production_artifact, expected_metadata_sha256=pin)
    assert len(created) == 1
    assert not created[0]._worker.process.is_alive()


def test_wrapper_consumes_bound_workers_without_reloading_or_owning_them(production_artifact, monkeypatch):
    wrapper_module = _load_wrapper(monkeypatch)
    pin = _pin(production_artifact)
    with _binding_type().from_artifact(
        production_artifact, expected_metadata_sha256=pin
    ) as binding:
        (production_artifact / "metadata.json").write_bytes(b"replaced before scene")
        monkeypatch.setattr(
            wrapper_module, "_build_fingertip_priors",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("mutable path reread")),
        )
        env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=1))
        for _trial in range(2):
            wrapper = wrapper_module.M1DualPandaO6BimanualWrapper(
                env, fingertip_prior_artifact=production_artifact,
                fingertip_prior_metadata_sha256=pin, fingertip_prior_binding=binding,
            )
            priors = binding.for_metadata_pin(pin)
            assert wrapper.runtime.left_hand_mpc.expert_prior is priors[0]
            assert wrapper.runtime.right_hand_mpc.expert_prior is priors[1]
            wrapper.close(close_env=False)
            assert all(prior._worker.process.is_alive() for prior in priors)
    assert all(not prior._worker.process.is_alive() for prior in priors)


def test_wrapper_attaches_catalog_grasp_goal_provider_to_supplied_runtime(monkeypatch):
    wrapper_module = _load_wrapper(monkeypatch)
    provider = lambda _snapshot: None
    supplied_runtime = BimanualRuntime()
    wrapper = object.__new__(wrapper_module.M1DualPandaO6BimanualWrapper)
    wrapper._fingertip_priors = ()
    wrapper._grasp_goal_provider = provider

    result = wrapper._runtime_for_lane(supplied_runtime)

    assert result is supplied_runtime
    assert supplied_runtime._grasp_goal_provider is provider


def test_prepared_binding_rejects_construction_pin_mismatch_and_reuse_after_close(production_artifact):
    cls = _binding_type()
    with pytest.raises(TypeError, match="from_artifact"):
        cls(object(), object())
    pin = _pin(production_artifact)
    binding = cls.from_artifact(production_artifact, expected_metadata_sha256=pin)
    try:
        with pytest.raises(ValueError, match="pin mismatch"):
            binding.for_metadata_pin("f" * 64)
    finally:
        binding.close()
    with pytest.raises(ValueError, match="closed"):
        binding.for_metadata_pin(pin)


@pytest.mark.parametrize("name", ["play", "probe"])
@pytest.mark.parametrize("fail", [False, True])
def test_entrypoint_binds_before_scene_boundary_and_cleans_every_exit(
    name, fail, production_artifact, monkeypatch
):
    module = _load_entrypoint(name)
    pin = _pin(production_artifact)
    seen = []

    def scene_boundary(parser, binding):
        # AppLauncher import and scene creation occur only inside this function.
        # Both frozen workers must already exist before that boundary is crossed.
        priors = binding.for_metadata_pin(pin)
        seen.extend(priors)
        assert all(prior._worker.process.is_alive() for prior in priors)
        (production_artifact / "metadata.json").write_bytes(b"changed at scene startup")
        if fail:
            raise RuntimeError("launcher/scene failure")
        assert all(
            prior.target(_sample(), torch.zeros(6, dtype=torch.float64)).target is not None
            for prior in priors
        )
        return 0

    monkeypatch.setattr(module, "_run_with_fingertip_prior", scene_boundary)
    monkeypatch.setattr(sys, "argv", [
        name, "--fingertip-prior-artifact", str(production_artifact),
        "--fingertip-prior-metadata-sha256", pin,
    ])
    if fail:
        with pytest.raises(RuntimeError, match="launcher/scene failure"):
            module.main()
    else:
        assert module.main() == 0
    assert len(seen) == 2
    assert all(not prior._worker.process.is_alive() for prior in seen)


@pytest.mark.parametrize("name", ["play", "probe"])
def test_entrypoint_pin_mismatch_never_crosses_scene_boundary(name, production_artifact, monkeypatch):
    module = _load_entrypoint(name)
    monkeypatch.setattr(
        module, "_run_with_fingertip_prior",
        lambda *_a: (_ for _ in ()).throw(AssertionError("scene started before rejection")),
    )
    monkeypatch.setattr(sys, "argv", [
        name, "--fingertip-prior-artifact", str(production_artifact),
        "--fingertip-prior-metadata-sha256", "f" * 64,
    ])
    with pytest.raises(ValueError, match="expected metadata SHA-256 pin"):
        module.main()


@pytest.mark.parametrize("name", ["play", "probe"])
def test_real_launcher_import_boundary_sees_bound_workers_and_failure_cleans_them(
    name, production_artifact, monkeypatch
):
    module = _load_entrypoint(name)
    cls = _binding_type()
    pin = _pin(production_artifact)
    factory = cls.from_artifact
    prepared = []

    def record_binding(*args, **kwargs):
        binding = factory(*args, **kwargs)
        prepared.append(binding)
        return binding

    monkeypatch.setattr(cls, "from_artifact", record_binding)
    original_import = builtins.__import__
    launcher_module = ModuleType("isaaclab.app")

    class ExplodingLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, args):
            assert args.fingertip_prior_metadata_sha256 == pin
            raise RuntimeError("actual launcher construction failure")

    launcher_module.AppLauncher = ExplodingLauncher

    def checked_import(name, *args, **kwargs):
        if name == "isaaclab.app":
            assert len(prepared) == 1
            assert all(
                prior._worker.process.is_alive()
                for prior in prepared[0].for_metadata_pin(pin)
            )
            return launcher_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setattr(sys, "argv", [
        name, "--fingertip-prior-artifact", str(production_artifact),
        "--fingertip-prior-metadata-sha256", pin,
    ])
    with pytest.raises(RuntimeError, match="actual launcher construction failure"):
        module.main()
    assert len(prepared) == 1
    assert all(not prior._worker.process.is_alive() for prior in prepared[0]._priors)
