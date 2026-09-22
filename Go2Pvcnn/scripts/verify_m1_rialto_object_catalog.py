#!/usr/bin/env python3
"""Verify the local RialTo object catalog without downloading assets.

The verifier has two deliberately separate layers.  The offline layer checks
the pinned source manifest, materialized USDs, hashes, dependency closure and
catalog metadata.  The optional GPU0 layer is only started after the offline
layer has passed.  Missing prepared files therefore produce a durable
``preparation_required`` report, never a synthetic smoke pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "config" / "m1_object_catalog.json"
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "assets" / "m1_objects" / "rialto"
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "m1_rialto_object_catalog"
EXPECTED_CLASSES = ("bottle", "cup", "bowl", "book", "cube", "cylinder")
EXPECTED_SOURCE_BY_CLASS = {
    "bottle": "bottle_fixed.usd",
    "cup": "coffeecup.usdz",
    "bowl": "bowlnrack2.usd",
    "book": "book_fixed.usd",
    "cube": "box.glb",
    "cylinder": "poly.glb",
}
EXPECTED_RESOLVED_BY_CLASS = {
    "bottle": "sources/bottle_fixed.usd",
    "cup": "converted/coffeecup.usd",
    "bowl": "sources/bowlnrack2.usd",
    "book": "sources/book_fixed.usd",
    "cube": "converted/box.usd",
    "cylinder": "converted/poly.usd",
}
ALLOWED_COLLISION_PROFILES = frozenset({"box", "convex_decomposition"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PREPARATION_SCRIPT = "Go2Pvcnn/scripts/m1_rialto_object_assets.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _preparation_command(asset_root: Path) -> str:
    return (
        "PYTHONPATH=Go2Pvcnn python "
        f"{PREPARATION_SCRIPT} prepare --destination {asset_root} --allow-network"
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} root must be an object: {path}")
    return value


def partition_target_and_obstacles(
    instances: Sequence[Any], target_object_id: str | None
) -> tuple[Any | None, tuple[Any, ...]]:
    """Return the explicit target and stable obstacle partition."""

    if isinstance(instances, (str, bytes)):
        raise TypeError("instances must be a sequence of ObjectInstance values")
    ordered = []
    seen: set[str] = set()
    for instance in instances:
        object_id = getattr(instance, "object_id", None)
        if not isinstance(object_id, str) or not object_id:
            raise TypeError("instances must contain ObjectInstance values")
        if object_id in seen:
            raise ValueError(f"duplicate object_id: {object_id!r}")
        object_class = getattr(instance, "object_class", None)
        if object_class not in EXPECTED_CLASSES:
            raise ValueError(f"unknown object class: {object_class!r}")
        seen.add(object_id)
        if bool(getattr(instance, "enabled", False)):
            ordered.append(instance)
    ordered.sort(key=lambda instance: instance.object_id)
    if target_object_id is None:
        return None, tuple(ordered)
    if not isinstance(target_object_id, str) or not target_object_id:
        raise TypeError("target_object_id must be a non-empty string or None")
    target = next((item for item in ordered if item.object_id == target_object_id), None)
    if target is None:
        raise ValueError(f"target_object_id is not an enabled instance: {target_object_id!r}")
    return target, tuple(item for item in ordered if item.object_id != target_object_id)


def _offline_catalog_report(
    catalog_path: Path,
    asset_root: Path,
    candidate_instances: Sequence[Any],
    target_object_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    missing_assets: list[dict[str, str]] = []
    classes_report: dict[str, Any] = {}
    catalog_payload: dict[str, Any] = {}
    source_payload: dict[str, Any] = {}

    if not catalog_path.is_file():
        return {
            "status": "preparation_required",
            "errors": [],
            "missing_assets": [{"path": str(catalog_path), "reason": "catalog is missing"}],
            "classes": {},
        }
    try:
        catalog_payload = _load_json(catalog_path, "object catalog")
    except (OSError, ValueError) as error:
        return {"status": "failed", "errors": [str(error)], "missing_assets": [], "classes": {}}

    if catalog_payload.get("schema_version") != 1:
        errors.append("catalog schema_version must be 1")
    classes = catalog_payload.get("classes")
    if not isinstance(classes, dict):
        errors.append("catalog classes must be an object")
        classes = {}
    if tuple(classes) != EXPECTED_CLASSES:
        errors.append(f"catalog classes must be exactly {EXPECTED_CLASSES!r}")

    source_manifest_path = asset_root / "source_manifest.json"
    if not source_manifest_path.is_file():
        missing_assets.append({"path": str(source_manifest_path), "reason": "source manifest is missing"})
    else:
        try:
            source_payload = _load_json(source_manifest_path, "RialTo source manifest")
        except (OSError, ValueError) as error:
            errors.append(str(error))
    source_assets = source_payload.get("assets", {})
    if not isinstance(source_payload.get("repo"), str) or not source_payload["repo"].startswith(("https://", "http://")):
        errors.append("source manifest repo is invalid")
    revision = source_payload.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("source manifest revision must be a 40-character lowercase Git SHA")
    if not isinstance(source_assets, dict):
        errors.append("source manifest assets must be an object")
        source_assets = {}
    elif set(source_assets) != set(EXPECTED_SOURCE_BY_CLASS.values()):
        errors.append("source manifest assets must contain exactly the six pinned RialTo files")

    for object_class in EXPECTED_CLASSES:
        raw = classes.get(object_class)
        if not isinstance(raw, dict):
            errors.append(f"{object_class}: catalog record must be an object")
            continue
        expected_source_name = EXPECTED_SOURCE_BY_CLASS[object_class]
        source_record = source_assets.get(expected_source_name)
        if not isinstance(source_record, dict):
            errors.append(f"{object_class}: source manifest entry is missing for {expected_source_name}")
            source_record = {}
        if source_record.get("resolved_path") != f"sources/{expected_source_name}":
            errors.append(f"{object_class}: source manifest resolved_path is invalid")
        source_url = source_record.get("source_url")
        if not isinstance(source_url, str) or not source_url.startswith(("https://", "http://")):
            errors.append(f"{object_class}: source manifest source_url is invalid")
        source_sha = source_record.get("source_sha256")
        catalog_source_sha = raw.get("source_sha256")
        source_path = asset_root / "sources" / expected_source_name
        source_present = source_path.is_file()
        measured_source_sha = sha256_file(source_path) if source_present else None
        if not source_present:
            missing_assets.append({"path": str(source_path), "reason": f"{object_class} source is not materialized"})
        if not isinstance(source_sha, str) or SHA256_RE.fullmatch(source_sha) is None:
            errors.append(f"{object_class}: source manifest SHA-256 is invalid")
        if catalog_source_sha != source_sha:
            errors.append(f"{object_class}: catalog source_sha256 does not match source manifest")
        if measured_source_sha is not None and measured_source_sha != source_sha:
            errors.append(f"{object_class}: source SHA-256 mismatch")

        expected_resolved = EXPECTED_RESOLVED_BY_CLASS[object_class]
        raw_path = raw.get("usd_path")
        if raw_path != expected_resolved:
            errors.append(f"{object_class}: usd_path must be {expected_resolved!r}")
        resolved_path = asset_root / expected_resolved
        resolved_present = resolved_path.is_file()
        measured_resolved_sha = sha256_file(resolved_path) if resolved_present else None
        resolved_sha = raw.get("resolved_sha256")
        if not resolved_present:
            missing_assets.append({"path": str(resolved_path), "reason": f"{object_class} resolved USD is not materialized"})
        if not isinstance(resolved_sha, str) or SHA256_RE.fullmatch(resolved_sha) is None:
            if resolved_sha is None:
                missing_assets.append({"path": str(resolved_path), "reason": f"{object_class} resolved SHA-256 is not recorded"})
            else:
                errors.append(f"{object_class}: resolved SHA-256 is invalid")
        sha_matches = resolved_present and measured_resolved_sha == resolved_sha
        if resolved_present and not sha_matches:
            errors.append(f"{object_class}: resolved SHA-256 mismatch")

        collision_profile = raw.get("collision_profile")
        collision_metadata = raw.get("collision_metadata")
        collision_metadata_valid = collision_profile in ALLOWED_COLLISION_PROFILES
        mass_kg = raw.get("mass_kg")
        scale = raw.get("scale")
        if (
            isinstance(mass_kg, bool)
            or not isinstance(mass_kg, (int, float))
            or not math.isfinite(float(mass_kg))
            or float(mass_kg) <= 0.0
        ):
            collision_metadata_valid = False
            errors.append(f"{object_class}: mass_kg collision metadata is invalid")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            collision_metadata_valid = False
            errors.append(f"{object_class}: scale collision metadata is invalid")
        if collision_metadata is not None and not isinstance(collision_metadata, dict):
            collision_metadata_valid = False
        if not collision_metadata_valid:
            errors.append(f"{object_class}: invalid collision metadata/profile")

        dependencies_resolved = False
        dependency_report: dict[str, Any] = {"dependencies_resolved": False, "missing_dependencies": []}
        if resolved_present:
            try:
                from scripts.m1_rialto_object_assets import inspect_usd

                dependency_report = inspect_usd(resolved_path, asset_root=asset_root)
                dependencies_resolved = bool(dependency_report.get("dependencies_resolved"))
                if not dependencies_resolved:
                    errors.append(f"{object_class}: USD dependencies are unresolved")
                if dependency_report.get("outside_root_dependencies"):
                    errors.append(f"{object_class}: USD dependencies are outside asset root")
            except Exception as error:
                errors.append(f"{object_class}: USD inspection failed: {error}")
        classes_report[object_class] = {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "measured_source_sha256": measured_source_sha,
            "source_sha_matches": measured_source_sha == source_sha if measured_source_sha else False,
            "usd_path": str(resolved_path),
            "resolved_sha256": resolved_sha,
            "measured_resolved_sha256": measured_resolved_sha,
            "sha256_matches": sha_matches,
            "dependencies_resolved": dependencies_resolved,
            "missing_dependencies": dependency_report.get("missing_dependencies", []),
            "outside_root_dependencies": dependency_report.get("outside_root_dependencies", []),
            "collision_profile": collision_profile,
            "collision_metadata": collision_metadata,
            "collision_metadata_valid": collision_metadata_valid,
        }

    try:
        target, obstacles = partition_target_and_obstacles(candidate_instances, target_object_id)
        partition = {
            "target_object_id": None if target is None else target.object_id,
            "obstacle_object_ids": [item.object_id for item in obstacles],
            "unique_object_ids": len({item.object_id for item in candidate_instances}) == len(candidate_instances),
        }
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        partition = {"target_object_id": None, "obstacle_object_ids": [], "unique_object_ids": False}

    deduplicated_missing = sorted({(item["path"], item["reason"]): item for item in missing_assets}.values(), key=lambda item: item["path"])
    status = "preparation_required" if deduplicated_missing else ("failed" if errors else "passed")
    return {
        "status": status,
        "errors": sorted(set(errors)),
        "missing_assets": deduplicated_missing,
        "classes": classes_report,
        "partition": partition,
        "catalog_sha256": sha256_file(catalog_path),
        "source_manifest_sha256": sha256_file(source_manifest_path) if source_manifest_path.is_file() else None,
    }


def _gpu0_smoke_report(
    catalog_path: Path,
    asset_root: Path,
    candidate_instances: Sequence[Any],
    target_object_id: str,
    *,
    device: str,
    steps: int,
    simulation_app: Any,
) -> dict[str, Any]:
    """Run the smallest catalog scene smoke after Isaac has been launched."""

    try:
        import gymnasium as gym
        import torch

        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import load_catalog
        from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import M1DualPandaO6BimanualEnvCfg

        catalog = load_catalog(catalog_path, asset_root)
        target, obstacles = partition_target_and_obstacles(candidate_instances, target_object_id)
        cfg = M1DualPandaO6BimanualEnvCfg(
            object_catalog=catalog,
            object_instances=tuple(candidate_instances),
        )
        cfg.scene.num_envs = 1
        env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
        try:
            env.reset(seed=7)
            scene = env.unwrapped.scene
            object_initialized: dict[str, bool] = {}
            missing_object_ids: list[str] = []
            for instance in candidate_instances:
                if not instance.enabled:
                    continue
                try:
                    object_initialized[instance.object_id] = bool(
                        scene[instance.object_id].is_initialized
                    )
                except KeyError:
                    object_initialized[instance.object_id] = False
                    missing_object_ids.append(instance.object_id)
            contact_names: list[str] = []
            missing_contact_names: list[str] = []
            for name in ("o6_contacts", "right_o6_contacts", "box_contacts"):
                try:
                    sensor = scene[name]
                except KeyError:
                    missing_contact_names.append(name)
                    continue
                if bool(sensor.is_initialized):
                    contact_names.append(name)
            action_dim = int(getattr(env.action_space, "shape", (0,))[0])
            for _ in range(max(1, int(steps))):
                env.step(torch.zeros((1, action_dim), device=device))
            contact_ok = (
                bool(object_initialized)
                and all(object_initialized.values())
                and not missing_object_ids
                and not missing_contact_names
                and bool(contact_names)
            )
            return {
                "status": "passed" if contact_ok else "failed",
                "device": device,
                "physics_steps": max(1, int(steps)),
                "target_object_id": None if target is None else target.object_id,
                "obstacle_object_ids": [instance.object_id for instance in obstacles],
                "object_initialized": object_initialized,
                "missing_object_ids": missing_object_ids,
                "contact_sensor_names": contact_names,
                "missing_contact_sensor_names": missing_contact_names,
                "contact_initialization_passed": contact_ok,
            }
        finally:
            env.close()
    except Exception as error:
        return {
            "status": "failed",
            "device": device,
            "contact_initialization_passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def verify_catalog(
    catalog_path: Path = DEFAULT_CATALOG,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    *,
    evidence_dir: Path | None = DEFAULT_EVIDENCE_ROOT,
    candidate_instances: Sequence[Any] | None = None,
    target_object_id: str = "bottle_000",
    run_gpu: bool = True,
    device: str = "cuda:0",
    steps: int = 8,
    simulation_app: Any = None,
) -> dict[str, Any]:
    """Verify local catalog inputs and optionally run the GPU0 scene smoke."""

    from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import ObjectInstance

    catalog_path = Path(catalog_path)
    asset_root = Path(asset_root)
    if candidate_instances is None:
        candidate_instances = tuple(
            ObjectInstance(f"{name}_000", name, (0.65, index * 0.15, 1.20))
            for index, name in enumerate(("bottle", "cup", "cube", "cylinder"))
        )
    offline = _offline_catalog_report(catalog_path, asset_root, candidate_instances, target_object_id)
    if offline["status"] != "passed":
        gpu_smoke = {
            "status": "preparation_required" if offline["status"] == "preparation_required" else "not_run",
            "contact_initialization_passed": False,
            "reason": "offline catalog prerequisites did not pass",
        }
    elif not run_gpu:
        gpu_smoke = {"status": "not_requested", "contact_initialization_passed": False}
    elif simulation_app is None:
        gpu_smoke = {
            "status": "not_run",
            "contact_initialization_passed": False,
            "reason": "GPU0 smoke requires a launched Isaac headless application",
        }
    else:
        gpu_smoke = _gpu0_smoke_report(
            catalog_path,
            asset_root,
            candidate_instances,
            target_object_id,
            device=device,
            steps=steps,
            simulation_app=simulation_app,
        )
    hard_gates_passed = offline["status"] == "passed" and (
        (not run_gpu) or gpu_smoke["status"] == "passed"
    )
    if hard_gates_passed:
        verification_status = "passed"
    elif offline["status"] == "preparation_required":
        verification_status = "preparation_required"
    else:
        verification_status = "failed"
    result = {
        "schema_version": 1,
        "verification_status": verification_status,
        "hard_gates_passed": hard_gates_passed,
        "catalog": str(catalog_path.resolve()),
        "asset_root": str(asset_root.resolve()),
        "preparation_command": _preparation_command(asset_root),
        "offline": offline,
        "gpu_smoke": gpu_smoke,
    }
    if evidence_dir is not None:
        _write_evidence(Path(evidence_dir), result)
    return result


def _write_evidence(evidence_dir: Path, result: dict[str, Any]) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report_path = evidence_dir / "catalog_verification.json"
    _atomic_json(report_path, result)
    files = []
    for path in sorted(evidence_dir.iterdir()):
        if path.is_file() and path.name != "aggregate.manifest.json":
            files.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    catalog_path = Path(str(result["catalog"]))
    source_manifest_path = Path(str(result["asset_root"])) / "source_manifest.json"
    for path, label in ((catalog_path, "catalog.json"), (source_manifest_path, "source_manifest.json")):
        if path.is_file():
            files.append({"path": label, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    files.sort(key=lambda item: item["path"])
    body = {"schema_version": 1, "verification_status": result["verification_status"], "files": files}
    aggregate = {**body, "aggregate_sha256": hashlib.sha256(_canonical_json(body)).hexdigest()}
    _atomic_json(evidence_dir / "aggregate.manifest.json", aggregate)


def _parser(*, launcher: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--target-object-id", default="bottle_000")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    if launcher:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action="store_true", help="launch Isaac headlessly for GPU0 smoke")
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate_instances = tuple(
        __import__("go2_pvcnn.control.m1_bimanual_coordination.object_catalog", fromlist=["ObjectInstance"]).ObjectInstance(
            f"{name}_000", name, (0.65, index * 0.15, 1.20)
        )
        for index, name in enumerate(("bottle", "cup", "cube", "cylinder"))
    )
    preliminary = verify_catalog(
        args.catalog,
        args.asset_root,
        evidence_dir=args.evidence_dir,
        candidate_instances=candidate_instances,
        target_object_id=args.target_object_id,
        run_gpu=False,
        device=args.device,
        steps=args.steps,
    )
    if args.offline_only or preliminary["offline"]["status"] != "passed":
        result = preliminary
    else:
        try:
            from isaaclab.app import AppLauncher

            launcher_parser = _parser(launcher=True)
            app_args = launcher_parser.parse_args()
            app = AppLauncher(app_args).app
            result = verify_catalog(
                args.catalog,
                args.asset_root,
                evidence_dir=args.evidence_dir,
                candidate_instances=candidate_instances,
                target_object_id=args.target_object_id,
                run_gpu=True,
                device=args.device,
                steps=args.steps,
                simulation_app=app,
            )
        except Exception as error:
            result = {**preliminary, "verification_status": "failed", "hard_gates_passed": False,
                      "gpu_smoke": {"status": "failed", "contact_initialization_passed": False,
                                    "error": f"{type(error).__name__}: {error}"}}
            _write_evidence(args.evidence_dir, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
