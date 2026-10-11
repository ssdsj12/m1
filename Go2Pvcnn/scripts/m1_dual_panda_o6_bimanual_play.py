"""GUI playback of the deterministic M1 dual-Panda O6 MPC mission."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from collections.abc import Sequence

from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectInstance,
    load_catalog,
)
from go2_pvcnn.control.m1_bimanual_coordination.wire_coupling import (
    WireCouplingDiagnostics,
)
from go2_pvcnn.control.m1_bimanual_coordination.wire_contact_acceptance import (
    WireAcceptanceState,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_CATALOG = ROOT / "config" / "m1_object_catalog.json"
DEFAULT_OBJECT_ASSET_ROOT = ROOT / "assets" / "m1_objects" / "rialto"

_DEFAULT_OBJECT_POSES: dict[str, tuple[float, float, float]] = {
    "bottle": (0.65, 0.00, 1.20),
    "cup": (0.65, 0.28, 1.20),
    "bowl": (0.65, -0.28, 1.20),
    "book": (0.88, 0.00, 1.20),
    "cube": (0.88, 0.28, 1.20),
    "cylinder": (0.88, -0.28, 1.20),
    "wire": (0.65, 0.0, 1.20),
}
_DEFAULT_OBJECT_IDS = frozenset(
    f"{object_class}_000" for object_class in _DEFAULT_OBJECT_POSES
)


class _WireReportRecorder:
    """Retain state transitions and the latest immutable diagnostic snapshot."""

    def __init__(self) -> None:
        self._transitions: list[dict[str, object]] = []
        self._final: dict[str, object] | None = None
        self._last_state: str | None = None

    def record(self, diagnostics: WireCouplingDiagnostics) -> None:
        if not isinstance(diagnostics, WireCouplingDiagnostics):
            raise TypeError("diagnostics must be WireCouplingDiagnostics")
        if self._last_state is None and diagnostics.state != "SCENE_TRACKED":
            initial = replace(
                diagnostics,
                state="SCENE_TRACKED",
                transition_reason="runtime_initialized",
                left_contact_ticks=0,
                right_contact_ticks=0,
                bind_count=0,
                deformx_elapsed_time=0.0,
                wire_goal_committed=False,
            )
            self._transitions.append(initial.to_json())
            self._last_state = initial.state
        payload = diagnostics.to_json()
        if diagnostics.state != self._last_state:
            self._transitions.append(payload)
            self._last_state = diagnostics.state
        self._final = payload

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "transitions": list(self._transitions),
            "final": self._final,
        }


class _WireContactReportRecorder:
    """Snapshot selected-wire diagnostics without deriving contact from geometry."""

    def __init__(self, *, object_id: str, seed: int) -> None:
        self.records: list[dict[str, object]] = []
        self._latest: dict[str, object] = {
            "schema_version": 1, "object_id": object_id, "seed": seed,
            "step": 0, "elapsed_simulation_time_s": 0.0,
            "wire_center_height_m": None, "reset_wire_center_height_m": None,
            "lift_height_m": None, "hold_elapsed_s": 0.0,
            "bilateral_contact_elapsed_s": 0.0,
            "left_valid_contacts": 0, "right_valid_contacts": 0,
            "success": False, "terminal_reason": None,
        }
        self._maxima = {f"max_{key}": 0.0 for key in (
            "left_valid_contacts", "right_valid_contacts", "lift_height_m",
            "hold_elapsed_s", "bilateral_contact_elapsed_s",
        )}
        self.finished = False

    def reset(self, *, wire_center_height_m: float, control_dt_s: float) -> None:
        self._latest["reset_wire_center_height_m"] = wire_center_height_m
        self._latest["control_interval_s"] = control_dt_s

    def record(
        self, *, step: int, diagnostics: dict[str, object],
        acceptance_state: WireAcceptanceState,
        wire_coupling: WireCouplingDiagnostics | None = None,
    ) -> None:
        if self.finished:
            raise RuntimeError("wire contact report is already finalized")
        state = asdict(acceptance_state)
        state["acceptance_phase"] = state.pop("phase").value
        row = {**self._latest, **deepcopy(diagnostics), **state}
        row.update(
            record_type="sample", step=step,
            elapsed_simulation_time_s=step * row["control_interval_s"],
            lift_height_m=(diagnostics["wire_center_height_m"]
                           - row["reset_wire_center_height_m"]),
            wire_coupling=None if wire_coupling is None else wire_coupling.to_json(),
        )
        nonfinite_fields: list[str] = []

        def finite_json(value, path=""):
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite_fields.append(path)
                return None
            if isinstance(value, dict):
                return {key: finite_json(item, f"{path}.{key}" if path else key)
                        for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(item, f"{path}[{index}]")
                        for index, item in enumerate(value)]
            return value

        row = finite_json(row)
        row["nonfinite_fields"] = nonfinite_fields
        for key in self._maxima:
            value = row[key.removeprefix("max_")]
            if value is not None:
                self._maxima[key] = max(self._maxima[key], value)
        self.records.append(row)
        self._latest = row

    def finish(self, terminal_reason: str) -> dict[str, object]:
        if not self.finished:
            final = deepcopy(self._latest)
            final.update(self._maxima)
            final.update(record_type="final", terminal_reason=terminal_reason,
                         success=bool(final["success"] and terminal_reason == "accepted"))
            self.records.append(final)
            self.finished = True
        return self.records[-1]


def _write_report_text_atomic(path: Path, encoded: str) -> None:
    """Flush a sibling temporary file before replacing the complete report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_wire_report_atomic(path: Path, report: dict[str, object]) -> None:
    """Keep the legacy staged-coupling JSON report format available."""
    _write_report_text_atomic(
        path, json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _write_wire_contact_report_atomic(path: Path, records: Sequence[dict[str, object]]) -> None:
    _write_report_text_atomic(path, "".join(
        json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in records
    ))


def _positive_steps(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("max-steps must be positive")
    return result


def _metadata_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "value must be 64 lowercase hexadecimal characters"
        )
    return value


def _validate_prior_pair(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        args.fingertip_prior_artifact is not None
        and args.fingertip_prior_metadata_sha256 is None
    ):
        parser.error(
            "--fingertip-prior-artifact requires "
            "--fingertip-prior-metadata-sha256"
        )
    if (
        args.fingertip_prior_artifact is None
        and args.fingertip_prior_metadata_sha256 is not None
    ):
        parser.error(
            "--fingertip-prior-metadata-sha256 requires "
            "--fingertip-prior-artifact"
        )


def _parser(*, wire_contact_probe: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(wire_contact_probe=wire_contact_probe)
    parser.add_argument("--mpc-backend", choices=("reference-cpu", "bimanual-rti-cuda", "auto"), default="reference-cpu")
    parser.add_argument("--qp-backend", choices=("reference-cpu", "osqp-cuda", "auto"), default="reference-cpu")
    parser.add_argument("--control-device", default="cuda:0", help="requested control device; reference-cpu always uses CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=_positive_steps, default=4000)
    parser.add_argument("--object-id", default=None, required=wire_contact_probe)
    parser.add_argument(
        "--object-catalog",
        type=Path,
        default=None,
        help=(
            "optional object catalog JSON; omitted keeps the legacy /Box scene, "
            "while --object-id uses the default catalog"
        ),
    )
    parser.add_argument(
        "--object-assets-root",
        type=Path,
        default=DEFAULT_OBJECT_ASSET_ROOT,
        help="local prepared object asset root used with --object-catalog",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--wire-report",
        type=Path,
        default=None,
        required=wire_contact_probe,
        help="atomic wire report: sampled JSONL for the probe or .jsonl paths; legacy coupling JSON otherwise",
    )
    parser.add_argument("--mode", choices=("teacher", "latent"), default="teacher")
    parser.add_argument("--fingertip-prior-artifact", type=Path, default=None)
    parser.add_argument(
        "--fingertip-prior-metadata-sha256", type=_metadata_sha256, default=None
    )
    return parser


def _catalog_class_names(path: Path) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"object catalog does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read object catalog: {path}") from error
    classes = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(classes, dict) or not classes:
        raise ValueError("object catalog classes must be a non-empty object")
    for object_class in classes:
        if not isinstance(object_class, str) or not object_class:
            raise ValueError("object catalog class names must be non-empty strings")
    return tuple(classes)


def _fallback_pose(index: int) -> tuple[float, float, float]:
    x = 0.65 + 0.23 * (index // 3)
    y = (0.28, -0.28, 0.56)[index % 3]
    return (x, y, 1.20)


def default_object_instances(
    object_classes: Sequence[str],
) -> tuple[ObjectInstance, ...]:
    """Build one deterministic enabled instance per catalog class."""

    if isinstance(object_classes, (str, bytes)):
        raise TypeError("object_classes must be a sequence of class names")
    instances: list[ObjectInstance] = []
    seen: set[str] = set()
    for index, object_class in enumerate(object_classes):
        if not isinstance(object_class, str) or not object_class:
            raise ValueError("object class names must be non-empty strings")
        if object_class in seen:
            raise ValueError(f"duplicate object class: {object_class!r}")
        seen.add(object_class)
        instances.append(
            ObjectInstance(
                f"{object_class}_000",
                object_class,
                _DEFAULT_OBJECT_POSES.get(object_class, _fallback_pose(index)),
            )
        )
    return tuple(instances)


def _validate_target_object_id(
    object_id: str | None, instances: Sequence[ObjectInstance]
) -> None:
    if object_id is None or object_id == "box":
        return
    if not isinstance(object_id, str) or not object_id:
        raise TypeError("target object_id must be a non-empty string or None")
    valid_ids = {instance.object_id for instance in instances if instance.enabled}
    if object_id not in valid_ids:
        available = ", ".join(sorted(valid_ids))
        raise ValueError(
            f"unknown target object_id: {object_id!r}; available catalog object IDs: "
            f"{available}"
        )


def _uses_catalog_scene(object_id: str | None, object_catalog: Path | None) -> bool:
    return object_catalog is not None or (object_id not in {None, "box"})


def _prepare_object_scene(
    *,
    object_id: str | None,
    object_catalog: Path | None,
    object_assets_root: Path,
) -> tuple[ObjectCatalog | None, tuple[ObjectInstance, ...]]:
    if not _uses_catalog_scene(object_id, object_catalog):
        return None, ()
    catalog_path = Path(object_catalog) if object_catalog is not None else DEFAULT_OBJECT_CATALOG
    all_instances = default_object_instances(_catalog_class_names(catalog_path))
    _validate_target_object_id(object_id, all_instances)
    if object_id is None or object_id == "box":
        instances = all_instances
    else:
        instances = tuple(
            instance for instance in all_instances if instance.object_id == object_id
        )
    required_classes = tuple(
        instance.object_class for instance in instances if instance.enabled
    )
    catalog = load_catalog(
        catalog_path,
        Path(object_assets_root),
        required_classes=required_classes,
    )
    return catalog, instances


def _load_env_cfg_type():
    from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_env_cfg import (
        M1DualPandaO6BimanualEnvCfg,
    )

    return M1DualPandaO6BimanualEnvCfg


def build_play_config(
    *,
    seed: int,
    object_id: str | None,
    object_catalog: Path | None = None,
    object_assets_root: Path = DEFAULT_OBJECT_ASSET_ROOT,
    catalog: ObjectCatalog | None = None,
    object_instances: tuple[ObjectInstance, ...] | None = None,
):
    """Construct the env cfg used by shipped play before ``gym.make``."""

    if catalog is None and object_instances is None and not _uses_catalog_scene(
        object_id, object_catalog
    ):
        object_instances = ()
    elif (
        catalog is None
        and object_instances is None
        and object_catalog is None
        and object_id not in _DEFAULT_OBJECT_IDS
    ):
        available = ", ".join(sorted(_DEFAULT_OBJECT_IDS))
        raise ValueError(
            f"unknown target object_id: {object_id!r}; available catalog object IDs: "
            f"{available}"
        )
    elif catalog is None and object_instances is None:
        catalog, object_instances = _prepare_object_scene(
            object_id=object_id,
            object_catalog=object_catalog,
            object_assets_root=object_assets_root,
        )
    elif (
        catalog is None
        and object_instances == ()
        and object_catalog is None
        and object_id in {None, "box"}
    ):
        pass
    elif catalog is None or object_instances is None:
        raise ValueError("catalog and object_instances must be supplied together")
    _validate_target_object_id(object_id, object_instances)

    cfg_type = _load_env_cfg_type()
    if catalog is None:
        cfg = cfg_type(target_object_id=object_id)
    else:
        cfg = cfg_type(
            target_object_id=object_id,
            object_catalog=catalog,
            object_instances=tuple(object_instances),
        )
    cfg.scene.num_envs = 1
    cfg.seed = seed
    return cfg


def main(parser: argparse.ArgumentParser | None = None) -> int:
    parser = _parser() if parser is None else parser
    early_args, _unknown_launcher_args = parser.parse_known_args()
    _validate_prior_pair(parser, early_args)
    from go2_pvcnn.control.m1_bimanual_coordination.gpu_rti.config import resolve_backend_selection

    try:
        backend_selection = resolve_backend_selection(early_args.mpc_backend, early_args.qp_backend, early_args.control_device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    parser.set_defaults(backend_selection=backend_selection)
    binding = None
    try:
        if early_args.fingertip_prior_artifact is not None:
            from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime_binding import (
                PreparedO6FingertipPriors,
            )

            binding = PreparedO6FingertipPriors.from_artifact(
                early_args.fingertip_prior_artifact,
                expected_metadata_sha256=early_args.fingertip_prior_metadata_sha256,
            )
        return _run_with_fingertip_prior(parser, binding)
    finally:
        if binding is not None:
            binding.close()


class _ReducedWirePlayMpc:
    """Attach reduced forecasts at the existing wire object-planning boundary."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.upstream = runtime.object_mpc
        self.reduced = None
        self.attachment = None
        self.requested_lift = None
        self.reset_status()

    def __getattr__(self, name):
        return getattr(self.upstream, name)

    def reset_status(self):
        self.runtime.latest_reduced_wire_update = None
        self.runtime.latest_reduced_wire_status = {
            "active": False, "reason": "awaiting_bilateral_latch",
            "prediction_feasible": None,
        }

    def reset(self):
        self.upstream.reset()
        self.reduced = None
        self.attachment = None
        self.requested_lift = None
        self.reset_status()

    def boundary_frames(self, snapshot):
        from go2_pvcnn.control.m1_bimanual_coordination.dlo_frame_provider import dual_palm_attached_grasp_frames
        frames = dual_palm_attached_grasp_frames(snapshot, self.attachment)
        if self.requested_lift is not None:
            for side in ("left", "right"):
                # Boundary twist is a requested control, while pose remains
                # measured. Keep rigid-offset angular velocity in pinch space.
                arm = getattr(snapshot, f"{side}_arm")
                frames[f"{side}_frame"]["velocity"][2] += self.requested_lift - float(arm.palm_twist_b[2])
        return frames

    def plan(self, sample):
        import numpy as np
        from go2_pvcnn.control.m1_bimanual_coordination.dlo_frame_provider import (
            capture_palm_grasp_attachment, dual_palm_attached_grasp_frames,
        )
        from go2_pvcnn.control.m1_bimanual_coordination.reduced_wire_runtime import ReducedWireRuntime
        from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualPhase

        coupling = self.runtime.wire_coupling_state
        if self.reduced is None and not (coupling.left_latched and coupling.right_latched):
            self.reset_status()
            return self.upstream.plan(sample)
        snapshot = sample.snapshot
        self.requested_lift = None
        if sample.phase is BimanualPhase.LIFT:
            height_error = float(sample.target_box_pose_b[-1, 2] - snapshot.box.pose_b[2])
            self.requested_lift = min(self.runtime.dlo_trajectory_planner.palm_speed_m_s,
                                      max(0., height_error / self.upstream.cfg.horizon_seconds))
        if self.reduced is None:
            geometry = self.runtime.wire_geometry_provider(snapshot)
            self.attachment = capture_palm_grasp_attachment(snapshot)
            self.reduced = ReducedWireRuntime(
                geometry.centerline,
                frame_provider=self.boundary_frames,
                material={"mass_kg": float(snapshot.box.mass),
                          "friction_coefficient": self.upstream.cfg.friction_coefficient},
            )
        # Forces are selected-target sensor measurements. Do not also consume
        # DeformX impulses here: the staged runtime owns that backend's stepping.
        wrenches = []
        frames = dual_palm_attached_grasp_frames(snapshot, self.attachment)
        goal = self.runtime.latest_wire_goal
        for side in ("left", "right"):
            hand = getattr(snapshot, f"{side}_hand")
            forces = hand.fingertip_forces_b[hand.contact_mask].numpy()
            positions = hand.fingertip_positions_b[hand.contact_mask].numpy()
            force = forces.sum(axis=0)
            moment = np.cross(positions - frames[f"{side}_frame"]["position"], forces).sum(axis=0)
            normal = (getattr(goal, f"{side}_palm_approach_normal") if goal is not None
                      else (0., -1. if side == "left" else 1., 0.))
            wrenches.append({"wrench": np.concatenate((force, moment)), "normal": normal})
        # The forecast grid must match the 25-node object MPC (40 ms).
        update = self.reduced.update(snapshot, wrenches, self.upstream.cfg.dt)
        self.runtime.latest_reduced_wire_update = update
        self.runtime.latest_reduced_wire_status = {
            "active": True, "reason": update.failure_reason,
            "prediction_feasible": update.prediction is not None,
            "requested_lift_m_s": self.requested_lift,
            "stretch": update.state.axial_stretch,
            "bending": update.state.bending_proxy,
            "left_normal_force_n": update.state.left_normal_force_n,
            "right_normal_force_n": update.state.right_normal_force_n,
        }
        if update.failure_reason is not None:
            return self.upstream._fallback(sample, f"reduced_wire:{update.failure_reason}")
        # Task 3's optimizer uses palm positions; retain physical geometry and
        # material diagnostics while translating endpoint references to palms.
        prediction = update.prediction
        positions = prediction.endpoint_positions.copy()
        velocities = prediction.endpoint_velocities.copy()
        for index, side in enumerate(("left", "right")):
            arm = getattr(snapshot, f"{side}_arm")
            frame = frames[f"{side}_frame"]
            positions[:, index] += arm.palm_pose_b[:3].numpy() - frame["position"]
            velocities[:, index] += arm.palm_twist_b[:3].numpy() - frame["velocity"]
        palm_prediction = replace(prediction, endpoint_positions=positions,
                                  endpoint_velocities=velocities,
                                  endpoint_separation_m=np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1))
        return self.upstream.plan(replace(sample, wire_prediction=palm_prediction,
                                          hold_measured_palm_orientation=True))


class _ReducedWirePlayTrajectory:
    def __init__(self, runtime, upstream):
        self.runtime, self.upstream = runtime, upstream

    def __getattr__(self, name):
        return getattr(self.upstream, name)

    def plan_state(self, snapshot, solution, state):
        update = self.runtime.latest_reduced_wire_update
        if update is None:
            return self.upstream.plan_state(snapshot, solution, state)
        plan = self.upstream.plan_prediction(snapshot, solution, update)
        # Keep the QP's position/velocity/slip-constrained output authoritative.
        return replace(plan, object_solution=solution)


def _configure_reduced_wire_runtime(wrapper):
    runtime = wrapper.runtime
    if getattr(runtime, "wire_geometry_provider", None) is None:
        return
    if isinstance(runtime.object_mpc, _ReducedWirePlayMpc):
        return
    runtime.object_mpc = _ReducedWirePlayMpc(runtime)
    runtime.dlo_trajectory_planner = _ReducedWirePlayTrajectory(runtime, runtime.dlo_trajectory_planner)


def _play_episode(wrapper, simulation_app, args, wire_report, contact_report) -> str:
    reset_snapshot = wrapper.reset(seed=args.seed)
    _configure_reduced_wire_runtime(wrapper)
    if contact_report is not None:
        contact_report.reset(
            wire_center_height_m=float(reset_snapshot.box.pose_b[2]),
            control_dt_s=float(wrapper.env.unwrapped.physics_dt),
        )
    previous_phase = wrapper.runtime.mission.phase.name
    for step in range(args.max_steps):
        if not simulation_app.is_running():
            return "application_closed"
        wrapper.step()
        wire_diagnostics = wrapper.runtime.latest_wire_diagnostics
        if wire_diagnostics is not None:
            wire_report.record(wire_diagnostics)
        state = wrapper.wire_acceptance_state
        if contact_report is not None:
            diagnostics = wrapper.wire_acceptance_diagnostics
            if state is None or diagnostics is None:
                raise RuntimeError("selected wire acceptance diagnostics are unavailable")
            diagnostics["reduced_wire"] = getattr(wrapper.runtime, "latest_reduced_wire_status", None)
            contact_report.record(step=step + 1, diagnostics=diagnostics,
                                  acceptance_state=state, wire_coupling=wire_diagnostics)
        phase = wrapper.runtime.mission.phase.name
        if args.diagnostics and (phase != previous_phase or phase in {"DONE", "TERMINATED"}):
            latest = wrapper.runtime.latest_solutions
            reasons = {
                key: None if value is None else value.diagnostics.fallback_reason
                for key, value in latest.items() if key != "arm"
            }
            print(
                f"step={step + 1} phase={phase} feasible={wrapper.last_command.feasible} "
                f"reasons={reasons} base_state={wrapper.last_snapshot.base_state.tolist()}",
                flush=True,
            )
        previous_phase = phase
        if state is not None:
            if state.terminal_reason is not None:
                return state.terminal_reason
            if state.success:
                return "accepted"
        if phase in {"DONE", "TERMINATED"}:
            return f"mission_{phase.lower()}"
    return "max_steps"


def _keep_play_open(wrapper, simulation_app, *, headless: bool) -> None:
    if headless or not simulation_app.is_running():
        return
    wrapper.env.unwrapped.sim.pause()
    while simulation_app.is_running():
        simulation_app.update()


def _publish_wire_contact_report(args, recorder, terminal_reason: str) -> None:
    final = recorder.finish(terminal_reason)
    _write_wire_contact_report_atomic(args.wire_report, recorder.records)
    print(json.dumps(final, allow_nan=False, sort_keys=True), flush=True)


def _run_with_fingertip_prior(parser, binding) -> int:
    # The pinned left/right workers are already bound. Never reload their
    # mutable artifact path after crossing the launcher/scene boundary.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    contact_trace_requested = args.wire_contact_probe or (
        args.wire_report is not None and args.wire_report.suffix.lower() == ".jsonl"
    )
    contact_report = (_WireContactReportRecorder(object_id=args.object_id, seed=args.seed)
                      if contact_trace_requested else None)
    try:
        catalog, object_instances = _prepare_object_scene(
            object_id=args.object_id,
            object_catalog=args.object_catalog,
            object_assets_root=args.object_assets_root,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        if contact_report is not None:
            _publish_wire_contact_report(args, contact_report, f"preflight_error:{type(error).__name__}")
        parser.error(str(error))
    if contact_trace_requested and not any(
        instance.object_id == args.object_id and instance.object_class == "wire"
        for instance in object_instances
    ):
        parser.error("wire contact JSONL requires an explicit selected wire object ID")
    if args.diagnostics:
        print(f"backend_selection={args.backend_selection}", flush=True)
    wire_report = _WireReportRecorder()
    try:
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        import gymnasium as gym

        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import (
            M1DualPandaO6BimanualWrapper,
        )

        cfg = build_play_config(
            seed=args.seed,
            object_id=args.object_id,
            catalog=catalog,
            object_instances=object_instances,
        )
        env = gym.make("Isaac-M1-DualPanda-O6-Bimanual-Lift-v0", cfg=cfg)
        with M1DualPandaO6BimanualWrapper(
            env,
            mode=args.mode,
            object_id=args.object_id,
            fingertip_prior_artifact=args.fingertip_prior_artifact,
            fingertip_prior_metadata_sha256=args.fingertip_prior_metadata_sha256,
            fingertip_prior_binding=binding,
        ) as wrapper:
            if args.object_id is not None:
                wrapper.runtime.wire_object_id = args.object_id
                wrapper.runtime.wire_target_prim_path = cfg.target_contact_prim_path.replace(
                    "{ENV_REGEX_NS}", "/World/envs/env_0"
                )
            terminal_reason = _play_episode(wrapper, simulation_app, args, wire_report, contact_report)
            print(f"final_phase={wrapper.runtime.mission.phase.name}", flush=True)
            if contact_report is not None:
                _publish_wire_contact_report(args, contact_report, terminal_reason)
            elif args.wire_report is not None:
                _write_wire_report_atomic(args.wire_report, wire_report.to_json())
            _keep_play_open(wrapper, simulation_app, headless=args.headless)
    except BaseException as error:
        if contact_report is not None and not contact_report.finished:
            reason = "interrupted" if isinstance(error, KeyboardInterrupt) else f"runtime_error:{type(error).__name__}"
            _publish_wire_contact_report(args, contact_report, reason)
        raise
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
