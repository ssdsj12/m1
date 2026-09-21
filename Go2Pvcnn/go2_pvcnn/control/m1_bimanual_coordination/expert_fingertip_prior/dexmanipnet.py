"""Strict, offline-only loading for pinned DexManipNet successful rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Final

import h5py
import numpy as np

from .geometry_overlay import VerifiedGeometryResolver
from .sources import SOURCE_HANDS


_SOURCE_SIDES: Final = {
    "favor": frozenset({"rh"}),
    "oakinkv2": frozenset({"rh", "lh"}),
}
_SIDES: Final = frozenset({"rh", "lh"})
INTERACTION_MODE_SIDES = MappingProxyType({
    "lh_main": frozenset({"lh"}),
    "rh_main": frozenset({"rh"}),
    "bh_main": frozenset({"rh", "lh"}),
})
_ROOT_STATE_DIM: Final = 13
_TIP_FORCE_DIM: Final = 15


@dataclass(frozen=True)
class SequenceAudit:
    """One atomic acceptance or rejection for one source sequence and hand."""

    source: str
    sequence: str
    side: str
    hand: str | None
    frames: int
    accepted: bool
    reason: str
    input_sha256: str


@dataclass(frozen=True)
class LoadedHandSequence:
    """Geometry-only rollout data plus the provenance needed for later shards."""

    source: str
    sequence: str
    side: str
    source_hand_key: str
    rollout_name: str
    total_reward: float
    q: np.ndarray
    dq: np.ndarray
    root_state: np.ndarray
    object_state: np.ndarray
    tip_force: np.ndarray | None
    object_geometry_path: Path | None
    source_sha256: str

    def __post_init__(self) -> None:
        if self.source not in _SOURCE_SIDES:
            raise ValueError("unsupported source")
        if self.side not in _SIDES or self.side not in _SOURCE_SIDES[self.source]:
            raise ValueError("unsupported source-side pair")
        if self.source_hand_key not in SOURCE_HANDS:
            raise ValueError("unsupported source hand")
        if SOURCE_HANDS[self.source_hand_key].side != self.side:
            raise ValueError("source hand side mismatch")
        if not self.sequence or not self.rollout_name:
            raise ValueError("sequence and rollout provenance must be non-empty")
        if not math.isfinite(self.total_reward):
            raise ValueError("total reward must be finite")
        if (
            len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        if self.object_geometry_path is not None and (not isinstance(self.object_geometry_path, Path) or not self.object_geometry_path.is_file()):
            raise ValueError("object geometry path must be an existing file")

        joint_dim = len(SOURCE_HANDS[self.source_hand_key].joint_order)
        expected = {
            "q": (self.q, (None, joint_dim)),
            "dq": (self.dq, (None, joint_dim)),
            "root_state": (self.root_state, (None, _ROOT_STATE_DIM)),
            "object_state": (self.object_state, (None, _ROOT_STATE_DIM)),
        }
        frames = self.q.shape[0] if isinstance(self.q, np.ndarray) and self.q.ndim else -1
        for name, (array, shape) in expected.items():
            _validate_loaded_array(name, array, shape, frames)
            array.setflags(write=False)
        if self.tip_force is not None:
            _validate_loaded_array("tip_force", self.tip_force, (None, _TIP_FORCE_DIM), frames)
            self.tip_force.setflags(write=False)


@dataclass
class _AuditContext:
    hand: str | None = None
    frames: int = 0
    input_sha256: str = ""


class _SequenceRejected(ValueError):
    pass


def _reject(reason: str) -> None:
    raise _SequenceRejected(reason)


def _validate_geometry_resolver(
    geometry_resolver: VerifiedGeometryResolver | None,
) -> None:
    if geometry_resolver is not None and not isinstance(
        geometry_resolver, VerifiedGeometryResolver
    ):
        raise ValueError("geometry_resolver must be a VerifiedGeometryResolver")


def _validate_loaded_array(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int | None, ...],
    frames: int,
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if array.ndim != len(expected_shape):
        raise ValueError(f"{name} shape mismatch")
    for actual, expected in zip(array.shape, expected_shape, strict=True):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} shape mismatch")
    if array.shape[0] != frames:
        raise ValueError(f"{name} length mismatch")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real numeric array")


def _sequence_name(path: object) -> str:
    try:
        return Path(path).name
    except (TypeError, ValueError):
        return ""


def _validate_source_side(source: object, side: object) -> tuple[str, str]:
    if type(source) is not str or source not in _SOURCE_SIDES:
        _reject("unsupported_source")
    if type(side) is not str or side not in _SIDES:
        _reject("unsupported_side")
    if side not in _SOURCE_SIDES[source]:
        _reject("unsupported_source_side")
    return source, side


def _load_seq_info(sequence_path: Path) -> dict:
    info_path = sequence_path / "seq_info.json"
    if not info_path.is_file():
        _reject("missing_seq_info")
    try:
        value = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _reject("invalid_seq_info")
    if type(value) is not dict:
        _reject("invalid_seq_info")
    return value


def _source_root(sequence_path: Path) -> Path:
    if sequence_path.parent.name == "sequences":
        return sequence_path.parent.parent.resolve()
    return sequence_path.parent.resolve()


def _contained_regular_file(
    root: Path, relative_path: PurePosixPath
) -> tuple[Path | None, bool]:
    """Return a local regular file and whether a failed lookup was simply missing."""

    candidate = root
    final_index = len(relative_path.parts) - 1
    for index, part in enumerate(relative_path.parts):
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            return None, True
        except OSError:
            return None, False
        if stat.S_ISLNK(mode):
            return None, False
        if index < final_index and not stat.S_ISDIR(mode):
            return None, False
    if final_index < 0 or not stat.S_ISREG(mode):
        return None, False
    return candidate, False


def _normalized_overlay_reference(raw: str) -> PurePosixPath:
    posix_path = PurePosixPath(raw)
    if (
        "\\" in raw
        or "\x00" in raw
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw.split("/"))
        or posix_path.suffix.lower() != ".urdf"
    ):
        _reject("invalid_object_geometry")
    return posix_path


def _resolve_object_geometry(
    sequence_path: Path,
    seq_info: dict,
    side: str,
    source: str,
    geometry_resolver: VerifiedGeometryResolver | None,
) -> Path:
    raw = seq_info.get(f"obj_{side}_path")
    if type(raw) is not str:
        _reject("missing_object_geometry")
    if source == "oakinkv2" and geometry_resolver is not None:
        posix_path = _normalized_overlay_reference(raw)
    elif not raw:
        _reject("missing_object_geometry")
    else:
        posix_path = PurePosixPath(raw)
    if (
        posix_path.is_absolute()
        or ".." in posix_path.parts
        or posix_path.suffix.lower() != ".urdf"
    ):
        _reject("invalid_object_geometry")
    root = _source_root(sequence_path)
    candidate, missing = _contained_regular_file(root, posix_path)
    if candidate is None:
        if not missing:
            _reject("invalid_object_geometry")
        if source == "oakinkv2" and geometry_resolver is not None:
            try:
                return geometry_resolver.resolve(raw, source=source)
            except (OSError, TypeError, ValueError):
                _reject("invalid_object_geometry")
        _reject("missing_object_geometry")
    return candidate


def _hash_files(paths: tuple[Path, ...]) -> str:
    digest = sha256()
    for path in paths:
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _best_effort_input_hash(path: object, side: object) -> str:
    """Fingerprint available source bytes without turning a rejection into an error."""

    try:
        sequence_path = Path(path).resolve()
        if not sequence_path.is_dir():
            return ""
        info_path = sequence_path / "seq_info.json"
        rollout_path = sequence_path / "rollouts.hdf5"
        paths = [candidate for candidate in (info_path, rollout_path) if candidate.is_file()]
        if info_path.is_file() and type(side) is str:
            seq_info = json.loads(info_path.read_text(encoding="utf-8"))
            raw = seq_info.get(f"obj_{side}_path") if type(seq_info) is dict else None
            if type(raw) is str:
                posix_path = PurePosixPath(raw)
                if not posix_path.is_absolute() and ".." not in posix_path.parts:
                    geometry, _ = _contained_regular_file(_source_root(sequence_path), posix_path)
                    if geometry is not None:
                        paths.append(geometry)
        return _hash_files(tuple(paths)) if paths else ""
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return ""


def _hard_member(group: h5py.Group, name: str, expected_type, missing_reason: str):
    link = group.get(name, getlink=True)
    if link is None:
        _reject(missing_reason)
    if isinstance(link, h5py.ExternalLink):
        _reject("external_hdf5_link")
    if not isinstance(link, h5py.HardLink):
        _reject("nonlocal_hdf5_link")
    member = group.get(name)
    if not isinstance(member, expected_type):
        _reject(missing_reason)
    return member


def _reject_nonlocal_hdf5_links(group: h5py.Group, seen: set[int] | None = None) -> None:
    """Walk link metadata without dereferencing soft or external targets."""

    if seen is None:
        seen = set()
    address = int(h5py.h5o.get_info(group.id).addr)
    if address in seen:
        return
    seen.add(address)
    for name in group.keys():
        link = group.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            _reject("external_hdf5_link")
        if not isinstance(link, h5py.HardLink):
            _reject("nonlocal_hdf5_link")
        member = group.get(name)
        if isinstance(member, h5py.Group):
            _reject_nonlocal_hdf5_links(member, seen)


def _dataset_array(group: h5py.Group, name: str) -> np.ndarray:
    dataset = _hard_member(group, name, h5py.Dataset, "missing_rollout_field")
    try:
        if dataset.is_virtual or dataset.external:
            _reject("nonlocal_hdf5_storage")
        array = np.asarray(dataset)
    except _SequenceRejected:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        _reject("invalid_rollout_dataset")
    if array.dtype.kind not in "iuf":
        _reject("invalid_rollout_dataset")
    return array


def _validate_frame_length(array: np.ndarray, frames: int) -> None:
    if array.ndim == 0:
        _reject("invalid_rollout_dataset")
    if array.shape[0] != frames:
        _reject("length_mismatch")


def _finite(array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        _reject("non_finite")


def _validate_rollout(
    rollout: h5py.Group,
    *,
    side: str,
    frames: int,
    joint_dim: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    reward = _dataset_array(rollout, "reward")
    q = _dataset_array(rollout, f"q_{side}")
    dq = _dataset_array(rollout, f"dq_{side}")
    root_state = _dataset_array(rollout, f"state_{side}")
    object_state = _dataset_array(rollout, f"state_manip_obj_{side}")
    tip_name = f"tip_force_{side}"
    tip_force = _dataset_array(rollout, tip_name) if tip_name in rollout else None

    for array in (reward, q, dq, root_state, object_state):
        _validate_frame_length(array, frames)
    if tip_force is not None:
        _validate_frame_length(tip_force, frames)

    if reward.ndim not in (1, 2) or (reward.ndim == 2 and reward.shape[1] != 1):
        _reject("reward_shape")
    if q.ndim != 2 or dq.ndim != 2:
        _reject("joint_state_shape")
    if q.shape[1] != joint_dim or dq.shape[1] != joint_dim:
        _reject("joint_dimension_mismatch")
    if root_state.ndim != 2 or root_state.shape[1] != _ROOT_STATE_DIM:
        _reject("root_state_shape")
    if object_state.ndim != 2 or object_state.shape[1] != _ROOT_STATE_DIM:
        _reject("object_state_shape")
    if tip_force is not None and (tip_force.ndim != 2 or tip_force.shape[1] != _TIP_FORCE_DIM):
        _reject("tip_force_shape")

    for array in (reward, q, dq, root_state, object_state):
        _finite(array)
    if tip_force is not None:
        _finite(tip_force)
    score = float(np.sum(reward, dtype=np.float64))
    if not math.isfinite(score):
        _reject("non_finite")
    return score, q, dq, root_state, object_state, tip_force


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _load_best(
    path: str | Path,
    source: str,
    side: str,
    context: _AuditContext,
    geometry_resolver: VerifiedGeometryResolver | None,
    require_geometry: bool = True,
) -> LoadedHandSequence:
    source, side = _validate_source_side(source, side)
    try:
        sequence_path = Path(path).resolve()
    except (OSError, TypeError, ValueError):
        _reject("invalid_sequence_path")
    if not sequence_path.is_dir():
        _reject("invalid_sequence_path")

    seq_info = _load_seq_info(sequence_path)
    frames = seq_info.get("seq_len")
    if type(frames) is not int or frames <= 0:
        _reject("invalid_seq_info")
    context.frames = frames
    interaction_mode = seq_info.get("interaction_mode")
    if type(interaction_mode) is not str or interaction_mode not in INTERACTION_MODE_SIDES:
        _reject("invalid_interaction_mode")
    if side not in INTERACTION_MODE_SIDES[interaction_mode]:
        _reject("interaction_mode_side_mismatch")
    dexhand = seq_info.get("dexhand")
    if type(dexhand) is not str or not dexhand:
        _reject("invalid_seq_info")
    hand_key = f"{dexhand}_{side}"
    if hand_key not in SOURCE_HANDS or SOURCE_HANDS[hand_key].side != side:
        _reject("unsupported_hand")
    context.hand = hand_key
    joint_dim = len(SOURCE_HANDS[hand_key].joint_order)
    geometry_path = (
        _resolve_object_geometry(sequence_path, seq_info, side, source, geometry_resolver)
        if require_geometry else None
    )

    rollout_path = sequence_path / "rollouts.hdf5"
    if not rollout_path.is_file():
        _reject("missing_rollouts")
    try:
        with h5py.File(rollout_path, "r") as h5:
            _reject_nonlocal_hdf5_links(h5)
            rollouts = _hard_member(h5, "rollouts", h5py.Group, "missing_successful_rollouts")
            successful = _hard_member(
                rollouts, "successful", h5py.Group, "missing_successful_rollouts"
            )
            names = sorted(successful.keys())
            if not names:
                _reject("no_successful_rollout")

            best_key: tuple[float, str] | None = None
            best_data = None
            for name in names:
                rollout = _hard_member(
                    successful, name, h5py.Group, "invalid_rollout_group"
                )
                data = _validate_rollout(
                    rollout, side=side, frames=frames, joint_dim=joint_dim
                )
                key = (data[0], name)
                if best_key is None or key > best_key:
                    best_key = key
                    best_data = data
                    rollout_name = name
    except _SequenceRejected:
        raise
    except (OSError, RuntimeError, ValueError):
        _reject("invalid_rollouts")

    assert best_data is not None and best_key is not None
    score, q, dq, root_state, object_state, tip_force = best_data
    try:
        hash_paths = [sequence_path / "seq_info.json", rollout_path]
        if geometry_path is not None:
            hash_paths.append(geometry_path)
        context.input_sha256 = _hash_files(tuple(hash_paths))
    except OSError:
        _reject("input_hash_failed")
    return LoadedHandSequence(
        source=source,
        sequence=sequence_path.name,
        side=side,
        source_hand_key=hand_key,
        rollout_name=rollout_name,
        total_reward=score,
        q=_readonly_copy(q),
        dq=_readonly_copy(dq),
        root_state=_readonly_copy(root_state),
        object_state=_readonly_copy(object_state),
        tip_force=None if tip_force is None else _readonly_copy(tip_force),
        object_geometry_path=geometry_path,
        source_sha256=context.input_sha256,
    )


def load_best_successful_rollout(
    path: str | Path,
    source: str,
    side: str,
    *,
    geometry_resolver: VerifiedGeometryResolver | None = None,
) -> LoadedHandSequence:
    """Validate a sequence atomically and return its deterministic best rollout."""

    _validate_geometry_resolver(geometry_resolver)
    context = _AuditContext()
    try:
        return _load_best(path, source, side, context, geometry_resolver)
    except _SequenceRejected as error:
        raise ValueError(f"sequence rejected: {error}") from error


def load_best_successful_trajectory(path: str | Path, source: str, side: str) -> LoadedHandSequence:
    """Load a successful rollout for fingertip trajectory learning only."""
    _validate_geometry_resolver(None)
    context = _AuditContext()
    try:
        return _load_best(path, source, side, context, None, require_geometry=False)
    except _SequenceRejected as error:
        raise ValueError(f"sequence rejected: {error}") from error


def audit_sequence(
    path: str | Path,
    source: str,
    side: str,
    *,
    geometry_resolver: VerifiedGeometryResolver | None = None,
) -> SequenceAudit:
    """Return one stable audit record; invalid sequences never return partial arrays."""

    _validate_geometry_resolver(geometry_resolver)
    context = _AuditContext()
    try:
        loaded = _load_best(path, source, side, context, geometry_resolver)
    except _SequenceRejected as error:
        if not context.input_sha256:
            context.input_sha256 = _best_effort_input_hash(path, side)
        return SequenceAudit(
            source=source if type(source) is str else "",
            sequence=_sequence_name(path),
            side=side if type(side) is str else "",
            hand=context.hand,
            frames=context.frames,
            accepted=False,
            reason=str(error),
            input_sha256=context.input_sha256,
        )
    return SequenceAudit(
        source=loaded.source,
        sequence=loaded.sequence,
        side=loaded.side,
        hand=loaded.source_hand_key,
        frames=loaded.q.shape[0],
        accepted=True,
        reason="accepted",
        input_sha256=loaded.source_sha256,
    )


__all__ = [
    "INTERACTION_MODE_SIDES",
    "LoadedHandSequence",
    "SequenceAudit",
    "audit_sequence",
    "load_best_successful_rollout",
    "load_best_successful_trajectory",
]
