#!/usr/bin/env python3
"""Convert pinned DexManipNet geometry into deterministic offline fingertip shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    DEXMANIPNET_REVISION,
    MANIPTRANS_COMMIT,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.dexmanipnet import (
    audit_sequence,
    load_best_successful_rollout,
    load_best_successful_trajectory,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.download import (
    verify_pinned_external_inputs,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.preprocess import (
    convert_loaded_sequence,
    convert_loaded_sequence_trajectory_only,
    object_geometry_sha256,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import SOURCE_HANDS
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import (
    StreamingShardWriter,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.urdf_fk import (
    UrdfKinematicTree,
)


_SOURCE_SIDES = {"favor": ("rh",), "oakinkv2": ("lh", "rh")}
_ARCHIVE_NAMES = ("dexmanipnet_favor.tar.gz", "dexmanipnet_oakinkv2.tar.gz")
_MANIPTRANS_REPOSITORY = "https://github.com/ManipTrans/ManipTrans.git"


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "external" / "dexmanipnet"


def _audit_dict(audit: object) -> dict[str, Any]:
    return {
        "accepted": bool(audit.accepted),
        "frames": int(audit.frames),
        "hand": audit.hand,
        "input_sha256": audit.input_sha256,
        "reason": audit.reason,
        "sequence": audit.sequence,
        "side": audit.side,
        "source": audit.source,
    }


def run(
    root: Path,
    output: Path,
    *,
    seed: int,
    shard_size: int,
    manifest_path: Path | None = None,
    trajectory_only: bool = False,
) -> Path:
    """Run the offline-only conversion; rejected sides contribute audit rows only."""

    manifest = root / "manifests" / "download_manifest.json" if manifest_path is None else manifest_path
    verified = verify_pinned_external_inputs(
        root,
        manifest,
        archive_names=_ARCHIVE_NAMES,
        expected_revision=DEXMANIPNET_REVISION,
        expected_commit=MANIPTRANS_COMMIT,
        source_repository=_MANIPTRANS_REPOSITORY,
    )
    source_root = root / "source_maniptrans"
    extracted = {
        "favor": root / "extracted" / "dexmanipnet_favor",
        "oakinkv2": root / "extracted" / "dexmanipnet_oakinkv2",
    }
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing pinned ManipTrans source: {source_root}")
    for source, source_path in extracted.items():
        if not (source_path / "sequences").is_dir():
            raise FileNotFoundError(f"missing extracted {source} sequences: {source_path}")

    audit_rows: list[dict[str, Any]] = []
    group_hands: dict[str, str] = {}
    tree_cache: dict[str, UrdfKinematicTree] = {}
    writer = StreamingShardWriter(output, seed=seed, shard_size=shard_size)
    window_count = 0
    scanned_count = 0
    accepted_count = 0
    try:
        for source in sorted(extracted):
            sequences_root = extracted[source] / "sequences"
            for sequence_path in sorted(sequences_root.iterdir(), key=lambda item: item.name):
                if not sequence_path.is_dir():
                    continue
                for side in _SOURCE_SIDES[source]:
                    scanned_count += 1
                    print(f"progress-start scanned={scanned_count} source={source} sequence={sequence_path.name} side={side}", flush=True)
                    if trajectory_only:
                        try:
                            loaded = load_best_successful_trajectory(sequence_path, source=source, side=side)
                            row = {"accepted": True, "frames": int(loaded.q.shape[0]), "hand": loaded.source_hand_key,
                                   "input_sha256": loaded.source_sha256, "reason": "accepted", "sequence": sequence_path.name,
                                   "side": side, "source": source}
                        except (OSError, TypeError, ValueError) as error:
                            audit_rows.append({"accepted": False, "frames": 0, "hand": None, "input_sha256": "",
                                               "reason": f"trajectory_load_rejected:{error}", "sequence": sequence_path.name,
                                               "side": side, "source": source})
                            print(f"progress scanned={scanned_count} accepted={accepted_count} windows={window_count} rejected={source}/{sequence_path.name}/{side}:trajectory_load", flush=True)
                            continue
                    else:
                        source_audit = audit_sequence(sequence_path, source=source, side=side)
                        row = _audit_dict(source_audit)
                        if not source_audit.accepted:
                            audit_rows.append(row)
                            print(f"progress scanned={scanned_count} accepted={accepted_count} windows={window_count} rejected={source}/{sequence_path.name}/{side}:{source_audit.reason}", flush=True)
                            continue
                        loaded = load_best_successful_rollout(sequence_path, source=source, side=side)
                    try:
                        if loaded.source_hand_key not in tree_cache:
                            spec = SOURCE_HANDS[loaded.source_hand_key]
                            hand_urdf = source_root.joinpath(*Path(spec.urdf_relpath).parts)
                            tree_cache[loaded.source_hand_key] = UrdfKinematicTree.from_file(hand_urdf)
                        converted = (convert_loaded_sequence_trajectory_only(loaded, tree_cache[loaded.source_hand_key])
                                     if trajectory_only else convert_loaded_sequence(loaded, tree_cache[loaded.source_hand_key]))
                        if not converted:
                            raise ValueError("fewer than 20 future 100 Hz nodes")
                        group = converted[0].source_group
                        if any(window.source_group != group for window in converted):
                            raise ValueError("converted sequence has inconsistent source groups")
                        geometry_sha256 = None if trajectory_only else object_geometry_sha256(loaded.object_geometry_path)
                    except (OSError, TypeError, ValueError) as error:
                        row.update(
                            {
                                "accepted": False,
                                "reason": f"conversion_rejected:{error}",
                                "windows": 0,
                            }
                        )
                    else:
                        # No later sequence-level rejection may follow this append:
                        # a rejected audit row must never have a spooled window.
                        writer.append(converted)
                        row.update(
                            {
                                "accepted": True,
                                "reason": "accepted",
                                "windows": len(converted),
                                **({} if geometry_sha256 is None else {"object_geometry_sha256": geometry_sha256}),
                            }
                        )
                        group_hands[group] = loaded.source_hand_key
                        window_count += len(converted)
                        accepted_count += 1
                    audit_rows.append(row)
                    print(f"progress scanned={scanned_count} accepted={accepted_count} windows={window_count} last={source}/{sequence_path.name}/{side}", flush=True)

        if not window_count:
            raise RuntimeError("no DexManipNet sequence produced a usable fingertip window")
        aggregate = writer.finalize(
            audits=audit_rows,
            archive_manifest_sha256=verified["archive_manifest_sha256"],
            source_manifest_sha256=verified["source_manifest_sha256"],
            group_hands=group_hands,
            verified_inputs=verified["facts"],
        )
    except BaseException:
        writer.abort()
        raise
    print(
        json.dumps(
            {
                "aggregate_manifest": str(output / "aggregate_manifest.json"),
                "aggregate_sha256": aggregate.aggregate_sha256,
                "shards": len(aggregate.shards),
                "windows": window_count,
            },
            sort_keys=True,
        )
    )
    return output / "aggregate_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root(), help="pinned external data root")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        help="new artifact directory (default: ROOT/artifacts/shards)",
    )
    parser.add_argument("--manifest", type=Path, help="explicit pinned download manifest")
    parser.add_argument("--seed", type=int, default=42, help="deterministic group split seed")
    parser.add_argument("--shard-size", type=int, default=4096, help="maximum windows per NPZ shard")
    parser.add_argument("--trajectory-only", action="store_true", help="learn fingertip motion without object geometry")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output = args.output or args.root / "artifacts" / "shards"
    run(
        args.root,
        output,
        seed=args.seed,
        shard_size=args.shard_size,
        manifest_path=args.manifest,
        trajectory_only=args.trajectory_only,
    )


if __name__ == "__main__":
    main()
