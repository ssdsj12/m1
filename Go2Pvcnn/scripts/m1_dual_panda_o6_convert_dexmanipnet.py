#!/usr/bin/env python3
"""Convert pinned DexManipNet geometry into deterministic offline fingertip shards."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.preprocess import (
    convert_loaded_sequence,
    object_geometry_sha256,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import SOURCE_HANDS
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import (
    deterministic_group_split,
    write_shards,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.urdf_fk import (
    UrdfKinematicTree,
)


_SOURCE_SIDES = {"favor": ("rh",), "oakinkv2": ("lh", "rh")}


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "external" / "dexmanipnet"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _load_pinned_manifest(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned download manifest: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("download manifest is invalid") from error
    if type(document) is not dict:
        raise ValueError("download manifest must be a JSON object")
    if document.get("dataset_revision") != DEXMANIPNET_REVISION:
        raise ValueError("download manifest has the wrong DexManipNet revision")
    if document.get("maniptrans_commit") != MANIPTRANS_COMMIT:
        raise ValueError("download manifest has the wrong ManipTrans commit")
    archives = document.get("archives")
    required_archives = {"dexmanipnet_favor.tar.gz", "dexmanipnet_oakinkv2.tar.gz"}
    if type(archives) is not list or len(archives) != len(required_archives) or {
        item.get("name") for item in archives if type(item) is dict
    } != required_archives:
        raise ValueError("download manifest does not contain both pinned archives")
    for item in archives:
        if type(item) is not dict:
            raise ValueError("download manifest archive record is invalid")
        archive_sha = item.get("sha256")
        if (
            type(archive_sha) is not str
            or len(archive_sha) != 64
            or any(character not in "0123456789abcdef" for character in archive_sha)
        ):
            raise ValueError("download manifest archive SHA-256 is invalid")
        size = item.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("download manifest archive size is invalid")
    tree = document.get("maniptrans_tree")
    if (
        type(tree) is not str
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in tree)
    ):
        raise ValueError("download manifest has no valid ManipTrans tree")
    source_document = {"maniptrans_commit": MANIPTRANS_COMMIT, "maniptrans_tree": tree}
    return _sha256_file(path), sha256(_canonical_json(source_document)).hexdigest()


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
) -> Path:
    """Run the offline-only conversion; rejected sides contribute audit rows only."""

    manifest = root / "manifests" / "download_manifest.json" if manifest_path is None else manifest_path
    archive_manifest_sha, source_manifest_sha = _load_pinned_manifest(manifest)
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

    windows = []
    audit_rows: list[dict[str, Any]] = []
    group_hands: dict[str, str] = {}
    tree_cache: dict[str, UrdfKinematicTree] = {}
    for source in sorted(extracted):
        sequences_root = extracted[source] / "sequences"
        for sequence_path in sorted(sequences_root.iterdir(), key=lambda item: item.name):
            if not sequence_path.is_dir():
                continue
            for side in _SOURCE_SIDES[source]:
                source_audit = audit_sequence(sequence_path, source=source, side=side)
                row = _audit_dict(source_audit)
                if not source_audit.accepted:
                    audit_rows.append(row)
                    continue
                loaded = load_best_successful_rollout(sequence_path, source=source, side=side)
                try:
                    if loaded.source_hand_key not in tree_cache:
                        spec = SOURCE_HANDS[loaded.source_hand_key]
                        hand_urdf = source_root.joinpath(*Path(spec.urdf_relpath).parts)
                        tree_cache[loaded.source_hand_key] = UrdfKinematicTree.from_file(hand_urdf)
                    converted = convert_loaded_sequence(loaded, tree_cache[loaded.source_hand_key])
                    if not converted:
                        raise ValueError("fewer than 20 future 100 Hz nodes")
                    row.update(
                        {
                            "accepted": True,
                            "reason": "accepted",
                            "windows": len(converted),
                            "object_geometry_sha256": object_geometry_sha256(
                                loaded.object_geometry_path
                            ),
                        }
                    )
                    group = converted[0].source_group
                    group_hands[group] = loaded.source_hand_key
                    windows.extend(converted)
                except (OSError, TypeError, ValueError) as error:
                    row.update(
                        {
                            "accepted": False,
                            "reason": f"conversion_rejected:{error}",
                            "windows": 0,
                        }
                    )
                audit_rows.append(row)

    if not windows:
        raise RuntimeError("no DexManipNet sequence produced a usable fingertip window")
    split = deterministic_group_split([window.source_group for window in windows], seed=seed)
    aggregate = write_shards(
        output,
        windows,
        split,
        shard_size=shard_size,
        audits=audit_rows,
        archive_manifest_sha256=archive_manifest_sha,
        source_manifest_sha256=source_manifest_sha,
        group_hands=group_hands,
    )
    print(
        json.dumps(
            {
                "aggregate_manifest": str(output / "aggregate_manifest.json"),
                "aggregate_sha256": aggregate.aggregate_sha256,
                "shards": len(aggregate.shards),
                "windows": len(windows),
            },
            sort_keys=True,
        )
    )
    return output / "aggregate_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root(), help="pinned external data root")
    parser.add_argument("--output", type=Path, help="new artifact directory (default: ROOT/artifacts/shards)")
    parser.add_argument("--manifest", type=Path, help="explicit pinned download manifest")
    parser.add_argument("--seed", type=int, default=42, help="deterministic group split seed")
    parser.add_argument("--shard-size", type=int, default=4096, help="maximum windows per NPZ shard")
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
    )


if __name__ == "__main__":
    main()
