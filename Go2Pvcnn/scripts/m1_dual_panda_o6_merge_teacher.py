"""Merge teacher NPZ shards without starting Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge(shards: list[Path], output_dir: Path) -> int:
    archives = []
    for shard in shards:
        with np.load(shard / "teacher_failures.npz", allow_pickle=False) as source:
            archives.append({name: source[name] for name in source.files})
    names = set(archives[0])
    if any(set(archive) != names for archive in archives[1:]):
        raise ValueError("teacher shard schemas differ")
    merged = {}
    for name in sorted(names):
        if name == "action_order":
            if any(not np.array_equal(archive[name], archives[0][name]) for archive in archives[1:]):
                raise ValueError("action order differs across shards")
            merged[name] = archives[0][name]
        else:
            merged[name] = np.concatenate([archive[name] for archive in archives], axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "teacher_failures.npz", **merged)
    return int(merged["reason"].shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"merged_rows={merge(args.shards, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
