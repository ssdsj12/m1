"""Prepare finite feasible rows from an incomplete teacher rollout for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(dataset_dir: Path, groups: int = 10) -> dict[str, int]:
    source = dataset_dir / "teacher_failures.npz"
    with np.load(source, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {"state", "teacher_action", "teacher_task", "phase", "seed", "trial", "reason", "action_order"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"missing teacher arrays: {missing}")
    feasible = data["reason"].astype(str) == "mission_not_done"
    for name in ("state", "teacher_action", "teacher_task", "phase"):
        feasible &= np.isfinite(data[name].reshape(data[name].shape[0], -1)).all(axis=1)
    indices = np.flatnonzero(feasible)
    if len(indices) < groups:
        raise ValueError(f"only {len(indices)} finite feasible rows; need at least {groups}")
    output = {
        name: values[indices]
        for name, values in data.items()
        if name not in {"reason", "action_order"}
    }
    output["trial"] = np.minimum(
        np.arange(len(indices), dtype=np.int64) * groups // len(indices), groups - 1
    )
    output["action_order"] = data["action_order"]
    np.savez_compressed(dataset_dir / "teacher_success.npz", **output)
    report = {
        "source_rows": int(data["reason"].shape[0]),
        "prepared_rows": int(len(indices)),
        "rejected_rows": int(data["reason"].shape[0] - len(indices)),
        "split_groups": groups,
        "mission_complete": False,
    }
    (dataset_dir / "prepared_teacher_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    args = parser.parse_args()
    if args.groups < 3:
        parser.error("--groups must be at least 3")
    print(json.dumps(prepare(args.dataset_dir, args.groups), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
