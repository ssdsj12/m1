#!/usr/bin/env python3
"""Prepare a stable roll checkpoint for first-time M1 leg-action release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2_pvcnn.tasks.m1_curriculum import expand_checkpoint_observations, prepare_wave_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observation-dim", type=int, default=None)
    parser.add_argument("--leg-noise-std", type=float, default=None)
    parser.add_argument("--preserve-leg-outputs", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    prepared = checkpoint
    if not args.preserve_leg_outputs:
        prepared = prepare_wave_checkpoint(checkpoint, leg_noise_std=args.leg_noise_std)
    if args.observation_dim is not None:
        prepared = expand_checkpoint_observations(prepared, args.observation_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
