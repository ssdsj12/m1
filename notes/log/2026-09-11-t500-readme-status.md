# T500 root README status

- Parent: [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
- Stage: repository landing documentation.
- Baseline Ref: `7c6701d`.
- Candidate Ref: README documentation working tree.
- Key File: [root README](../../README.md).

## Purpose

Replace the placeholder root README with a branch-specific overview that separates implemented
T500 improvements from physical acceptance that remains open.

## Included documentation

- SO(3), contact sensing, lift motion primitives, exact-step multi-GPU orchestration, and teacher data improvements.
- Focused verification evidence and its limits.
- Probe, Play, and fixed-budget runner entry points.
- Outstanding cross-seed, 30/30 physical acceptance, offline model, Vulkan/DRM, and WebRTC work.
- Relative links to the T500 branch page, design, plan, and verification logs.

## Verification

- Markdown links are repository-relative.
- `git diff --check` must pass before commit.
- No claim is made that real Isaac Sim cross-seed or formal 30/30 acceptance has passed.
