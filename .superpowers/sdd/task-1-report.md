# Task 1 Report: Freeze Source, Sample, and Artifact Contracts

## Status

DONE

## Implementation

- Added the standalone `expert_fingertip_prior` package with frozen DexManipNet and ManipTrans pins, five-finger order, dimensions, seven `PriorPhase` values, and left-reflection contract.
- Added strict geometry-only `ExpertWindow`, four-component `MixtureDistribution`, and `StudentArtifactMetadata` dataclasses. Their construction validates tensor type/dtype/shape/finiteness, enum, SHA, model dimensions, frozen order, mirror matrix, seeds, and artifact fields.
- Added explicit pinned ManipTrans source manifests for Inspire and Shadow right/left hands, including official URDF-relative paths, joint orders, palm links, and thumb/index/middle/ring/pinky tip links.
- Added focused contract tests and the required T500 todo/log evidence.

## Files

- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/__init__.py`
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/contracts.py`
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/sources.py`
- `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_contracts.py`
- `notes/log/2026-09-11-t500-dexmanipnet-prior-contracts.md`, `notes/log/index.md`, `notes/todo.md`, and `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`

## TDD Evidence

### RED

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py
```

Before production code, collection failed as intended with:

```text
ModuleNotFoundError: No module named 'go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior'
1 error in 0.80s
```

### GREEN and current-contract regression

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py
```

Relevant output:

```text
................                                                         [100%]
16 passed in 0.78s
```

### Pinned source audit

The pinned ManipTrans checkout at `a3d08cfe3c3a5868a7f057533bcaf759c5af4705` was parsed outside the product package. Every registered spec's URDF contained its declared palm, all five tip links, and every declared joint: `verified 4 pinned source-hand specs`.

## Commits

- `8b5523cc3b9d181cf2e4bb94fe565bd8667210ab` — `feat: freeze DexManipNet fingertip prior contracts`
- `3e58c82c027eb61b21ab9f90427a469e1299e6af` — `docs: record fingertip prior contracts`

## Self-Review

- Confirmed the staged feature commit contained only Task 1 package, test, and repository-required evidence; Graphify cache/memory changes were neither staged nor committed.
- `git diff --cached --check` passed before both commits.
- Verified the frozen output order is exactly `15 + 15 + 5 + 7 = 42`, and the target is exactly `(20, 5, 3)`.
- Confirmed the source registry is static: it has no source checkout, URDF parser, download, or runtime external-data dependency.

## Concerns

None for Task 1. No DexManipNet data was downloaded, no artifact was trained, and no Hand MPC or Isaac physical behavior was changed or claimed.

## Review Fix: Strict Artifact Layout and Registry Immutability

The Task 1 review findings are addressed by `eddcd91d54fe292455a00f4aa00ce562f765c799` (`fix: harden fingertip prior contracts`):

- `StudentArtifactMetadata` now stores and validates the exact geometry-only input field order (`fingertip_position_palm`, `fingertip_velocity_palm`, `contact_mask`, `phase_one_hot`) and mixture output axes (`mixture_component`, `horizon`, `finger`, `xyz`).
- Frozen schema dimensions and seeds require strict `int` values; `hidden` requires a non-empty tuple of positive strict ints; SHA, dtype, finite mirror matrix, phase/finger orders, and layouts are covered by focused negative tests.
- `SourceHandSpec` requires tuple-valued `joint_order` and `fingertip_links`, with strict non-empty string names. The test fixes all four registry keys and every official field value.
- `LEFT_REFLECTION` is immutable tuple data and `SOURCE_HANDS` is a `MappingProxyType`.

Review RED:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py
```

Expected failure observed: missing `MIXTURE_OUTPUT_AXIS_ORDER` during collection (`1 error in 0.81s`).

Review GREEN/current regression:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py
```

Result: `30 passed in 0.79s`.

Self-review: staged only Task 1 contracts, source registry, focused test, and required evidence; `git diff --check` passed; no Graphify or parent-owned progress/brief files were staged. No new concern.
