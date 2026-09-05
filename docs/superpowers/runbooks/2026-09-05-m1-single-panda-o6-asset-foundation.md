# M1 + Right Panda + Right O6 Asset Foundation Runbook

## Scope

This runbook builds and verifies T600.1 only. It does not register an environment or run MPC/RL.

## Inputs

- Read-only vendor O6 source: `/home/xk/coding/o6asset`
- Project O6 source manifest: `Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json`
- Project M1/Panda source: `Go2Pvcnn/assets/m1_panda/`
- Python environment: `/home/xk/miniconda3/envs/go2/bin/python`
- Device: NVIDIA GPU0 via `CUDA_VISIBLE_DEVICES=0`

## Build

Run from the repository root:

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/build_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --o6-source-root assets/m1_dual_panda_o6 \
  --force-panda-conversion \
  --headless
```

The Isaac importer and USD flattener may serialize equivalent material and
prototype child order differently in the Panda base and prefixed O6
intermediate layers. The project-owned versions of those layers are pinned by
Git LFS. The combined `m1_single_panda_o6.usd` and static build manifest must
retain their accepted hashes across a rebuild.

## Verify

Run from `Go2Pvcnn/` after the build:

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --steps 2000 \
  --device cuda:0 \
  --headless
```

Accept only exit `0` plus `hard_gates_passed=true`. The verifier must report
34 measured physical DOFs, 29 active controls, four-wheel contact ratio `1.0`,
and zero non-finite, hard-limit, unexpected-contact, unexpected-reset, and
base-instability counts.

## Failure Policy

Do not edit the vendor source, relax a hard gate, or rewrite the manifest after
a failed run. Diagnose the named offline/runtime metric and rerun from the
project-owned generated asset. Do not accept or commit intermediate USD churn
based only on a binary hash change; compare the USDA scene content and rerun the
full physical gate.
