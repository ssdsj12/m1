# Task 1 Report: Project-Owned Offline Asset Inputs

## Status

DONE

Task 1 is implemented. The project now contains local M1 and Panda source trees, the exact source manifest, a floating M1 overlay with a project-relative sublayer, the required static tests, and a checksum checkpoint.

## Implementation

- Copied the authoritative M1 tree from `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/` to `Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/` with `cp -a`.
- Copied the Isaac Sim-bundled Panda package from `/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description/` to `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/` with `cp -a`.
- Added the exact specified `source_manifest.json`.
- Reproduced the existing `assets/m1_usd/ZJ_V3_URDF_V1_0_floating.usda` as `assets/m1_panda/m1_floating.usda`, changing only its sublayer from the machine-local absolute source path to `@./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@`.
- Added the two specified static tests.
- Generated `assets/m1_panda/source_files.sha256` with the corrected task command that excludes the checksum file itself.

Produced interfaces:

- `M1_LOCAL_OVERLAY`: `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- `PANDA_LOCAL_URDF`: `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/robots/panda_arm_hand.urdf`
- Project-contained source root: `Go2Pvcnn/assets/m1_panda/`

## TDD Evidence

### RED

Command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
```

Result before production assets were added:

```text
FF
2 failed in 0.04s
```

Both failures were the expected `FileNotFoundError`: `assets/m1_panda/source_manifest.json` and `assets/m1_panda/m1_floating.usda` did not exist. This confirmed the test was red for the missing feature, not a typo or collection problem.

### GREEN

The same command after copying the source trees and adding the manifest/overlay produced:

```text
..                                                                       [100%]
2 passed in 0.01s
```

### Final focused regression

The same command was run once more after checksum creation and self-review:

```text
..                                                                       [100%]
2 passed in 0.01s
```

No warnings or errors were emitted.

## Checksum Checkpoint

Corrected generation command:

```bash
find assets/m1_panda -type f ! -name source_files.sha256 -print0 | sort -z | xargs -0 sha256sum > assets/m1_panda/source_files.sha256
```

- Checksum file: `Go2Pvcnn/assets/m1_panda/source_files.sha256`
- Size: 4,874 bytes
- Lines: 35
- SHA-256 of the resulting checksum file: `e7c44f8cf461a9d90f981357fa36699e226e7ffba6c85bc1a51a946253af0abb`
- Manifest SHA-256: `93c1b13efb96b2ab0727731b2e4c3020c5f705741ad7eba0d107ab0aa433f541`
- Overlay SHA-256: `b4eb87c46a12d653e50aa012f32de45abd6d5447ca72a9a5b2afdaf60531e63c`
- Test SHA-256: `c4b02a0faee877e89122a315f1a9526dcd0efb9b996130d938d006d3f57cbe08`

Verification command:

```bash
sha256sum -c assets/m1_panda/source_files.sha256
```

Result: all 35 entries reported success.

## File Changes

Task 1 files and required T400 verification records:

- `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- `Go2Pvcnn/assets/m1_panda/source_manifest.json`
- `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- `Go2Pvcnn/assets/m1_panda/source_files.sha256`
- `Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/` containing 4 source files
- `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/` containing 29 source files
- `.superpowers/sdd/task-1-report.md`
- `notes/log/2026-08-14-m1-panda-offline-asset-inputs.md`
- `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- `notes/todo.md`
- `notes/log/index.md`

The resulting `assets/m1_panda` tree is approximately 126 MiB and contains 36 files including the manifest, overlay, and checksum file.

Git commits: unavailable because `/home/xk/coding/M1` is not a Git work tree.

## Self-Review

- Confirmed both authoritative entry files existed before copying.
- `diff -qr` found no differences between either authoritative source tree and its copied destination.
- A normalized `diff -u` confirmed the new overlay differs from the existing overlay only in the required sublayer path.
- The manifest content and paths match the task brief exactly.
- The overlay contains the required relative reference and contains no `/home/`, `omniverse://`, `http://`, or `https://` reference.
- File counts match the sources: 4 M1 files and 29 Panda files.
- No runtime code outside Task 1 was modified; repository notes were aligned with the completed Task 1 evidence during review follow-up.

## Concern

None. The corrected checksum command excludes `source_files.sha256` itself, so the complete manifest is directly verifiable.

No simulation or USD runtime loading was requested or performed in Task 1; verification is static asset ownership, path, copy identity, and checksum verification only.

## Checksum Self-Reference Fix

The corrected generation command was run:

```bash
find assets/m1_panda -type f ! -name source_files.sha256 -print0 | sort -z | xargs -0 sha256sum > assets/m1_panda/source_files.sha256
```

The complete checksum verification was then run:

```bash
sha256sum -c assets/m1_panda/source_files.sha256
```

Output summary: all 35 listed files reported `成功`; exit code `0`.

The Task 1 focused test was rerun:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
```

Output:

```text
..                                                                       [100%]
2 passed in 0.01s
```

The regenerated checksum file has 35 lines, contains no self-entry, and has SHA-256 `e7c44f8cf461a9d90f981357fa36699e226e7ffba6c85bc1a51a946253af0abb`.

## T400 Verification Record Review Fix

The Task 1 review found that implementation evidence had not yet been aligned into the repository T400 working memory. This documentation-only review item was fixed by:

- adding `notes/log/2026-08-14-m1-panda-offline-asset-inputs.md` with source/destination, RED/GREEN, hashes, checksum, Git-unavailable, and no-simulation evidence;
- updating `notes/todo/T400-m1-panda-force-aware-teacher-student.md` to state Task 1 is complete and Task 2 is pending;
- updating the T400 current state and Recent Logs in `notes/todo.md`;
- updating Recent Logs and the T400 topic index in `notes/log/index.md`.

Post-fix verification commands:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
sha256sum -c assets/m1_panda/source_files.sha256
```

Output summary: focused tests `2 passed in 0.01s`; all 35 checksum entries reported `成功`; both commands exited `0`. No simulation was run.
