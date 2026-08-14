# M1 + Panda Offline Asset Inputs

## Purpose

记录 T400 asset/wrench foundation Task 1 的实现与验证证据：将 M1 physics USD 和 Isaac Sim 自带 Panda URDF 包闭包复制到项目内，建立后续转换与装配只依赖的本地资产输入。

## Stage

T400 / asset-wrench foundation / Task 1 / project-owned offline asset inputs。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Source And Destination

- M1 source: `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/`
- M1 destination: `Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/`
- Panda source: `/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description/`
- Panda destination: `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/`
- Local M1 overlay: `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- Source manifest: `Go2Pvcnn/assets/m1_panda/source_manifest.json`

## RED

Command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
```

Result before the asset root was created: `2 failed in 0.04s`. Both failures were expected `FileNotFoundError` results for the missing `source_manifest.json` and `m1_floating.usda`.

## GREEN

The same focused command after implementation produced:

```text
..                                                                       [100%]
2 passed in 0.01s
```

Final review rerun on 2026-08-14 produced the same `2 passed in 0.01s` result with exit code `0`.

## Checksum Verification

The checksum manifest excludes itself and contains 35 entries. Full verification command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
sha256sum -c assets/m1_panda/source_files.sha256
```

Result: all 35 entries reported `成功`; exit code `0`.

- `source_manifest.json`: `93c1b13efb96b2ab0727731b2e4c3020c5f705741ad7eba0d107ab0aa433f541`
- `m1_floating.usda`: `b4eb87c46a12d653e50aa012f32de45abd6d5447ca72a9a5b2afdaf60531e63c`
- `source_files.sha256`: `e7c44f8cf461a9d90f981357fa36699e226e7ffba6c85bc1a51a946253af0abb`

## Result

Pass. Task 1 now produces project-owned offline M1 and Panda source inputs, an exact manifest, and a relative local M1 overlay. Task 2 remains pending.

## Scope

No simulation, USD runtime loading, URDF conversion, Robot Assembler operation, or wrench runtime probe was performed in Task 1.

## Git Refs

- Git Ref: unavailable
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Key Files:
  - `Go2Pvcnn/assets/m1_panda/`
  - `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
  - [Task 1 report](../../.superpowers/sdd/task-1-report.md)

## Follow-up

Execute foundation Task 2 using only the project-contained inputs established here.
