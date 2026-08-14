# M1 + Panda Single Articulation Build

## Purpose

记录 T400 asset/wrench foundation Task 2 的 TDD、Panda URDF 转换、RobotAssembler 单 articulation 装配和生成文件校验。

## Stage And Todo

- Stage: T400 / asset-wrench foundation / Task 2
- Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Input Conditions

- `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/robots/panda_arm_hand.urdf`
- Git Ref: unavailable
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy

## RED

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py::test_builder_declares_single_robot_mount_contract
```

Result: expected missing-builder failure, `1 failed in 0.03s`, exit code `1`.

The local Isaac Sim 4.5 headless experience also exposed a missing extension enable. A second static RED required `enable_extension("isaacsim.robot_setup.assembler")` after AppLauncher startup and before the assembler import: `1 failed in 0.02s`, exit code `1`.

## GREEN

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py
```

Result after the compatibility fix: `4 passed in 0.01s`, exit code `0`.

## Real Build

The requested wrapper defaults to base Python 3.13 in this shell. The compatible local stack is `/home/xk/miniconda3/envs/loco/bin/python` (Python 3.10, Isaac Sim 4.5.0, Isaac Lab 2.1.0), whose RobotAssembler source provides `assemble_articulations`, `single_robot`, and `AssembledRobot.fixed_joint`.

Wrapper command used with that environment:

```bash
OMNI_KIT_ACCEPT_EULA=Y CONDA_PREFIX=/home/xk/miniconda3/envs/loco \
PATH=/home/xk/miniconda3/envs/loco/bin:$PATH TERM=xterm \
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

The wrapper generated both files, but `isaaclab.sh` ends its `-p` case with `break`, so a Python failure can be masked as shell exit `0`. A direct reliable-status rerun used the same compatible interpreter:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 30 \
/home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Result: printed `/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd`, exit code `0`. The in-builder assertion confirmed the fixed joint path is `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`. Isaac Sim emitted URDF mass/joint-axis warnings and RobotAssembler `_refresh_asset` recomposition warnings, but no traceback or assertion failure.

## Checksums

```bash
cd Go2Pvcnn
sha256sum assets/m1_panda/panda/panda.usd assets/m1_panda/m1_panda.usd > assets/m1_panda/generated_files.sha256
sha256sum -c assets/m1_panda/generated_files.sha256
```

Result: two rows verified, exit code `0`.

- `panda/panda.usd`: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- `m1_panda.usd`: `67d4288bfd43141d5a7cdd6126ed0806e4e5cc6ac3b84f6238ed434ddac15ab4`

## Result And Follow-up

Task 2 build contract is implemented and generated assets are present. Independent reopen inspection entered Isaac Sim but hung during shutdown and was terminated; post-reopen articulation-root counting and physics stepping remain unverified and belong to later runtime validation.

## Key Files

- `Go2Pvcnn/scripts/build_m1_panda_asset.py`
- `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- `Go2Pvcnn/assets/m1_panda/panda/panda.usd`
- `Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- `Go2Pvcnn/assets/m1_panda/generated_files.sha256`
- [Task 2 report](../../.superpowers/sdd/task-2-report.md)
