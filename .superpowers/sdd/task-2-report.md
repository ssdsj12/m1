# Task 2 Report: Local Panda Conversion And M1 Assembly

## Status

DONE_WITH_CONCERNS. The specified builder contract, static TDD, real URDF conversion, RobotAssembler assembly, output path, and hashes are complete. Independent post-reopen articulation-root counting/physics stepping is not verified.

Git Ref: unavailable

## Implementation

- Added `Go2Pvcnn/scripts/build_m1_panda_asset.py`.
- Extended `Go2Pvcnn/tests/test_m1_panda_asset_static.py` with the specified builder contract and a local Isaac Sim compatibility-order regression.
- Consumes only project-contained `m1_floating.usda` and `panda_arm_hand.urdf`.
- Produces `/M1Panda`, mounts `/BASE_LINK` to `/panda_link0`, uses `single_robot=True`, asserts `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`, exports `m1_panda.usd`, and returns its resolved path.
- Generated Panda converter sidecars: `.asset_hash`, `config.yaml`, and `configuration/panda_{base,physics,sensor}.usd`.

## API And Environment Deviations

1. With no compatible conda environment active, `/home/xk/coding/IsaacLab/isaaclab.sh -p` selected base Python 3.13. The project-compatible API was verified in `loco` Python 3.10 / Isaac Sim 4.5.0 / Isaac Lab 2.1.0.
2. Isaac Sim 4.5 headless does not enable `isaacsim.robot_setup.assembler` by default. The minimal source-backed fix calls `enable_extension("isaacsim.robot_setup.assembler")` after AppLauncher startup and before importing RobotAssembler.
3. Isaac Sim 4.5 reads `OMNI_KIT_ACCEPT_EULA`, not `ACCEPT_EULA`.
4. `isaaclab.sh` masks the Python status because its `-p` branch ends with `break`; therefore a direct compatible-interpreter rerun supplied the reliable build exit code.

## TDD Evidence

- RED 1: missing builder, `1 failed in 0.03s`, exit `1`.
- GREEN 1: initial specified contract passed.
- Runtime failure: `ModuleNotFoundError: No module named 'isaacsim.robot_setup'` under the headless experience.
- RED 2: missing extension enable/order, `1 failed in 0.02s`, exit `1`.
- GREEN 2: full static file `4 passed in 0.01s`, exit `0`.

## Real Build Evidence

Wrapper command:

```bash
OMNI_KIT_ACCEPT_EULA=Y CONDA_PREFIX=/home/xk/miniconda3/envs/loco \
PATH=/home/xk/miniconda3/envs/loco/bin:$PATH TERM=xterm \
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Reliable-status rerun:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 30 \
/home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Output included the absolute combined USD path; direct exit code `0`. The build emitted nonfatal Panda URDF importer warnings (missing link8 mass/collider and joint-axis adjustment), `DriverShaderCacheManager` warning, and RobotAssembler `_refresh_asset` invalid temporary reference/payload warnings.

## Generated Hashes

- `Go2Pvcnn/assets/m1_panda/panda/panda.usd` (1592 bytes): `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- `Go2Pvcnn/assets/m1_panda/m1_panda.usd` (2787 bytes): `67d4288bfd43141d5a7cdd6126ed0806e4e5cc6ac3b84f6238ed434ddac15ab4`
- `sha256sum -c assets/m1_panda/generated_files.sha256`: both successful, exit `0`.

## Files

- Created: `Go2Pvcnn/scripts/build_m1_panda_asset.py`
- Modified: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Generated: `Go2Pvcnn/assets/m1_panda/panda/` converter outputs
- Generated: `Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- Generated: `Go2Pvcnn/assets/m1_panda/generated_files.sha256`
- Notes: `notes/log/2026-08-14-m1-panda-single-articulation-build.md`, T400 branch/dashboard/index updates

## Self-review And Concerns

- Builder startup ordering remains AppLauncher first; the one local compatibility enable is after app startup.
- Required constants/call/path assertion exactly match the brief.
- No external asset URL was introduced.
- The build succeeded twice with stable hashes.
- Not verified: independent post-reopen articulation-root count, simulation physics step, visual mount clearance, or collision behavior. The attempted reopen probe hung during Isaac Sim shutdown and was terminated; no asset was changed by that probe.
