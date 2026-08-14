### Task 6: Add and run the deterministic real-simulation wrench probe

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_wrench_probe.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wrench_probe_static.py`
- Modify: `notes/log/2026-08-14-m1-panda-force-aware-teacher-student-design.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`

**Interfaces:**
- Consumes: Gym id `Isaac-M1-Panda-Smoke-v0` and `m1_panda_mount_wrench_b`.
- Produces: JSONL rows for `settle`, `force_x`, `force_y`, `force_z`, `torque_x`, `torque_y`, and `torque_z`, including measured mean wrench and sign/error checks.

- [ ] **Step 1: Write the failing probe-contract test**

```python
from pathlib import Path


def test_probe_covers_all_six_axes_and_clears_external_wrench():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/m1_panda_wrench_probe.py").read_text()
    for case in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"):
        assert f'"{case}"' in source
    assert "set_external_force_and_torque" in source
    assert "torch.zeros(0, 3" in source
    assert 'TASK_ID = "Isaac-M1-Panda-Smoke-v0"' in source
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_wrench_probe_static.py`

Expected: FAIL because the probe script is absent.

- [ ] **Step 3: Implement the six-axis probe**

The probe must:

1. launch one headless environment;
2. hold zero M1 actions for 100 settling steps;
3. find `panda_hand`, `panda_link0`, and `BASE_LINK` body ids exactly once;
4. record the 50-step mean baseline wrench;
5. apply each local test wrench independently to `panda_hand` for 50 steps using magnitudes `20 N` and `5 N·m`;
6. clear the external wrench with `torch.zeros(0, 3, device=robot.device)` between cases;
7. subtract the baseline, write JSONL, and require the excited measured channel to have stable sign and magnitude greater than 20% of the applied channel;
8. exit non-zero on non-finite data, body lookup mismatch, unexpected reset, or failed channel check.

Use this case table verbatim:

```python
CASES = {
    "force_x": ([20.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "force_y": ([0.0, 20.0, 0.0], [0.0, 0.0, 0.0]),
    "force_z": ([0.0, 0.0, 20.0], [0.0, 0.0, 0.0]),
    "torque_x": ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
    "torque_y": ([0.0, 0.0, 0.0], [0.0, 5.0, 0.0]),
    "torque_z": ([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]),
}
```

The 20% gate checks signal routing and sign, not force-estimation accuracy; exact static equilibrium values depend on the whole-body controller and contact constraints.

- [ ] **Step 4: Run local tests and the real Isaac Lab smoke**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_wrench_probe_static.py \
  tests/test_m1_asset_static.py \
  tests/test_m1_smoke_cfg_static.py

cd /home/xk/coding/M1
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_panda_wrench_probe.py \
  --headless \
  --output Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl
```

Expected: pytest passes; real probe exits `0`, produces seven finite JSONL rows, reports one 25-DOF articulation, and passes all six channel checks.

- [ ] **Step 5: Run offline denial verification**

Disconnect or block Nucleus/network access for this process, then run:

```bash
cd /home/xk/coding/M1
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --headless
```

Expected: exit `0` with `remote_dependencies: []`. If the environment cannot enforce network denial, record dependency-closure verification as passed and network-denial execution as unverified; do not claim full offline runtime proof.

- [ ] **Step 6: Align notes and close this foundation phase**

Update the T400 branch and log with:

- exact commands and exit codes;
- generated asset checksums;
- dependency count and remote dependency count;
- body/joint names and 25-DOF result;
- six probe rows and channel pass/fail;
- whether network denial was actually exercised;
- `Git Ref: unavailable` unless the user initializes a repository before execution.

Mark only the asset/wrench foundation child complete. Leave residual policy, Teacher–Student training, IK/OSC, grasping, sensor driver, mechanical validation, and real-hardware validation open.

## Plan Self-Review

- Spec coverage in this phase: local asset closure, single articulation, fixed mount, 25-DOF topology, unchanged M1 hybrid action scope, base-frame six-dimensional total wrench, deterministic tests, and offline verification are covered.
- Deliberately deferred: residual controller, Teacher/Student networks and losses, domain randomization, Panda IK/OSC, object curriculum, real sensor driver, safety state machine, and hardware tests. Each depends on the verified asset/wrench interface produced here.
- Type consistency: the wrench helper always returns `torch.Tensor[num_envs, 6]`; body names and wrench order are identical across asset config, environment, probe, and tests.
- Repository state: commit steps are replaced by checksum/log checkpoints because `/home/xk/coding/M1` has no `.git` directory.
