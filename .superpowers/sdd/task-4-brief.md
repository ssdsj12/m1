### Task 4: Add the combined asset config and isolated smoke environment

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/assets/m1_panda.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`

**Interfaces:**
- Consumes: `M1_JOINT_NAMES`, `M1_LEG_JOINT_NAMES`, `M1_WHEEL_JOINT_NAMES`, and local `m1_panda.usd`.
- Produces: `M1_PANDA_CFG`, `M1_PANDA_MOUNT_BODY_NAME`, `M1_PANDA_BASE_BODY_NAME`, `M1_PANDA_DOF_COUNT`, `M1PandaSmokeEnvCfg`, and Gym id `Isaac-M1-Panda-Smoke-v0`.

- [ ] **Step 1: Write the failing static config test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_combined_cfg_and_smoke_task_keep_m1_action_contract():
    asset = (ROOT / "go2_pvcnn/assets/m1_panda.py").read_text()
    env = (ROOT / "go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py").read_text()
    registry = (ROOT / "go2_pvcnn/tasks/register_m1_envs.py").read_text()
    assert 'M1_PANDA_MOUNT_BODY_NAME = "panda_link0"' in asset
    assert 'M1_PANDA_BASE_BODY_NAME = "BASE_LINK"' in asset
    assert "M1_PANDA_DOF_COUNT = 25" in asset
    assert 'usd_path=M1_PANDA_USD_PATH' in asset
    assert "joint_names=list(M1_LEG_JOINT_NAMES)" in env
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in env
    assert 'id="Isaac-M1-Panda-Smoke-v0"' in registry
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_smoke_cfg_static.py`

Expected: FAIL because both new modules are absent.

- [ ] **Step 3: Implement `M1_PANDA_CFG`**

Start from the current `M1_CFG` rigid/articulation properties and initial M1 pose. Add Panda actuators with the installed Franka limits:

```python
M1_PANDA_USD_PATH = str(Path(__file__).resolve().parents[2] / "assets/m1_panda/m1_panda.usd")
M1_PANDA_BASE_BODY_NAME = "BASE_LINK"
M1_PANDA_MOUNT_BODY_NAME = "panda_link0"
M1_PANDA_DOF_COUNT = 25

M1_PANDA_CFG = M1_CFG.copy()
M1_PANDA_CFG.spawn.usd_path = M1_PANDA_USD_PATH
M1_PANDA_CFG.init_state.joint_pos.update({
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
})
M1_PANDA_CFG.actuators.update({
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"], effort_limit=87.0,
        velocity_limit=2.175, stiffness=80.0, damping=4.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"], effort_limit=12.0,
        velocity_limit=2.61, stiffness=80.0, damping=4.0,
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"], effort_limit=200.0,
        velocity_limit=0.2, stiffness=2000.0, damping=100.0,
    ),
})
```

- [ ] **Step 4: Implement the smoke environment and registration**

Derive `M1PandaSmokeEnvCfg` from the existing M1 smoke structure, but explicitly scope every joint observation and action to `M1_JOINT_NAMES`. This task must remain independently importable and runnable, so it does not reference the mount-wrench function introduced in Task 5.

The action block must be exactly:

```python
leg_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=list(M1_LEG_JOINT_NAMES),
    scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)},
)
wheel_vel = mdp.JointVelocityActionCfg(
    asset_name="robot", joint_names=list(M1_WHEEL_JOINT_NAMES),
    scale=8.0, use_default_offset=True, clip={".*": (-8.0, 8.0)},
)
```

Register:

```python
gym.register(
    id="Isaac-M1-Panda-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": M1PandaSmokeEnvCfg, "rsl_rl_cfg_entry_point": None},
)
```

- [ ] **Step 5: Run static GREEN**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py tests/test_m1_panda_smoke_cfg_static.py tests/test_m1_smoke_cfg_static.py
```

Expected: all tests pass and the old M1 smoke contract remains green. Record the result; Git commit is unavailable.

---

