# M1 + Panda Residual Action Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable stateful PyTorch composer that safely adds a normalized 12-leg/4-wheel residual to an existing M1 base action.

**Architecture:** A frozen configuration defines M1 action scales and physical residual/slew limits. A pure PyTorch class owns one detached physical-residual history row per environment, validates calls atomically, exposes cloned diagnostics, and returns the base action plus the bounded residual converted back to the existing normalized M1 action space.

**Tech Stack:** Python 3, dataclasses, PyTorch, pytest.

## Global Constraints

- Work only under `/home/xk/coding/M1`; active code lives under `Go2Pvcnn/`.
- Keep this phase independent of Isaac Sim, Isaac Lab, RSL-RL, checkpoints, Teacher/Student networks, Panda IK/OSC and grasping.
- Input and output action order is exactly 12 M1 leg-position channels followed by four M1 wheel-velocity channels.
- Network residual input is normalized and clipped to `[-1, 1]`; physical units are `rad` for legs and `rad/s` for wheels.
- Defaults are `0.25` leg action scale, `8.0` wheel action scale, `0.05 rad` leg residual limit, `1.0 rad/s` wheel residual limit, `0.01 rad/step` leg slew limit and `0.2 rad/s/step` wheel slew limit.
- The composer limits only the residual and does not clip or rewrite `base_action`.
- Every environment owns independent history; full and selective reset clear history and diagnostics.
- Validate a complete call before mutating state; invalid calls must be atomic.
- Preserve current-step gradients and detach history stored across control steps.
- Diagnostic properties return clones rather than mutable internal tensors.
- Use TDD for every production-code change: observe RED before GREEN.
- `/home/xk/coding/M1` is not a Git worktree. Do not invent commits; record every commit checkpoint as `Git Ref: unavailable`.
- Keep repository-relative links in all notes.

---

## File Structure

- Create `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py`: frozen configuration, stateful composer, validation, composition, reset and cloned diagnostic properties.
- Create `Go2Pvcnn/tests/test_m1_residual_action.py`: CPU-only behavior, validation, reset, state isolation and gradient tests.
- Modify `notes/todo/T400-m1-panda-force-aware-teacher-student.md`: track Tasks 1–4 and final state.
- Modify `notes/todo.md`: keep the T400 dashboard/open leaf current.
- Create `notes/log/2026-08-14-m1-panda-residual-action-composer-implementation.md`: RED/GREEN and regression evidence.
- Modify `notes/log/index.md`: index the implementation evidence.

### Task 1: Define configuration, constructor state and cloned diagnostics

**Files:**
- Create: `Go2Pvcnn/tests/test_m1_residual_action.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py`

**Interfaces:**
- Consumes: `torch.device | str`, `torch.dtype`, positive `num_envs` and the six numeric defaults from the approved design.
- Produces: `M1ResidualActionComposerCfg`; `M1ResidualActionComposer(cfg, num_envs, device, dtype=torch.float32)`; properties `physical_residual`, `amplitude_clipped`, `slew_clipped`, each shaped `[num_envs, 16]` and returned as a clone.

- [ ] **Step 1: Write failing configuration/state tests**

Create `Go2Pvcnn/tests/test_m1_residual_action.py` with:

```python
import pytest
import torch

from go2_pvcnn.tasks.m1_residual_action import (
    M1ResidualActionComposer,
    M1ResidualActionComposerCfg,
)


def test_default_configuration_and_state_contract():
    cfg = M1ResidualActionComposerCfg()
    composer = M1ResidualActionComposer(cfg, num_envs=3, device="cpu")

    assert cfg.leg_action_scale == pytest.approx(0.25)
    assert cfg.wheel_action_scale == pytest.approx(8.0)
    assert cfg.leg_residual_limit_rad == pytest.approx(0.05)
    assert cfg.wheel_residual_limit_rad_s == pytest.approx(1.0)
    assert cfg.leg_slew_limit_rad_per_step == pytest.approx(0.01)
    assert cfg.wheel_slew_limit_rad_s_per_step == pytest.approx(0.2)
    assert composer.physical_residual.shape == (3, 16)
    assert composer.physical_residual.dtype == torch.float32
    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


def test_diagnostic_properties_are_clones():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )

    leaked_physical = composer.physical_residual
    leaked_amplitude = composer.amplitude_clipped
    leaked_slew = composer.slew_clipped
    leaked_physical.fill_(7.0)
    leaked_amplitude.fill_(True)
    leaked_slew.fill_(True)

    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"leg_action_scale": 0.0}, "leg_action_scale must be finite and > 0"),
        ({"wheel_action_scale": -1.0}, "wheel_action_scale must be finite and > 0"),
        ({"leg_residual_limit_rad": -0.1}, "leg_residual_limit_rad must be finite and >= 0"),
        ({"wheel_residual_limit_rad_s": float("inf")}, "wheel_residual_limit_rad_s must be finite and >= 0"),
        ({"leg_slew_limit_rad_per_step": float("nan")}, "leg_slew_limit_rad_per_step must be finite and >= 0"),
        ({"wheel_slew_limit_rad_s_per_step": -0.1}, "wheel_slew_limit_rad_s_per_step must be finite and >= 0"),
    ],
)
def test_configuration_rejects_invalid_numbers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M1ResidualActionComposerCfg(**kwargs)


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_num_envs(num_envs):
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        M1ResidualActionComposer(
            M1ResidualActionComposerCfg(), num_envs=num_envs, device="cpu"
        )


def test_constructor_rejects_non_floating_dtype():
    with pytest.raises(TypeError, match="dtype must be a floating torch.dtype"):
        M1ResidualActionComposer(
            M1ResidualActionComposerCfg(), num_envs=1, device="cpu", dtype=torch.int64
        )
```

- [ ] **Step 2: Run Task 1 tests to observe RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q tests/test_m1_residual_action.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'go2_pvcnn.tasks.m1_residual_action'`.

- [ ] **Step 3: Implement configuration, state and diagnostic properties**

Create `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py` with:

```python
"""Stateful bounded residual composition for M1 hybrid actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


M1_ACTION_DIM = 16
M1_LEG_ACTION_DIM = 12


def _require_finite_bound(name: str, value: float, *, positive: bool) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if positive:
        valid = valid and value > 0
        condition = "> 0"
    else:
        valid = valid and value >= 0
        condition = ">= 0"
    if not valid:
        raise ValueError(f"{name} must be finite and {condition}")


@dataclass(frozen=True)
class M1ResidualActionComposerCfg:
    """Physical limits and existing M1 action scales."""

    leg_action_scale: float = 0.25
    wheel_action_scale: float = 8.0
    leg_residual_limit_rad: float = 0.05
    wheel_residual_limit_rad_s: float = 1.0
    leg_slew_limit_rad_per_step: float = 0.01
    wheel_slew_limit_rad_s_per_step: float = 0.2

    def __post_init__(self) -> None:
        _require_finite_bound("leg_action_scale", self.leg_action_scale, positive=True)
        _require_finite_bound("wheel_action_scale", self.wheel_action_scale, positive=True)
        _require_finite_bound("leg_residual_limit_rad", self.leg_residual_limit_rad, positive=False)
        _require_finite_bound("wheel_residual_limit_rad_s", self.wheel_residual_limit_rad_s, positive=False)
        _require_finite_bound(
            "leg_slew_limit_rad_per_step", self.leg_slew_limit_rad_per_step, positive=False
        )
        _require_finite_bound(
            "wheel_slew_limit_rad_s_per_step",
            self.wheel_slew_limit_rad_s_per_step,
            positive=False,
        )


class M1ResidualActionComposer:
    """Compose bounded physical residuals with normalized M1 base actions."""

    def __init__(
        self,
        cfg: M1ResidualActionComposerCfg,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating torch.dtype")

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (num_envs, M1_ACTION_DIM)
        self._physical_residual = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._amplitude_clipped = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self._slew_clipped = torch.zeros(shape, device=self.device, dtype=torch.bool)

    @property
    def physical_residual(self) -> torch.Tensor:
        return self._physical_residual.clone()

    @property
    def amplitude_clipped(self) -> torch.Tensor:
        return self._amplitude_clipped.clone()

    @property
    def slew_clipped(self) -> torch.Tensor:
        return self._slew_clipped.clone()
```

- [ ] **Step 4: Run Task 1 tests to observe GREEN**

Run the Task 1 command again.

Expected: all tests currently present in `test_m1_residual_action.py` pass.

- [ ] **Step 5: Record the unavailable commit checkpoint**

Run:

```bash
git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree
```

Expected: exit nonzero with `fatal: 不是 git 仓库` (or the locale-equivalent message). Record `Task 1 Git Ref: unavailable`; do not initialize a repository.

### Task 2: Compose bounded residuals with gradients and diagnostics

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_residual_action.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py`

**Interfaces:**
- Consumes: `compose(base_action: torch.Tensor, normalized_residual: torch.Tensor)` with same-device, same-dtype tensors shaped `[num_envs, 16]`.
- Produces: combined action tensor `[num_envs, 16]`; updated detached `physical_residual`; Boolean `amplitude_clipped` and `slew_clipped` diagnostics.

- [ ] **Step 1: Add failing mapping, clipping, slew and gradient tests**

Append to `Go2Pvcnn/tests/test_m1_residual_action.py`:

```python
def test_zero_residual_preserves_base_action_after_initialization():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    base = torch.randn(2, 16)

    result = composer.compose(base, torch.zeros_like(base))

    assert torch.equal(result, base)


def test_maps_leg_and_wheel_physical_residuals_to_action_space():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    base = torch.zeros(1, 16)
    residual = torch.full((1, 16), 0.5)

    result = composer.compose(base, residual)

    assert torch.allclose(result[:, :12], torch.full((1, 12), 0.1))
    assert torch.allclose(result[:, 12:], torch.full((1, 4), 0.0625))
    assert torch.allclose(composer.physical_residual[:, :12], torch.full((1, 12), 0.025))
    assert torch.allclose(composer.physical_residual[:, 12:], torch.full((1, 4), 0.5))


def test_clips_normalized_amplitude_and_reports_mask():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    residual = torch.tensor([[2.0] + [0.0] * 11 + [-3.0] + [0.0] * 3])

    composer.compose(torch.zeros_like(residual), residual)

    assert composer.physical_residual[0, 0] == pytest.approx(0.05)
    assert composer.physical_residual[0, 12] == pytest.approx(-1.0)
    assert composer.amplitude_clipped[0, 0]
    assert composer.amplitude_clipped[0, 12]
    assert composer.amplitude_clipped.sum().item() == 2


def test_slew_limits_positive_and_negative_transitions():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    base = torch.zeros(1, 16)

    first = composer.compose(base, torch.ones_like(base))
    second = composer.compose(base, -torch.ones_like(base))

    assert torch.allclose(first[:, :12], torch.full((1, 12), 0.04))
    assert torch.allclose(first[:, 12:], torch.full((1, 4), 0.025))
    assert torch.allclose(second, torch.zeros_like(second))
    assert composer.slew_clipped.all()


def test_current_step_keeps_gradients_and_history_is_detached():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    base = torch.zeros(1, 16, requires_grad=True)
    residual = torch.full((1, 16), 0.5, requires_grad=True)

    result = composer.compose(base, residual)
    result.sum().backward()

    assert torch.equal(base.grad, torch.ones_like(base))
    assert torch.allclose(residual.grad[:, :12], torch.full((1, 12), 0.2))
    assert torch.allclose(residual.grad[:, 12:], torch.full((1, 4), 0.125))
    assert not composer.physical_residual.requires_grad
```

- [ ] **Step 2: Run the new tests to observe RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q tests/test_m1_residual_action.py -k "zero_residual or maps_leg or clips_normalized or slew_limits or keeps_gradients"
```

Expected: failures with `AttributeError: 'M1ResidualActionComposer' object has no attribute 'compose'`.

- [ ] **Step 3: Implement physical limit tensors and `compose()`**

Add this constructor state after the diagnostic tensors:

```python
        self._physical_limit = torch.tensor(
            [cfg.leg_residual_limit_rad] * M1_LEG_ACTION_DIM
            + [cfg.wheel_residual_limit_rad_s] * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        self._slew_limit = torch.tensor(
            [cfg.leg_slew_limit_rad_per_step] * M1_LEG_ACTION_DIM
            + [cfg.wheel_slew_limit_rad_s_per_step] * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        self._action_scale = torch.tensor(
            [cfg.leg_action_scale] * M1_LEG_ACTION_DIM
            + [cfg.wheel_action_scale] * (M1_ACTION_DIM - M1_LEG_ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
```

Add these methods inside `M1ResidualActionComposer`:

```python
    def _validate_action_tensor(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_shape = (self.num_envs, M1_ACTION_DIM)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on device {self.device}, got {value.device}")
        if value.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    def compose(
        self, base_action: torch.Tensor, normalized_residual: torch.Tensor
    ) -> torch.Tensor:
        """Return the base action plus an amplitude- and slew-limited residual."""
        self._validate_action_tensor("base_action", base_action)
        self._validate_action_tensor("normalized_residual", normalized_residual)

        amplitude_clipped = normalized_residual.abs() > 1.0
        clipped_normalized = normalized_residual.clamp(-1.0, 1.0)
        target_physical = clipped_normalized * self._physical_limit
        requested_delta = target_physical - self._physical_residual
        limited_delta = torch.maximum(
            torch.minimum(requested_delta, self._slew_limit), -self._slew_limit
        )
        physical_residual = self._physical_residual + limited_delta
        slew_clipped = requested_delta.abs() > self._slew_limit
        combined_action = base_action + physical_residual / self._action_scale

        self._physical_residual = physical_residual.detach().clone()
        self._amplitude_clipped = amplitude_clipped.detach().clone()
        self._slew_clipped = slew_clipped.detach().clone()
        return combined_action
```

- [ ] **Step 4: Run the complete focused test file to observe GREEN**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q tests/test_m1_residual_action.py
```

Expected: every Task 1 and Task 2 test passes.

- [ ] **Step 5: Record the unavailable commit checkpoint**

Run `git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree` and record `Task 2 Git Ref: unavailable`; do not initialize Git.

### Task 3: Enforce atomic validation and selective reset

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_residual_action.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py`

**Interfaces:**
- Consumes: `reset(env_ids: torch.Tensor | Sequence[int] | None = None)` where tensor/sequence indices are integers and in `[0, num_envs)`.
- Produces: full or selective state clearing; deterministic errors for invalid tensors and reset indices; no state mutation on failure.

- [ ] **Step 1: Add failing reset and atomic-failure tests**

Append to `Go2Pvcnn/tests/test_m1_residual_action.py`:

```python
def _snapshot(composer):
    return (
        composer.physical_residual,
        composer.amplitude_clipped,
        composer.slew_clipped,
    )


def test_selective_reset_clears_only_requested_environments():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=3, device="cpu"
    )
    action = torch.ones(3, 16)
    composer.compose(torch.zeros_like(action), action)

    composer.reset(torch.tensor([1], dtype=torch.int64))

    assert composer.physical_residual[0].abs().sum() > 0
    assert not composer.physical_residual[1].any()
    assert composer.physical_residual[2].abs().sum() > 0
    assert not composer.amplitude_clipped[1].any()
    assert not composer.slew_clipped[1].any()


def test_full_reset_clears_all_state_and_diagnostics():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    action = torch.full((2, 16), 2.0)
    composer.compose(torch.zeros_like(action), action)

    composer.reset()

    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


@pytest.mark.parametrize(
    ("env_ids", "error", "message"),
    [
        (torch.tensor([True, False]), TypeError, "env_ids must contain integers"),
        (torch.tensor([[0]], dtype=torch.int64), ValueError, "env_ids must be one-dimensional"),
        ([0, 1.5], TypeError, "env_ids must contain integers"),
        ([-1], IndexError, "env_ids contains out-of-range index -1"),
        ([2], IndexError, "env_ids contains out-of-range index 2"),
    ],
)
def test_invalid_reset_is_atomic(env_ids, error, message):
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    action = torch.ones(2, 16)
    composer.compose(torch.zeros_like(action), action)
    before = _snapshot(composer)

    with pytest.raises(error, match=message):
        composer.reset(env_ids)

    after = _snapshot(composer)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


@pytest.mark.parametrize(
    ("base", "residual", "error", "message"),
    [
        (torch.zeros(1, 15), torch.zeros(1, 16), ValueError, "base_action must have shape"),
        (torch.zeros(1, 16), torch.zeros(1, 15), ValueError, "normalized_residual must have shape"),
        (torch.zeros(1, 16, dtype=torch.float64), torch.zeros(1, 16), TypeError, "base_action must have dtype"),
        (torch.zeros(1, 16), torch.zeros(1, 16, dtype=torch.float64), TypeError, "normalized_residual must have dtype"),
        (torch.zeros(1, 16, device="meta"), torch.zeros(1, 16), ValueError, "base_action must be on device"),
        (torch.zeros(1, 16), torch.zeros(1, 16, device="meta"), ValueError, "normalized_residual must be on device"),
        (torch.full((1, 16), float("nan")), torch.zeros(1, 16), ValueError, "base_action must contain only finite values"),
        (torch.zeros(1, 16), torch.full((1, 16), float("inf")), ValueError, "normalized_residual must contain only finite values"),
    ],
)
def test_invalid_compose_is_atomic(base, residual, error, message):
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    valid = torch.ones(1, 16)
    composer.compose(torch.zeros_like(valid), valid)
    before = _snapshot(composer)

    with pytest.raises(error, match=message):
        composer.compose(base, residual)

    after = _snapshot(composer)
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_non_tensor_compose_input_has_clear_error():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    with pytest.raises(TypeError, match="base_action must be a torch.Tensor"):
        composer.compose([[0.0] * 16], torch.zeros(1, 16))
```

- [ ] **Step 2: Run the reset/atomic subset to observe RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q tests/test_m1_residual_action.py -k "reset or atomic or non_tensor"
```

Expected: reset tests fail with missing `reset`; existing compose validation tests may already pass and serve as atomic regression coverage.

- [ ] **Step 3: Implement reset index normalization and atomic state clearing**

Add the import:

```python
from collections.abc import Sequence
```

Add these methods inside `M1ResidualActionComposer`:

```python
    def _normalize_env_ids(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            integer_dtypes = {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            if env_ids.dtype not in integer_dtypes:
                raise TypeError("env_ids must contain integers")
            values = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            values = list(env_ids)
        else:
            raise TypeError("env_ids must be a one-dimensional integer tensor or integer sequence")

        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise TypeError("env_ids must contain integers")
        for index in values:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"env_ids contains out-of-range index {index}")
        return torch.tensor(values, device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear all state, or only state belonging to selected environments."""
        if env_ids is None:
            self._physical_residual.zero_()
            self._amplitude_clipped.zero_()
            self._slew_clipped.zero_()
            return

        normalized_ids = self._normalize_env_ids(env_ids)
        self._physical_residual[normalized_ids] = 0
        self._amplitude_clipped[normalized_ids] = False
        self._slew_clipped[normalized_ids] = False
```

- [ ] **Step 4: Run the entire focused test file to observe GREEN**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q tests/test_m1_residual_action.py
```

Expected: all configuration, composition, gradient, reset and atomicity tests pass.

- [ ] **Step 5: Run compile validation and record the unavailable commit checkpoint**

Run:

```bash
cd /home/xk/coding/M1
python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py \
  Go2Pvcnn/tests/test_m1_residual_action.py
git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree
```

Expected: `py_compile` exits `0`; Git probe exits nonzero. Record `Task 3 Git Ref: unavailable`.

### Task 4: Run regressions and align repository memory

**Files:**
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Create: `notes/log/2026-08-14-m1-panda-residual-action-composer-implementation.md`
- Modify: `notes/log/index.md`

**Interfaces:**
- Consumes: completed composer and focused test evidence from Tasks 1–3.
- Produces: regression evidence and aligned T400 state; no runtime Isaac Sim claim.

- [ ] **Step 1: Run focused and foundation regression tests**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=. pytest -q \
  tests/test_m1_residual_action.py \
  tests/test_m1_asset_static.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_wrench_probe_static.py
```

Expected: all selected tests pass with no failures.

- [ ] **Step 2: Run final source validation**

Run:

```bash
cd /home/xk/coding/M1
python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py \
  Go2Pvcnn/tests/test_m1_residual_action.py
marker_pattern='T''BD|T''ODO|PLACE''HOLDER'
rg -n "$marker_pattern" \
  Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py \
  Go2Pvcnn/tests/test_m1_residual_action.py
```

Expected: `py_compile` exits `0`; `rg` returns no matches.

- [ ] **Step 3: Write the implementation evidence log**

Create `notes/log/2026-08-14-m1-panda-residual-action-composer-implementation.md` with these concrete headings and populate them only from observed command output:

```markdown
# M1 + Panda Residual Action Composer Implementation

## Purpose
## Stage
## Related Todo
## Files
## RED Evidence
## GREEN Evidence
## Regression Evidence
## Result
## Limitations
## Follow-up
## Git Refs
```

The `Limitations` section must state that this phase did not load a checkpoint, integrate an Isaac Lab environment, train Teacher/Student, run Isaac Sim dynamics, implement IK/OSC, or establish real-hardware safety limits. The `Git Refs` section must use exact `Git Ref: unavailable` wording.

- [ ] **Step 4: Update T400 branch and dashboards**

Make these exact state transitions:

- Mark T400.4 residual composer implementation complete in `notes/todo/T400-m1-panda-force-aware-teacher-student.md` and move it to `Closed Children Archive`.
- Keep T400.3 mechanical verification open.
- Set the next software child to Teacher random six-dimensional disturbance baseline planning; do not claim it implemented.
- Update T400 rows and the recent-log table in `notes/todo.md`.
- Add the implementation log as the newest T400 row in `notes/log/index.md`.
- Use repository-relative links throughout.

- [ ] **Step 5: Verify notes and record the final unavailable commit checkpoint**

Run:

```bash
cd /home/xk/coding/M1
test -s notes/log/2026-08-14-m1-panda-residual-action-composer-implementation.md
rg -n "T400\.4|residual-action-composer-implementation|Git Ref: unavailable" \
  notes/todo.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md \
  notes/log/index.md \
  notes/log/2026-08-14-m1-panda-residual-action-composer-implementation.md
git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree
```

Expected: note checks find the completed node, indexed log and unavailable Git ref; Git probe remains nonzero. Record `Final Git Ref: unavailable`.

## Plan Self-Review

- Spec coverage: Tasks 1–3 cover configuration defaults, action ordering, physical units, amplitude limits, per-step slew, independent state, reset, gradients, cloned diagnostics, validation and atomic failure. Task 4 covers regression and required repository memory.
- Placeholder scan: the plan contains no unresolved marker, deferred implementation phrase, generic error-handling instruction or undefined interface.
- Type consistency: constructor, `compose()`, `reset()` and the three diagnostic property names match across all tasks and the approved spec.
- Scope: no task introduces checkpoint loading, environment integration, Teacher/Student networks, Panda IK/OSC, grasping or real-hardware claims.
- Git constraint: every nominal commit point is replaced by an explicit read-only repository probe and `Git Ref: unavailable`; the plan never initializes Git.
