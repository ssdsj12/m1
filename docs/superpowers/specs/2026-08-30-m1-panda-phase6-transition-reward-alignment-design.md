# M1 + Panda Phase 6 Transition Reward Alignment Design

## 1. Context and confirmed defect

The approved Phase 6 fixed-condition promotion run completed its exact
100-update short training and all 24 independent promotion workers.  Every
candidate remained hard-safe, but none passed the unchanged stability-first
promotion comparison, so the conditional 3000-update long run correctly did
not start.

Code-path inspection found a temporal mismatch in the public residual
environment boundary.  `M1PandaArmMpcResidualRuntime.compute()` currently
produces both actuator effort and reward from the pre-action state.  Only after
that reward has been fixed does the wrapper call `env.step(effort)` and refresh
the physical state.  Consequently the current action immediately receives its
residual magnitude/rate penalty, while its effects on balance, end-effector
tracking, slip, contact and mount wrench are attributed to the next action.
This violates the transition-reward semantics expected by the PPO rollout.

## 2. Selected solution

Use an explicit two-phase runtime contract:

1. `compute_action(actions, physics_step)` consumes the pre-transition state,
   runs Arm MPC/WBC/QP, returns the private 23-dimensional effort, and stores an
   immutable pending transition containing the current normalized residual and
   command-side diagnostics.
2. The wrapper applies exactly that effort through `env.step(effort)`.
3. The runtime refreshes state and wrench observations from physics step
   `physics_step + 1`.
4. `compute_transition_reward()` consumes the pending action plus the refreshed
   post-transition state, computes reward and training diagnostics, then clears
   the pending transition.
5. The wrapper settles reward before selectively resetting done environments.

There may be at most one pending transition.  Calling reward finalization
without a pending action, preparing a second action before finalization, or
resetting with an unsettled transition is a runtime contract error.

## 3. Reward and state semantics

The following signals must come from the post-transition state:

- roll and pitch;
- base-height error;
- support margin and wheel-contact count;
- Panda joint margin;
- hard failure and wheel slip;
- end-effector position and orientation error;
- measured mount wrench and its normalized prediction error.

The following values remain associated with the action that caused the
transition:

- current normalized 8D residual;
- previous normalized residual used by the rate penalty;
- MPC feasibility, QP feasibility and safety intervention/fallback state.

`previous_residual` advances only after reward finalization.  Done environments
receive their terminal transition reward before their runtime state is reset.
The underlying ManagerBasedRLEnv reward remains ignored, as in the existing
dedicated residual task.

## 4. Frozen contracts

This correction does not change:

- public observation shape `(103,)` or public action shape `(8,)`;
- private articulation effort shape `(23,)`;
- physics/WBC rate `200 Hz` or Arm MPC rate `50 Hz`;
- reward equations, weights, wrench scale or Small EE trajectory;
- physical residual bounds, safety projection, WBC/QP or Phase 5 hard gates;
- exact 100-update candidate schedule;
- nine zero-policy calibration pairs plus fifteen candidate evaluations;
- seeds `42/43/44`, 4000-step evaluation, promotion tolerances or comparison;
- the rule that long training starts only from `accepted=true` promotion.

Because the runtime wrapper and reward source files participate in promotion
lineage hashes, all earlier Phase 6 candidates become diagnostic-only.  Short
training and all 24 promotion workers must be rerun from fresh output paths.

## 5. Error handling

- Action preparation validates finite `(num_envs, 8)` actions atomically.
- Reward finalization requires a successful post-step refresh.
- A physics-step exception must not publish a reward or advance residual
  history.
- Selective reset clears pending/history state only after the terminal reward
  has been returned.
- Training diagnostics count finalized transitions, not prepared actions.

## 6. Verification

TDD must first reproduce the old order with a fake environment and runtime.  A
new wrapper test must prove the exact sequence:

```text
compute_action -> env.step -> refresh -> compute_transition_reward -> reset(done)
```

Runtime tests must prove that post-step changes to tilt, wrench and
end-effector state affect the current returned reward, residual history advances
only on finalization, and invalid pending-state transitions fail atomically.

After focused and regression tests, rerun the unchanged Phase 5 GPU0 seed-42
4000-step gate.  If it passes, run a fresh exact 100-update short training on
GPU0, then all 24 fixed-condition workers.  Only an atomic promotion manifest
with `accepted=true` may initialize and launch the 3000-update long run.

## 7. Non-goals

This correction does not tune learning rate, rollout length, reward weights,
trajectory difficulty, noise tolerance, promotion thresholds or physical
controllers.  Any such change requires separate evidence and approval.
