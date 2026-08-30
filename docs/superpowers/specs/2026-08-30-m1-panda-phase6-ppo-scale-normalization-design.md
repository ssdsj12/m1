# M1 + Panda Phase 6 PPO Scale and Normalization Design

## 1. Context and authoritative evidence

The aligned Phase 6 v2 path completed the exact 100-update GPU0 short run and
all 24 independent fixed-condition workers.  The authoritative promotion
manifest was `status=rejected` and `accepted=false`, so the 3000-update long
run correctly did not start.  All five candidates remained hard-safe, but none
produced a tolerance-decisive stability or end-effector improvement.

The v2 TensorBoard and checkpoint evidence identifies a training-scale defect:

- PPO aborted remaining mini-batches in 98 of 100 updates.  Most updates
  completed only one or two of the configured eight mini-batches.
- KL reached `3.175`, while the configured abort threshold was `0.015`.
- value loss stayed near `9e4--3e5` and the reported pre-clipping gradient norm
  reached `5.57e4`.
- the custom residual reward returned approximately five reward units at every
  200 Hz control step, yielding episode return near 20,000.  Unlike IsaacLab's
  `RewardManager`, it did not multiply the per-second reward density by the
  control interval `dt=0.005`.
- the policy used raw 103D observations with
  `empirical_normalization=false`.  A real zero-residual rollout measured group
  maxima of approximately `4.0` (M1), `3.23` (arm), `15.88` (wrench), and
  `1.62` (context).
- even after 100 updates, the deterministic normalized residual RMS on real
  observations was only about `0.00153`; nevertheless its accumulated KL from
  the exact-zero policy was about `0.367` because exploration standard
  deviation was only `0.005`.

These facts explain why the safety-first short run did not learn a policy large
and consistent enough to cross the unchanged fixed-condition promotion
tolerances.  This design corrects time scaling, observation scaling and the
initial exploration scale without weakening any physical or promotion gate.

## 2. Selected approach

Apply the following three coupled corrections:

1. Scale the complete custom residual reward by the actual control interval
   before returning it to PPO.
2. Enable RSL-RL empirical observation normalization and require every
   checkpoint inference path to restore and apply the saved normalizer.
3. Increase the initial physical standard deviation of the normalized 8D
   policy from `0.005` to `0.01`, while retaining the existing `[0.005, 0.02]`
   clipping bounds and unchanged KL target/abort thresholds.

This is preferred over changing only observation normalization, which would
leave the 200 Hz return scale incorrect, or changing only KL/value-loss
hyperparameters, which would mask the unit defect.  Promotion tolerances are
not adjusted: Phase 6 must still demonstrate a genuine improvement.

## 3. Reward time-scale contract

`compute_residual_reward(signals)` continues to compute the same dimensionless
reward density and the same named components:

- stability;
- gated task tracking;
- normalized bounded wrench and slip tracking;
- residual magnitude/rate/intervention regularization.

No coefficient, gate, wrench scale, trajectory bound or ordering changes.
`M1PandaArmMpcResidualRuntime` resolves once at construction:

```text
control_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
```

The value must be finite and positive.  After the post-physics transition has
been finalized, the runtime returns:

```text
ppo_reward = compute_residual_reward(signals).total * control_dt
```

Raw physical diagnostics, normalized wrench diagnostics, saturation counts,
hard failures and promotion metrics are not multiplied by `control_dt`.  This
uniform scaling preserves the reward optimum and all relative weights while
bringing critic targets onto the standard IsaacLab per-control-interval scale.

## 4. Observation and inference normalization contract

The residual PPO configuration sets `empirical_normalization=true`.  During
training, the existing RSL-RL running normalizers process actor and critic
observations before inference and value evaluation.  Candidate checkpoints
must atomically contain:

- `model_state_dict`;
- `optimizer_state_dict`;
- actor observation normalizer state;
- critic observation normalizer state;
- existing runner metadata.  The adjacent atomic run manifest remains the
  authority for completed-update count and diagnostics.

Every policy consumer uses one canonical normalized inference interface from
`OnPolicyRunner.get_inference_policy(device=...)`.  This applies to:

- fixed-condition candidate evaluation;
- GUI/headless play;
- any promoted checkpoint initialization or validation path;
- the 3000-update long lineage.

Consumers must not call `runner.alg.actor_critic.act_inference(raw_obs)` when
empirical normalization is enabled.  Loading a normalized checkpoint without
both normalizer state dictionaries fails closed.  Inference switches the
normalizers to evaluation mode so evaluation observations cannot mutate the
saved statistics.

The pre-update `candidate_u000.pt` remains an exact zero-mean actor.  Its zeroed
final actor layer makes the public residual exactly zero regardless of the
initial normalizer statistics.

## 5. Exploration and PPO contract

The initial normalized-action standard deviation becomes `0.01`.  Existing
settings remain unchanged unless a separate evidence-driven design is
approved:

- `clip_min_std=0.005`;
- `clip_max_std=0.02`;
- `desired_kl=0.01`;
- `kl_abort_threshold=0.015`;
- adaptive learning-rate bounds;
- rollout length 256, two epochs and four mini-batches;
- `gamma=0.9995`, `lam=0.995` and all physical action limits.

Doubling the standard deviation reduces KL caused by the same policy-mean
change by approximately four while keeping exploration far inside the existing
physical residual and safety projection limits.  No KL threshold is relaxed.

## 6. Diagnostic pilot gate

Before spending a full short/promotion cycle, run a separate GPU0 10-update
`pilot` stage from an empty directory.  `pilot` is a distinct entrypoint stage,
not a shortened `short` stage: it executes exactly ten updates, does not write
`candidate_u*.pt`, sets `accepted=false` and `promotion_required=false`, and
cannot authorize promotion or long training.  Its atomic run manifest records
all ten optimizer summaries plus a `pilot_accepted` decision.  It passes only
if:

- all optimizer, reward and environment diagnostics are finite;
- hard-failure count is zero;
- MPC feasible rate is at least `0.99`;
- QP feasible rate and four-wheel-contact rate are exactly `1.0`;
- all residual saturation fractions remain below `0.01`;
- at most three of ten updates report KL abort;
- median completed mini-batches is at least six of eight;
- median value loss is below `100`;
- effective action standard deviation remains in `[0.005, 0.02]`.

If the pilot fails, do not launch the official 100-update run.  Record the
exact diagnostics and return to root-cause analysis; do not loosen the pilot or
promotion gates in place.

The official short entrypoint requires the accepted pilot manifest and verifies
matching asset, config, reward-runtime bundle and pilot-schema hashes before it
creates a run directory.  This prevents a manual command from silently
bypassing the diagnostic gate.

## 7. Fresh v3 promotion and conditional long

After the pilot passes:

1. Run a fresh exact 100-update short on GPU0 with eight environments.
2. Save exactly `candidate_u000/u025/u050/u075/u100.pt` by completed-update
   count.
3. Require the short manifest to be `status=safe_complete`, 100/100 complete,
   `accepted=false`, and `promotion_required=true`.
4. Run nine zero-vs-zero calibration workers and fifteen candidate workers in
   fresh Isaac Sim processes for seeds `42/43/44`, 4000 steps each.
5. Preserve the existing engineering floors, noise-calibration equation,
   stability-first rank order, hard gates, non-regression checks and
   tie-breaking.
6. Do not reuse any v1/v2 checkpoint, worker JSON, tolerance file or promotion
   manifest because the reward/config and normalizer lineage have changed.
7. Start the 3000-update long run only if the new atomic promotion manifest has
   `accepted=true` and its asset, config, reward, short-manifest, checkpoint and
   normalizer lineage hashes all validate.

The short, worker, promotion and long manifests also record a
`runtime_sha256` for the residual wrapper and a `reward_runtime_bundle_sha256`
derived deterministically from the reward module and residual wrapper hashes.
Promotion workers and long startup recompute both values.  This makes the
post-physics ordering and `dt` scaling part of the fail-closed lineage instead
of relying only on the reward-module path.

A rejected v3 manifest leaves long unstarted and triggers another documented
root-cause iteration.  Starting long does not itself prove final acceptance;
the process must be monitored through its declared safety/finite/checkpoint
completion contract.

## 8. Error handling and atomicity

- Reject a non-finite or non-positive resolved control interval.
- Reject missing, non-finite or incompatible normalizer state on checkpoint
  load.
- Never silently replace normalized inference with raw actor inference.
- Preserve existing atomic checkpoint and JSON publication.
- Preserve fail-closed behavior for process failure, missing worker output,
  SHA mismatch, non-finite metrics and any hard-gate regression.
- Never overwrite v2 artifacts; use fresh pilot, short-v3 and long-v3 paths.
- Never launch long from a diagnostic pilot, safe short alone, or rejected
  promotion manifest.
- Reject a pilot, worker, promotion or long lineage whose runtime or
  reward-runtime bundle SHA does not match current sources.

## 9. Verification strategy

Use test-driven development before production edits:

1. Pure/runtime tests prove the PPO reward is exactly the existing total times
   the resolved control interval and diagnostics remain unscaled.
2. Configuration tests prove empirical normalization is enabled and initial
   standard deviation is exactly `0.01` with unchanged bounds/KL settings.
3. Runner/checkpoint tests prove normalizer states are saved and required on
   load.
4. Pilot tests prove exact ten-update execution, no candidate publication,
   atomic diagnostic aggregation, strict thresholds and short-start lineage
   validation.
5. Evaluation and play tests prove they call the canonical normalized inference
   policy and cannot bypass it.
6. Manifest tests prove runtime and reward-runtime bundle hashes are recorded
   and revalidated at every promotion/long boundary.
7. Existing tests continue to prove observation `(103,)`, public action `(8,)`,
   private effort `(23,)`, exact-zero u000, post-physics reward ordering and
   rejected-promotion long blocking.
8. Run focused and expanded CPU regression, compile checks and
   `git diff --check`.
9. Re-run the unchanged Phase 5 GPU0 4000-step regression before the pilot.
10. Run the 10-update pilot and validate its TensorBoard/manifest diagnostics.
11. Only after pilot acceptance run the official v3 short, all 24 workers, and
   the conditional long branch.

## 10. Frozen and non-goal boundaries

This revision does not change:

- the M1+Panda USD or zero-clearance mount;
- WBC, QP, Arm MPC, force sensor calibration or RNE dynamics;
- 200 Hz physics/WBC or 50 Hz Arm MPC rates;
- observation/action/effort dimensions or action channel order;
- reward relative weights, gates, wrench normalization or EE trajectory;
- physical/slew limits, safety projection or termination thresholds;
- Phase 5 gates, Phase 6 calibration floors, promotion tolerances, rank order,
  seeds or 4000-step evaluation length;
- the exact 100-update candidate schedule or conditional 3000-update rule.

It also does not authorize external-wrench curriculum, terrain, grasping,
Student distillation or real-hardware deployment.  Those remain later phases.
