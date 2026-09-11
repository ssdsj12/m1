# T500 DexManipNet Offline Expert Ensemble

## Purpose

Record Task 6's offline expert implementation and CPU synthetic evidence. This stage neither
downloads data nor changes Isaac, MPC, Hand QP, runtime, task configuration, or student loading.

## Stage And Related Todo

T500.5 / Task 6; see [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- The trainer calls `verify_aggregate_manifest` before every split load, recomputing aggregate,
  audit and every declared shard SHA before it opens NPZ data. It accepts only the seven Task 5
  arrays and builds exactly the frozen 42 palm-frame geometry values; labels are `(N,20,5,3)`
  float32 future velocities.
- Source groups are checked against the verified train/validation/test assignment. Distinct fixed
  seeds, deterministic loaders, residual mixture NLL plus finite acceleration/jerk penalties,
  atomic last/best checkpoints and manifest/selected-checkpoint-SHA/seed/architecture-bound resume
  validation are implemented. Overlapping split group lists reject before shards are opened.
- Held-out intervals combine every member and component into one predictive Gaussian mixture. Exact
  10th/90th marginal mixture quantiles include component variance and member disagreement. First
  node and all 20 nodes integrated with `0.01 s` use zero baselines. Production needs both RMSE
  improvements `>=10%` and coverage in `[0.65,0.95]`; synthetic provenance is aggregate-bound and
  always remains false even when a prior smoke aggregate is supplied through the normal CLI path.

## TDD And Verification

RED before the model/CLI existed: `ModuleNotFoundError: ...expert_fingertip_prior.model` (`1`
collection error in `0.80s`). Focused GREEN:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_model.py \
  tests/test_m1_bimanual_expert_prior_train_static.py
```

Initial GREEN result: `5 passed in 0.78s`. The review hardening regression (completed-range resume
and split-overlap rejection) brings the focused result to `7 passed in 5.12s`.

Tasks 1–6 regression:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py \
  tests/test_m1_bimanual_expert_prior_model.py \
  tests/test_m1_bimanual_expert_prior_train_static.py
```

Final result: `115 passed in 7.87s`. Task 6 `--help` and `py_compile` also exited `0`.

An uncommitted temporary synthetic Task 5 shard set trained for two epochs with three members. The
finite manifest reported first-step RMSE/zero/improvement `0.4473666814/0.4682388797/0.0445759615`,
endpoint `0.0911574656/0.0955388742/0.0458599565`, NLL `300.4348144531`, and coverage `1.0`.
It records `synthetic_smoke=true` and `production_deployable=false`. This is not a real-data quality
claim; it fails numeric gates and the synthetic gate independently.

## Git Refs

- Baseline Ref: `65e2b7b55fcc14fad11808ead2d1928bb0147296`
- Candidate Ref: `0011f651f581aa3be9391993af004ac5be1f57d0` (`fix: harden fingertip expert training gates`)
- Last Feature Commit: `0011f65` (base ensemble `ab0f672`)
- Last Verified Commit: `0011f65`
- Key Files: `model.py`, `m1_dual_panda_o6_train_fingertip_expert.py`, and the two Task 6 tests.

## Follow-up

No raw production source, shard or model was created or committed. Real full-data training must
meet every gate before Task 7 consumes an ensemble; Task 7 begins with compact student tests.
