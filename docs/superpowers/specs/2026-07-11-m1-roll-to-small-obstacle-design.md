# M1 Roll To Small Obstacle Design

## Goal

Produce a reproducible M1 curriculum that first yields stable forward wheel locomotion and then learns to cross low obstacles using the existing Go2Pvcnn perception pipeline.

## Stage Gates

1. `roll`: locked legs, four equal wheel velocities, 6 second flat episodes. Pass when 10 deterministic episodes have at least 95% timeout completion, positive X displacement, lateral drift below 25% of X displacement, and no body-orientation termination.
2. `wave-flat`: preserve the 16-action and 60-observation contracts, keep all wheels at the stable forward baseline, and release bounded leg-position residuals. Initialize from the accepted roll checkpoint.
3. `wave-small`: replace the plane with a curriculum of 0.02 m, 0.03 m, and 0.04 m obstacles. Add a local height scan teacher observation and reward forward progress, clearance, posture, and successful passage.
4. `pvcnn-small`: train a PVCNN observation adapter against the height-scan teacher, then fine-tune the locomotion policy with PVCNN features while retaining the teacher checkpoint as fallback.

## Components

- `m1_checkpoint_eval.py`: deterministic checkpoint evaluation and machine-readable pass/fail report.
- `m1_wave_env_cfg.py`: wheel-assisted leg policy on flat ground without changing action ordering.
- `m1_small_obstacle_env_cfg.py`: low-obstacle terrain, height scanner, curriculum, and crossing rewards.
- `run_m1_curriculum.py`: stage controller that waits for artifacts, evaluates checkpoints, resumes/retries training, and starts the next stage only after its gate passes.
- Existing `m1_train.py` and `m1_play.py` remain the training and visualization entrypoints.

## Control Contract

- Actions `0:12`: leg position residuals.
- Actions `12:16`: wheel velocity commands.
- Roll mode writes zero leg residuals and four equal wheel actions in `[-0.255, -0.245]`.
- Wave modes clamp leg residuals and keep all wheel actions equal around `-0.25`; the policy cannot create front/rear wheel mismatch.

## Failure Handling

- Failed stage evaluation never promotes a checkpoint.
- A crashed training process is recorded and restarted from the latest valid checkpoint.
- NaN metrics, missing checkpoints, negative X progress, or excessive tilt fail closed.
- Stage reports and console logs live under `Go2Pvcnn/logs/m1_curriculum/`.

## Verification

- Static contract tests cover task registration, action routing, terrain/sensor configuration, and command lines.
- Headless probes verify environment construction, finite observations, stable reset behavior, and root displacement.
- Every promoted checkpoint receives a JSON evaluation report with the stage gate decision.

