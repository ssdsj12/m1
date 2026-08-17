# M1 + Panda Prioritized WBC Teacher C0 Runbook

## Scope

This entry runs the deterministic C0 Teacher on the single M1 + Panda articulation. It is Teacher play, not PPO training and not a Student. C0 keeps the base command stationary, expects four wheel contacts, and drives a small seeded six-dimensional Panda end-effector trajectory through prioritized motion distribution and a 200 Hz whole-body QP.

C0 does not authorize rolling C1/C2 operation, external-wrench curriculum, object grasping, Student data collection/training, or real-hardware load tests. Self-collision is disabled by the combined articulation configuration; the reported self-collision count is therefore zero by construction in C0.

## Environment

Run from `/home/xk/coding/M1/Go2Pvcnn` with `/home/xk/miniconda3/envs/go2/bin/python`. GPU validation uses `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`. The accepted machine reported NVIDIA GeForce RTX 5070 with driver `580.159.03`.

The controller performs 100 unscored settling steps, then re-centers the end-effector trajectory and begins counting requested mission steps. The default trajectory is seeded and uses per-axis amplitudes of `0.005 m` and `0.01 rad`. `--disable-target-motion` selects the stationary comparison.

## GUI play

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --device cuda:0 --steps 0 --seed 42
```

`--steps 0` runs until the application closes. The Panda trajectory is enabled by default.

## Headless smoke

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 8 --seed 42 \
  --disable-target-motion \
  --summary-json /tmp/m1_panda_wbc_static8.json
```

## Headless C0 acceptance

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 2000 --seed 42 \
  --stats-interval 500 \
  --summary-json /tmp/m1_panda_wbc_c0_motion.json
```

## Safety and diagnostics

Safety states are `TRACK`, `SCALE`, `HOLD`, `RETRACT`, and `TERMINATE`. `SCALE` reduces commanded twist. `HOLD` freezes high-level distribution and re-centers the trajectory; recovery re-centers again on the measured pose so the arm never catches up to an obsolete target. `RETRACT` moves toward the safe target at a bounded rate. `TERMINATE` latches the run stop.

Periodic output includes end-effector error, singular value, QP residuals, joint speed, effort, target/actual pose, predicted/measured twist, root height, roll/pitch, contact count, lateral slip, base activation, safety reason, motion-distribution failure, and reset cause.

The JSON summary contains the required finite/QP/tracking/singularity/orientation/slip/contact/limit/reset fields plus safety reason counts, base-activation timing, maximum arm-target step, and arm-snap count.

## Hard gates

- `max_ee_position_error_m <= 0.03`
- `min_singular_value >= 0.1`, or a recorded safe base/safety activation before crossing
- `qp_feasible_rate >= 0.999`
- `max_lateral_slip_mps <= 0.05`
- absolute roll and pitch each `<= 10 deg`
- zero joint-limit violations, self-collisions, base contacts, non-finite state, arm snaps, and unexpected resets
- `exit_reason == "steps_complete"`

The accepted seed-42 2000-step run passed every gate. Do not treat a short smoke as equivalent to this acceptance.
