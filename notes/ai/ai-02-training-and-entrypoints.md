# AI Training And Entrypoints

## Navigation

- doc role: AI stage note
- paired human doc: [../human/human-02-training-and-entrypoints.md](../human/human-02-training-and-entrypoints.md)
- previous: [ai-01-overall-pipeline.md](ai-01-overall-pipeline.md)
- next: [ai-03-environment-and-observations.md](ai-03-environment-and-observations.md)
- master index: [../index.md](../index.md)

## Purpose

Index the main repository scripts and identify which ones are active default entrypoints versus legacy / specialized branches.

## Code Graph

```mermaid
graph LR
    train["train.py\n../../Go2Pvcnn/scripts/train.py"]
    play["play.py\n../../Go2Pvcnn/scripts/play.py"]
    collision["collision test\n../../Go2Pvcnn/scripts/test_go2_pvcnn_collision.py"]
    legacy["legacy PVCNN train\n../../Go2Pvcnn/scripts/train_go2_pvcnn.py"]
    launcher["AppLauncher\nisaaclab.app.AppLauncher"]
    register["register_envs.py\n../../Go2Pvcnn/go2_pvcnn/tasks/register_envs.py"]
    gymenv["gym.make -> ManagerBasedRLEnv"]
    wrapper["SimpleRslRlEnvWrapper\n../../Go2Pvcnn/scripts/train.py"]
    runner["OnPolicyRunner import path\nrsl_rl_2_01.runners"]

    train --> launcher
    play --> launcher
    collision --> launcher
    legacy --> launcher
    train --> register
    play --> register
    legacy --> register
    register --> gymenv
    gymenv --> wrapper
    wrapper --> runner
```

## Candidate Files

- `Go2Pvcnn/scripts/train.py`
- `Go2Pvcnn/scripts/train_go2_pvcnn.py` (legacy / dedicated PVCNN path)
- `Go2Pvcnn/scripts/play.py`
- `Go2Pvcnn/scripts/test_go2_pvcnn_collision.py`
- `Go2Pvcnn/scripts/test_go2_lidar.sh`

## Inputs

- CLI args
- environment variables
- checkpoint paths
- env cfg selection

## Outputs

- configured Isaac Lab environment
- wrapped RL env
- selected run mode

## Active-vs-Legacy Split

- `train.py`: active teacher training entrypoint (`teacher_semantic`, `teacher_without_semantic`, `teacher_elevation`, `teacher_elevation_semantic_map`, `teacher_elevation_trajectory`)
- `play.py`: active playback entrypoint for the same teacher experiments
- `train_go2_pvcnn.py`: older dedicated PVCNN training path for `Go2PvcnnEnv`; keep for reference or specialized runs, but do not treat it as the default mainline

## M1 + Panda Teacher A0/A1 Entrypoints

- `Go2Pvcnn/scripts/m1_panda_teacher_train.py`: force-aware Teacher A0/A1 entrypoint. It resolves the string Gym cfg through `isaaclab_tasks.utils.parse_env_cfg`, enforces the exact 60-observation/16-action contract, writes `run_manifest.json`, and validates base/resume checkpoints before loading.
- `Go2Pvcnn/scripts/m1_panda_teacher_play.py`: dedicated read-only inference entrypoint for the same exact 60/16 Teacher contract. Disturbance and GUI are on by default; `--disable-disturbance` clears external wrench without bypassing policy/composers. A1 requires both the original frozen A0 base checkpoint and the A1 checkpoint, with stage/tensor/base-SHA validation before runner load.
- `Go2Pvcnn/scripts/m1_panda_teacher_smoke.py`: real CPU acceptance chain in the fixed order A0 initial → A0 resume → A1 initial → A1 resume. Child stdout/stderr are retained and checkpoint suffixes must advance on resume.
- `Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py`: independent fresh PPO config factory; do not reuse a dict already passed to `OnPolicyRunner`, because the runner pops class names.
- Exact formal/resume commands are in [M1 + Panda Teacher runbook](../../docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md).
- Do not route these checkpoints through generic `m1_play.py` / `M1RslRlEnvWrapper`; that path does not reconstruct A1's frozen-A0 plus second residual composer.

## M1 + Panda Folded-Load Locomotion Entrypoints

- `Go2Pvcnn/scripts/m1_panda_folded_load_train.py` trains exactly one `L0-C0` through `L2-D3` stage. L0 accepts no parent; later stages require the immediate parent's accepted manifest and verified final-checkpoint SHA. It refuses non-empty run directories and never resumes optimizer state.
- `Go2Pvcnn/scripts/m1_panda_folded_load_eval.py` evaluates one eligible `model_best.pt` with 64 environments, deterministic actor means, the fixed balanced command table, and one of seeds `42/43/44`. Only all three atomic passing reports publish `model_final.pt` and `accepted=true`.
- The wrapper exposes lossless env-ID episode records directly to the training guard; runner summaries are not used to flatten command vectors. TensorBoard includes mean/max KL, KL abort, completed minibatches, gradient norm, active-only std range, inactive action, fold/effort/limit, and hard-failure diagnostics.
- This entrypoint chain is CPU/static verified only; see [Task 7 log](../log/2026-08-25-m1-panda-folded-load-train-eval-entrypoints.md). Do not use it as a locomotion acceptance claim before GPU probe and fixed physical evaluation.
