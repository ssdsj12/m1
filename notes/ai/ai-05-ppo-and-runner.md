# AI PPO And Runner

## Navigation

- doc role: AI stage note
- paired human doc: [../human/human-05-ppo-and-runner.md](../human/human-05-ppo-and-runner.md)
- previous: [ai-04-lidar-and-pvcnn.md](ai-04-lidar-and-pvcnn.md)
- next: [ai-06-assets-paths-and-experiments.md](ai-06-assets-paths-and-experiments.md)
- master index: [../index.md](../index.md)

## Purpose

Index the PPO update loop, rollout storage, checkpoint handling, and clarify that the active teacher scripts import `rsl_rl_2_01.runners.OnPolicyRunner` even though the tracked source sits under the vendored `Go2Pvcnn/rsl_rl/` tree.

## Code Graph

```mermaid
graph LR
    train["train.py\n../../Go2Pvcnn/scripts/train.py"]
    wrapper["VecEnv wrapper\n../../Go2Pvcnn/scripts/train.py"]
    runner["runner source\n../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py"]
    storage["rollout_storage.py\n../../Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py"]
    algo["ppo.py\n../../Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py"]
    logs["run outputs\n../../logs/rsl_rl/"]

    train --> wrapper
    wrapper --> runner
    runner --> storage
    storage --> algo
    algo --> runner
    runner --> logs
```

## Candidate Files

- `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`
- `Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py`
- `Go2Pvcnn/rsl_rl/rsl_rl/storage/replay_buffer.py`

## Inputs

- observations
- rewards
- dones
- policy / critic tensor groups
- legacy PVCNN branch semantic supervision payloads

## Outputs

- updated actor/critic
- checkpoints
- metrics

## Runtime Import Note

- active scripts: `train.py` and `play.py` import `OnPolicyRunner` from `rsl_rl_2_01.runners`
- tracked implementation source for note reading still lives under `Go2Pvcnn/rsl_rl/rsl_rl/`
- optional synchronous PVCNN training is mostly relevant to the older dedicated PVCNN branch, not the default teacher mainline

## M1 + Panda Teacher Checkpoint Boundary

- `m1_panda_teacher_train.py` imports the vendored `rsl_rl.runners.OnPolicyRunner` and passes it a deep copy of the fresh Teacher config.
- Checkpoints must have manifest schema 1, stage-specific metadata, observation/action dimensions 60/16, actor/critic hidden dimensions 256/128, and matching actual state tensor shapes. Optimizer state is mandatory for normal resume.
- A1 requires a valid A0 checkpoint. Its actor is loaded in eval mode with every parameter frozen, lives only in the environment wrapper, is excluded from the trainable PPO runner, and is SHA-256 checked before and after learning.
- The local runner stores the checkpoint iteration itself; the Teacher entrypoint advances `current_learning_iteration` by one after load so resume writes the next numeric checkpoint rather than overwriting the loaded file.
- `run_manifest.json` transitions from `running` to `completed` or `failed` and records the final checkpoint plus live 60/16/nonzero-wrench runtime diagnostics.
