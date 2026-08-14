#!/usr/bin/env python3
"""Distill the spatial M1 wave teacher into the policy leg output rows."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PVCNN_ROOT = ROOT.parent / "pvcnn"
RSL_RL_ROOT = ROOT / "rsl_rl"
for path in (ROOT, PVCNN_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

configure_pvcnn_cuda(ROOT.parent)


def build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0"
    )
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--student-rollout-final-weight", type=float, default=0.75)
    parser.add_argument("--teacher-forcing-fraction", type=float, default=0.25)
    parser.add_argument("--smoothness-weight", type=float, default=0.20)
    parser.add_argument("--overshoot-weight", type=float, default=0.10)
    parser.add_argument("--overshoot-margin", type=float, default=0.20)
    parser.add_argument("--nonwave-weight", type=float, default=0.10)
    parser.add_argument("--wheel-preservation-weight", type=float, default=0.10)
    parser.add_argument("--hierarchical-gate", action="store_true")
    parser.add_argument("--gate-positive-weight", type=float, default=3.0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--clip-actions", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args()
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(args).app
    exit_code = 1
    try:
        import gymnasium as gym
        import torch
        import torch.nn.functional as F

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_pvcnn_perception import M1PvcnnRslRlEnvWrapper
        from go2_pvcnn.tasks.m1_curriculum import (
            expand_checkpoint_observations,
            scheduled_student_rollout_weight,
        )
        from models.s3dis.pvcnn import PVCNN
        from rsl_rl.runners import OnPolicyRunner

        perception = torch.load(args.perception_checkpoint, map_location="cpu", weights_only=False)
        width_multiplier = float(perception["width_multiplier"])
        pvcnn_model = PVCNN(
            num_classes=3,
            extra_feature_channels=0,
            width_multiplier=width_multiplier,
        )
        pvcnn_model.load_state_dict(perception["pvcnn_state_dict"])

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.wave_sequential_policy_control = not args.hierarchical_gate
        env_cfg.wave_policy_phase_observation = not args.hierarchical_gate
        env_cfg.wave_gate_from_policy_action = bool(args.hierarchical_gate)
        pvcnn_model = pvcnn_model.to(env_cfg.sim.device).eval()
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = M1PvcnnRslRlEnvWrapper(env.unwrapped, pvcnn_model, clip_actions=args.clip_actions)

        observations, _ = wrapped.get_observations()
        source_checkpoint = torch.load(
            args.policy_checkpoint, map_location="cpu", weights_only=False
        )
        source_checkpoint = expand_checkpoint_observations(
            source_checkpoint, new_observation_dim=observations.shape[1]
        )
        run_name = args.run_name or datetime.now().strftime("m1_wave_distill_%Y-%m-%d_%H-%M-%S")
        log_dir = ROOT / "logs/m1_walk" / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        initialized_checkpoint = log_dir / "initialized_phase_observation.pt"
        torch.save(source_checkpoint, initialized_checkpoint)

        runner = OnPolicyRunner(
            wrapped, get_m1_train_cfg(), log_dir=None, device=env_cfg.sim.device
        )
        runner.alg.pvcnn_model = pvcnn_model
        runner.load(str(initialized_checkpoint), load_optimizer=False)
        actor_critic = runner.alg.actor_critic
        output_layer = actor_critic.actor[-1]
        if not isinstance(output_layer, torch.nn.Linear) or output_layer.out_features != 16:
            raise RuntimeError("Expected a 16-output final actor Linear layer")
        frozen_actor = deepcopy(actor_critic.actor).eval()
        for parameter in frozen_actor.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(actor_critic.actor.parameters(), lr=args.learning_rate)

        def save_checkpoint(step: int) -> Path:
            checkpoint = dict(source_checkpoint)
            checkpoint["model_state_dict"] = actor_critic.state_dict()
            checkpoint["pvcnn_state_dict"] = pvcnn_model.state_dict()
            checkpoint["iter"] = step
            checkpoint["infos"] = {"distillation_updates": step}
            path = log_dir / f"model_distill_{step}.pt"
            torch.save(checkpoint, path)
            return path

        previous_prediction = None
        for update in range(1, args.updates + 1):
            observations_for_target = observations.detach()
            with torch.no_grad():
                executed_actions = actor_critic.act_inference(observations_for_target).clone()
                student_weight = scheduled_student_rollout_weight(
                    update=update,
                    total_updates=args.updates,
                    final_weight=args.student_rollout_final_weight,
                    teacher_forcing_fraction=args.teacher_forcing_fraction,
                )
                if args.hierarchical_gate:
                    env_cfg.wave_policy_gate_weight = student_weight
                else:
                    env_cfg.wave_sequential_policy_weight = student_weight
                observations, _, dones, _ = wrapped.step(executed_actions)
                teacher_actions = env.unwrapped.m1_wave_reference_actions.detach().clone()
                gate_target = getattr(
                    env.unwrapped,
                    "m1_wave_gate_target",
                    torch.zeros(args.num_envs, dtype=torch.bool, device=env_cfg.sim.device),
                ).detach().clone()

            all_prediction = actor_critic.actor(observations_for_target)
            prediction = all_prediction[:, :12]
            with torch.no_grad():
                baseline_prediction = frozen_actor(observations_for_target)
            if args.hierarchical_gate:
                preserved_prediction = all_prediction[:, :15]
                preserved_baseline = baseline_prediction[:, :15]
            else:
                preserved_prediction = all_prediction[:, 12:]
                preserved_baseline = baseline_prediction[:, 12:]
            wheel_preservation_loss = F.smooth_l1_loss(
                preserved_prediction, preserved_baseline
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                all_prediction[:, 15],
                gate_target.to(all_prediction.dtype),
                pos_weight=all_prediction.new_tensor(float(args.gate_positive_weight)),
            )
            active_wave = torch.linalg.vector_norm(teacher_actions, dim=1) > 1.0e-6
            if bool(active_wave.any()):
                imitation_loss = F.smooth_l1_loss(
                    prediction[active_wave], teacher_actions[active_wave]
                )
                temporal_loss = (
                    F.smooth_l1_loss(
                        prediction[active_wave], previous_prediction[active_wave]
                    )
                    if previous_prediction is not None
                    else prediction.new_zeros(())
                )
                overshoot = torch.relu(
                    torch.abs(prediction[active_wave])
                    - torch.abs(teacher_actions[active_wave])
                    - float(args.overshoot_margin)
                )
                overshoot_loss = torch.mean(overshoot.square())
            else:
                imitation_loss = prediction.sum() * 0.0
                temporal_loss = prediction.new_zeros(())
                overshoot_loss = prediction.new_zeros(())
            inactive_wave = ~active_wave
            nonwave_loss = (
                F.smooth_l1_loss(
                    prediction[inactive_wave], torch.zeros_like(prediction[inactive_wave])
                )
                if bool(inactive_wave.any())
                else prediction.new_zeros(())
            )
            if args.hierarchical_gate:
                loss = gate_loss + float(
                    args.wheel_preservation_weight
                ) * wheel_preservation_loss
            else:
                loss = (
                    imitation_loss
                    + float(args.smoothness_weight) * temporal_loss
                    + float(args.overshoot_weight) * overshoot_loss
                    + float(args.nonwave_weight) * nonwave_loss
                    + float(args.wheel_preservation_weight) * wheel_preservation_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), 10.0)
            optimizer.step()
            previous_prediction = prediction.detach()
            previous_prediction = torch.where(
                dones.unsqueeze(-1), torch.zeros_like(previous_prediction), previous_prediction
            )

            if update == 1 or update % 100 == 0:
                print(
                    f"[distill] update={update} loss={loss.item():.6f} "
                    f"imitation={imitation_loss.item():.6f} "
                    f"temporal={temporal_loss.item():.6f} "
                    f"overshoot={overshoot_loss.item():.6f} "
                    f"nonwave={nonwave_loss.item():.6f} "
                    f"wheel_keep={wheel_preservation_loss.item():.6f} "
                    f"gate={gate_loss.item():.6f} "
                    f"student_rollout={student_weight:.3f} "
                    f"wave_rate={active_wave.float().mean().item():.3f}",
                    flush=True,
                )
            if update % args.checkpoint_interval == 0:
                print(f"[distill] saved={save_checkpoint(update)}", flush=True)

        final_path = save_checkpoint(args.updates)
        torch.save(
            {
                "pvcnn_state_dict": pvcnn_model.state_dict(),
                "width_multiplier": width_multiplier,
                "num_classes": 3,
            },
            log_dir / "perception_distilled.pt",
        )
        print(f"[distill] final={final_path}", flush=True)
        env.close()
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
