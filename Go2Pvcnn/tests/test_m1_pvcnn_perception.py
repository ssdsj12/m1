from pathlib import Path
import os

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_configure_pvcnn_cuda_uses_workspace_toolchain(tmp_path) -> None:
    from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

    cuda_root = tmp_path / ".cuda-nvcc-12.8"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "bin/nvcc").touch()
    (cuda_root / "targets/x86_64-linux/include").mkdir(parents=True)
    (cuda_root / "targets/x86_64-linux/lib").mkdir(parents=True)
    environ = {"PATH": "/usr/bin"}

    configured = configure_pvcnn_cuda(tmp_path, environ=environ)

    assert configured
    assert environ["CUDA_HOME"] == str(cuda_root)
    assert environ["CUDACXX"] == str(cuda_root / "bin/nvcc")
    assert environ["PATH"].startswith(f"{cuda_root}/bin{os.pathsep}")
    assert environ["CPATH"] == str(cuda_root / "targets/x86_64-linux/include")
    assert environ["LIBRARY_PATH"] == str(cuda_root / "targets/x86_64-linux/lib")
    assert environ["LD_LIBRARY_PATH"].startswith(str(cuda_root / "targets/x86_64-linux/lib"))
    assert environ["TORCH_CUDA_ARCH_LIST"] == "12.0"


def test_configure_pvcnn_cuda_preserves_explicit_cuda_home(tmp_path) -> None:
    from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

    environ = {"CUDA_HOME": "/opt/cuda"}

    assert not configure_pvcnn_cuda(tmp_path, environ=environ)
    assert environ == {"CUDA_HOME": "/opt/cuda"}


def test_grid_maps_to_point_cloud_preserves_grid_and_height() -> None:
    from go2_pvcnn.tasks.m1_pvcnn_perception import grid_elevation_to_point_cloud

    elevation = torch.tensor([[[0.0, 0.1], [0.2, 0.3]]])
    points = grid_elevation_to_point_cloud(elevation, x_size=2.0, y_size=2.0)

    assert points.shape == (1, 4, 3)
    torch.testing.assert_close(points[0, :, 2], -elevation.flatten())
    torch.testing.assert_close(points[0, :, 0].unique(), torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(points[0, :, 1].unique(), torch.tensor([-1.0, 1.0]))


def test_logits_to_semantic_channel_matches_class_expectation() -> None:
    from go2_pvcnn.tasks.m1_pvcnn_perception import logits_to_semantic_channel

    logits = torch.tensor(
        [[[10.0, 0.0], [0.0, 0.0], [0.0, 10.0]]], dtype=torch.float32
    )
    semantic = logits_to_semantic_channel(logits, height=1, width=2)

    assert semantic.shape == (1, 1, 2)
    torch.testing.assert_close(semantic[0, 0], torch.tensor([0.0, 2.0]), atol=2.0e-4, rtol=0.0)


def test_m1_pvcnn_wrapper_keeps_actor_and_critic_dimensions_compatible() -> None:
    source = (ROOT / "go2_pvcnn/tasks/m1_pvcnn_perception.py").read_text()

    assert "class M1PvcnnRslRlEnvWrapper(M1RslRlEnvWrapper)" in source
    assert 'obs_dict["policy_elevation_semantic_map"]' in source
    assert "self.pvcnn_model(point_cloud.transpose(1, 2).contiguous())" in source
    assert "last_point_cloud" in source
    assert "last_semantic_labels" in source
    assert '"critic": critic_obs' in source


def test_m1_pvcnn_train_and_pretrain_entrypoints_exist() -> None:
    pretrain = (ROOT / "scripts/m1_pvcnn_pretrain.py").read_text()
    train = (ROOT / "scripts/m1_pvcnn_train.py").read_text()

    assert "PVCNN(num_classes=3" in pretrain
    assert "cross_entropy" in pretrain
    assert "semantic_accuracy" in pretrain
    assert 'parser.add_argument("--seed"' in pretrain
    assert 'parser.add_argument("--eval-steps"' in pretrain
    assert "model.eval()" in pretrain
    assert "M1PvcnnRslRlEnvWrapper" in train
    assert "pvcnn_state_dict" in train
    assert "enable_pvcnn_sync_training" in train
    assert '"width_multiplier": width_multiplier' in train
    assert '"num_classes": 3' in train
    assert 'parser.add_argument("--leg-noise-std"' in train
    assert "runner.alg.actor_critic.std[:12].fill_(args.leg_noise_std)" in train
    assert 'parser.add_argument("--freeze-leg-noise-std"' in train
    assert "def _freeze_leg_std_gradient" in train
    assert "gradient[:12] = 0.0" in train
    assert "runner.alg.actor_critic.std.register_hook(_freeze_leg_std_gradient)" in train


def test_m1_wave_distillation_trains_only_leg_output_rows() -> None:
    distill = (ROOT / "scripts/m1_wave_distill.py").read_text()

    assert "m1_wave_reference_actions" in distill
    assert "build_teacher_student_residual" in distill
    assert "executed_actions[:, :12] = build_teacher_student_residual(" in distill
    assert "prediction = actor_critic.actor(observations_for_target)[:, :12]" in distill
    assert "output_layer.weight.grad[12:] = 0.0" in distill
    assert "output_layer.bias.grad[12:] = 0.0" in distill
    assert "pvcnn_state_dict" in distill


def test_checkpoint_eval_can_use_pvcnn_perception() -> None:
    source = (ROOT / "scripts/m1_checkpoint_eval.py").read_text()

    assert 'parser.add_argument("--perception-checkpoint"' in source
    assert "M1PvcnnRslRlEnvWrapper" in source
    assert 'perception["pvcnn_state_dict"]' in source
    assert '"perception_checkpoint"' in source


def test_m1_play_can_use_pvcnn_perception_checkpoint() -> None:
    source = (ROOT / "scripts/m1_play.py").read_text()

    assert 'parser.add_argument("--perception-checkpoint"' in source
    assert "PVCNN_ROOT" in source
    assert "M1PvcnnRslRlEnvWrapper" in source
    assert "PVCNN(" in source
    assert 'perception["pvcnn_state_dict"]' in source
    assert "args.perception_checkpoint and not args.checkpoint" in source
    assert "runner.alg.pvcnn_model = pvcnn_model" in source
