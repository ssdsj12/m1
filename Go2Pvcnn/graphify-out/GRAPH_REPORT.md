# Graph Report - .  (2026-08-25)

## Corpus Check
- 354 files · ~287,872 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 5149 nodes · 13039 edges · 220 communities (202 shown, 18 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 1324 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122
- Community 123
- Community 124
- Community 125
- Community 126
- Community 127
- Community 128
- Community 129
- Community 131
- Community 132
- Community 133
- Community 134
- Community 135
- Community 136
- Community 137
- Community 138
- Community 139
- Community 140
- Community 141
- Community 142
- Community 143
- Community 144
- Community 145
- Community 146
- Community 147
- Community 148
- Community 149
- Community 150
- Community 151
- Community 152
- Community 153
- Community 154
- Community 155
- Community 156
- Community 157
- Community 158
- Community 159
- Community 160
- Community 161
- Community 162
- Community 163
- Community 164
- Community 165
- Community 166
- Community 167
- Community 170
- Community 171
- Community 172
- Community 173
- Community 174
- Community 175
- Community 176
- Community 178
- Community 179
- Community 180
- Community 183
- Community 184
- Community 185
- Community 186
- Community 187
- Community 188
- Community 189
- Community 190
- Community 193
- Community 194
- Community 195
- Community 200
- Community 201
- Community 202
- Community 203
- Community 206
- Community 207
- Community 208
- Community 209
- Community 210
- Community 211
- Community 212
- Community 213
- Community 214

## God Nodes (most connected - your core abstractions)
1. `MpcPlannerTerrain` - 168 edges
2. `MpcPlannerCfg` - 103 edges
3. `SemanticObstacleCount` - 84 edges
4. `height_at()` - 77 edges
5. `_device()` - 77 edges
6. `SemanticObstacleCurriculumCfg` - 69 edges
7. `MpcTerrainDifficultyPair` - 68 edges
8. `M1PandaTeacherEnvWrapper` - 64 edges
9. `MpcRobotState` - 60 edges
10. `plan_segment()` - 54 edges

## Surprising Connections (you probably didn't know these)
- `CommandCase` --uses--> `SemanticCourseStage`  [INFERRED]
  tests/fixtures/viewer_runtime_diagnostics.py → extension/semantic_course.py
- `test_controlled_crossing_accumulator_records_reset_stage_after_foot_over()` --calls--> `ControlledCrossingAccumulator`  [INFERRED]
  tests/test_mpc_policy_eval_metrics.py → scripts/mpc_policy_eval.py
- `test_controlled_crossing_accumulator_summarizes_success_by_speed_and_lateral()` --calls--> `ControlledCrossingAccumulator`  [INFERRED]
  tests/test_mpc_policy_eval_metrics.py → scripts/mpc_policy_eval.py
- `test_folded_load_ppo_configuration_is_exact_and_fresh()` --calls--> `get_m1_panda_folded_load_train_cfg()`  [EXTRACTED]
  tests/test_m1_panda_folded_load_ppo.py → agent/m1_panda_folded_load_train_cfg.py
- `test_teacher_train_cfg_has_exact_small_mlp_ppo_contract_and_is_independent()` --calls--> `get_m1_panda_teacher_train_cfg()`  [EXTRACTED]
  tests/test_m1_panda_teacher_train_static.py → agent/m1_panda_teacher_train_cfg.py

## Import Cycles
- 3-file cycle: `go2_pvcnn/sensor/lidar/lidar_cfg.py -> go2_pvcnn/sensor/lidar/lidar_sensor.py -> go2_pvcnn/sensor/lidar/ray_caster.py -> go2_pvcnn/sensor/lidar/lidar_cfg.py`
- 3-file cycle: `go2_pvcnn/sensor/lidar/lidar_cfg.py -> go2_pvcnn/sensor/lidar/ray_caster_cfg.py -> go2_pvcnn/sensor/lidar/ray_caster.py -> go2_pvcnn/sensor/lidar/lidar_cfg.py`

## Communities (220 total, 18 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (82): MpcPlannerStatus, IntEnum, _adapt_mpc_result_for_viewer(), _append_kit_arg(), _apply_direct_playback_to_robot(), _apply_viewer_terrain_selection(), _attach_reference_manager_if_enabled(), build_arg_parser() (+74 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (78): _as_part_value(), _base_footprint_centers(), _body_geometry_query_points(), _body_id_tensor(), _cached_circle_offsets(), _current_body_part_sample_points(), _current_scanner_terrain(), _expand_centers_with_offsets() (+70 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (77): MpcPlannerCfg, validate_mpc_config(), Parametric MPC backend for planner-owned reference cache runtime., _merge_subset_result(), _normal_state(), _normal_tensor(), _normal_terrain(), _parametric_result_from_state() (+69 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (80): action_rate_l2(), air_time_variance_penalty(), base_height_l2(), base_height_recovery_l2(), energy(), feet_air_time(), feet_air_time_positive_reward(), feet_slide() (+72 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (66): _copy_if_has(), MpcBodyCollisionLossCfg, MpcClearanceLossCfg, MpcContactRegularizationLossCfg, MpcDiagnosticsCfg, MpcDiagonalPairLossCfg, MpcFkBodyLegCollisionLossCfg, MpcFootTrajectoryRegularizationLossCfg (+58 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (74): MDP helpers for trajectory-guided teacher experiments., compute_tracking_metrics(), Tensor, Tracking metrics for planner-guided trajectory imitation., Compute scalar tracking metrics from current and reference states., downsample_height_map(), downsampled_elevation_semantic_scan(), downsampled_height_scan() (+66 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (74): body_heightfield_collision_loss(), finite_horizon_touchdown_phase(), high_large_stepcap_continuity_loss(), high_obstacle_avoidance_loss(), knee_shank_heightfield_collision_loss(), low_small_crossing_progress_loss(), low_small_foot_crossing_loss(), low_small_foot_over_loss() (+66 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (65): Exception, bottom_to_center_offset(), build_course_anchors(), _candidate_local_xy(), clear_semantic_course_children(), course_anchor_counts(), CourseAnchor, deterministic_shape_key() (+57 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (50): contact_binary_loss(), contact_transition_loss(), Tensor, Contact-related losses., support_stability_loss(), command_tracking_loss(), progress_direction_loss(), Tensor (+42 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (64): _ensure_episode_collision_state(), Tensor, Update sticky per-episode small-obstacle collision flags from real contact force, Update sticky per-episode small-obstacle collision flags from inferred map conta, SemanticObstacleCurriculumState, update_episode_small_collision_from_forces(), update_episode_small_collision_from_map_contacts(), build_teacher_gains() (+56 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (60): build_semantic_spatial_wave_reference(), build_stabilized_task_space_wheel_actions(), build_task_space_wheel_joint_actions(), build_wave_reference_actions(), compose_sequential_leg_actions(), Return wheels that enter a crossbar's swept volume without enough clearance., Start wave phase on obstacle entry and hold its gate through one full cycle., Build raw 12-leg-action diagonal wave references for each environment. (+52 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (31): M1PandaTeacherEnvWrapper, Tensor, Apply Teacher disturbances and bounded residual composition around an env., _FakeEnv, _FakeFrozenActor, _FakeObservationManager, _FakeRobot, _FakeScene (+23 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (61): compute_velocity_bounds(), Tensor, Intersect position, velocity, and one-step acceleration bounds., coordinated_jacobian(), damped_pseudoinverse(), planar_base_spatial_jacobian(), Tensor, Kinematics for planar M1 base and seven-joint Panda coordination. (+53 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (62): _effective_planning_variant_for_semantic(), _jitter_metrics(), _loss_only_low_small_foot_over_extra_loss(), _loss_only_weights_for_variant(), _LossOnlySemanticWeights, _rolling_segment_playback_error_metrics(), _selector_clearance_jitter_sort_key(), _selector_jitter_sort_key() (+54 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (42): base_wrench_to_body_local(), clear_external_wrench(), M1PandaDisturbanceCfg, M1PandaDisturbanceScheduler, dtype, Tensor, _quat_rotate(), _quat_rotate_inverse() (+34 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (31): Register the active Go2 semantic MPC environments with Gymnasium., No-MPC play config for flat-small avoidance policies., TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY, _apply_initial_terrain_selection(), _apply_keyboard_velocity_command(), _attach_reference_manager_if_enabled(), build_arg_parser(), _collect_runtime_debug_snapshot() (+23 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (53): M1SemanticGlobalContactSensor, Semantic obstacle contacts for the four M1 wheel links., _m1_semantic_global_contact_sensor(), M1PvcnnCrossing100mmEnvCfg, M1PvcnnCrossing60mmContactFreePlayEnvCfg, M1PvcnnCrossing60mmContactFreeTrainEnvCfg, M1PvcnnCrossing60mmDistilledPlayEnvCfg, M1PvcnnCrossing60mmEnvCfg (+45 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (38): Shared dimensions, joint ordering, and tensor validation for M1 + Panda WBC., apply_impedance(), Tensor, Feedforward plus impedance effort composition for 23 WBC joints., Compose and symmetrically clamp one finite 23-channel effort command., LongitudinalCommand, Deterministic command and trajectory primitives for rolling WBC control., RollingTeacherCfg (+30 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (47): DenseQpResult, build_rolling_wbc_problem(), Rolling priority configuration over the accepted whole-body QP., Balance-first weights for C1a base and wheel acceleration tracking., Translate semantic rolling names to the shared QP weight contract., Build C1a with hard dynamics/contact and rolling tracking weights., Solve one C1a rolling whole-body problem., RollingWbcCfg (+39 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (51): atomic_write_manifest(), build_run_manifest(), checkpoint_iteration(), _expected_state_shapes(), file_sha256(), _json_compatible_dataclass(), _load_checkpoint(), load_frozen_teacher_actor() (+43 more)

### Community 20 - "Community 20"
Cohesion: 0.08
Nodes (39): _contact_pattern_metrics(), main(), Tensor, _viewer_plan_with_memory(), _binary_gate(), _clone_mpc_state(), _command_tensor_from_spec(), _enable_4096_runtime_test() (+31 more)

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (45): bounded_unit_interval(), cubic_bezier(), _cubic_bezier_with_leg_phase(), decode_parametric_trajectory(), DecodedParametricTrajectory, init_parametric_variables(), MpcParametricVariables, _optim_zeros() (+37 more)

### Community 22 - "Community 22"
Cohesion: 0.08
Nodes (35): Terrain importer that creates the static semantic course before scene sensors in, SemanticCourseTerrainImporter, SemanticObstacleCurriculumCfg, CriticElevationSemanticMapCfg, CriticStateCfg, _flat_small_avoidance_terrain_cfg(), PolicyElevationSemanticMapCfg, PolicyStateCfg (+27 more)

### Community 23 - "Community 23"
Cohesion: 0.12
Nodes (31): MotionDistributionResult, M1PandaWbcTeacher, Tensor, One-environment deterministic C0 Teacher without Isaac dependencies., TeacherCfg, TeacherState, BandLimitedPoseTrajectory, Tensor (+23 more)

### Community 24 - "Community 24"
Cohesion: 0.06
Nodes (46): apply_fixed_course_gate_safety_window(), blend_policy_wave_gate(), build_lateral_steering_correction(), build_spatial_axle_wheel_targets(), build_temporal_axle_wheel_targets(), Return wheel center X relative to a world-aligned fixed crossbar., Lift only the axle at the bar while all four wheels keep rolling., Low-pass a wave teacher target without clipping its steady-state amplitude. (+38 more)

### Community 25 - "Community 25"
Cohesion: 0.05
Nodes (18): build_sequential_phase_observation(), Encode non-wave/phase identity and bounded within-phase progress., M1RslRlEnvWrapper, Flatten the M1 smoke `policy` observation group for RSL-RL., test_sequential_phase_observation_disambiguates_wave_stage_and_progress(), test_m1_wrapper_applies_wheel_action_signs_after_target_locking(), test_m1_wrapper_can_equalize_fast_wheels_down_to_slowest_actual_velocity(), test_m1_wrapper_can_lock_all_wheel_targets_after_velocity_corrections() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.09
Nodes (41): evaluate_hard_reasons(), Tensor, Hard-diagnostics helpers for batch MPC results., Return hard reason mask shaped [B, R]., status_from_hard_reasons(), command_frame_axes(), _align_low_small_swing_to_touchdown(), _build_semantic_proximity_field() (+33 more)

### Community 27 - "Community 27"
Cohesion: 0.10
Nodes (32): BaseAssistCfg, BaseAssistDecision, compute_base_assist(), Tensor, Bounded planar M1 assistance for Panda null-space recovery., Choose a bounded planar correction only when it improves arm margin., _vector(), _check() (+24 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (34): LongitudinalCommandSchedule, LongitudinalScheduleCfg, PlanarBodyFrameTrajectory, Advect a band-limited local EE trajectory with planar root motion., Five-phase C1a speed schedule and its physical slew limit., Generate exactly one rate-limited longitudinal command per mission step., Deployable C1a mission and nominal commands for Student S1., StudentMissionSample (+26 more)

### Community 29 - "Community 29"
Cohesion: 0.09
Nodes (24): FakeEnv, FakeObservationManager, FakeRobot, FakeScene, _load_wrapper(), dict, test_constructor_does_not_label_pre_reset_physics_state_as_reset_dr(), test_default_wrapper_does_not_create_or_advance_disturbance() (+16 more)

### Community 30 - "Community 30"
Cohesion: 0.11
Nodes (41): MpcRuntimeCfg, _circular_abs_delta(), _circular_mean2(), _circular_signed_delta(), diagonal_pair_loss(), _forward_phase_distance(), phase_prior_loss(), Tensor (+33 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (34): M1PandaCoordinatedActionsCfg, M1PandaCoordinatedEnvCfg, M1PandaCoordinatedEventsCfg, M1PandaCoordinatedObservationsCfg, M1PandaCoordinatedRewardsCfg, PolicyCfg, ObsGroup, Combined M1 + Panda coordinated mission environment configuration. (+26 more)

### Community 32 - "Community 32"
Cohesion: 0.10
Nodes (32): build_wheel_contact_jacobian(), contact_point_linear_jacobian(), Tensor, Pure rolling-contact kinematics for the M1 wheel set., Measure longitudinal rolling residual and lateral slip at wheel bottoms., Physical wheel constants in canonical FAR/FBL/RAR/RBL order., Map one longitudinal base command to four signed wheel angular speeds., Map generalized velocity to a rigid body's offset-point linear velocity. (+24 more)

### Community 33 - "Community 33"
Cohesion: 0.05
Nodes (41): axle_pair_crossing_progress_score(), build_teacher_student_residual(), expand_checkpoint_observations(), merge_task_space_support_with_jointspace_active(), prepare_wave_checkpoint(), progress_potential_delta(), Pure helpers shared by the autonomous M1 curriculum scripts., Return dense stage, lift, and forward-swing progress for the active wheel. (+33 more)

### Community 34 - "Community 34"
Cohesion: 0.11
Nodes (30): _atomic_json_write(), _atomic_runner_save(), AtomicCheckpointController, _finite(), GuardDecision, GuardSnapshot, Path, PathLike (+22 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (23): Instantaneous bottom-point velocities expressed in the root heading frame., RollingContactMetrics, M1PandaRollingWbcTeacher, Tensor, One-environment deterministic C1a rolling whole-body Teacher., Restart scored commands without disturbing settled low-level state., RollingTeacherState, _rotation_2d() (+15 more)

### Community 36 - "Community 36"
Cohesion: 0.08
Nodes (22): Minimal VecEnv protocol used by the local RSL-RL runner., Base class for vectorized environments consumed by `OnPolicyRunner`., VecEnv, _finite_float(), freeze_environment_metrics(), freeze_episode_metrics(), LearnResult, OnPolicyRunner (+14 more)

### Community 37 - "Community 37"
Cohesion: 0.12
Nodes (28): _atomic_json_save(), _atomic_torch_save(), _clone_record(), DaggerRecord, _payload_to_record(), Path, PathLike, Versioned, hard-sample-aware replay storage for M1 + Panda Student S1. (+20 more)

### Community 38 - "Community 38"
Cohesion: 0.11
Nodes (26): M1ResidualActionComposer, M1ResidualActionComposerCfg, dtype, Tensor, Stateful bounded residual composition for M1 hybrid actions., Return the base action plus an amplitude- and slew-limited residual., Clear all state, or only state belonging to selected environments., Physical limits and existing M1 action scales. (+18 more)

### Community 39 - "Community 39"
Cohesion: 0.14
Nodes (27): SemanticObstacleCount, _Cfg, _Data, _Env, _force(), _install_fake_isaaclab(), _load_curriculums_module(), dict (+19 more)

### Community 40 - "Community 40"
Cohesion: 0.09
Nodes (34): extract_roll_pitch_batch(), extract_yaw_batch(), isaac_state_to_planner_state(), _normalize_planner_result_for_reference_cache(), planner_result_to_reference_cache(), Tensor, quat_wxyz_to_xyzw(), quat_xyzw_to_wxyz() (+26 more)

### Community 41 - "Community 41"
Cohesion: 0.08
Nodes (24): _m1_lateral_position_l2(), _m1_phase_aware_bad_orientation(), _m1_phase_aware_minimum_base_height(), _m1_prepared_leg_action_l2(), _m1_raw_leg_action_l2(), _m1_semantic_body_clearance_term(), _m1_semantic_pair_lift_reward(), _m1_semantic_wheel_over_term() (+16 more)

### Community 42 - "Community 42"
Cohesion: 0.08
Nodes (9): ActorCritic, get_activation(), ActorCriticRecurrent, Memory, _actor(), test_clip_std_uses_effective_units_for_both_modes(), test_log_std_is_exponentiated_and_has_log_state_key(), test_scalar_std_is_used_directly_and_keeps_legacy_state_key() (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.11
Nodes (26): atomic_write_summary(), build_arg_parser(), build_teacher_gains(), C0Summary, contact_point_linear_jacobian(), _cpu64(), _exact_id(), _format_diagnostics() (+18 more)

### Community 44 - "Community 44"
Cohesion: 0.08
Nodes (24): CommandTerm, CommandTermCfg, Command term configurations., _abs_range_from_signed_range(), GoalAnchoredVelocityCommand, GoalAnchoredVelocityCommandCfg, Tensor, Velocity command with curriculum support (ranges -> limit_ranges). (+16 more)

### Community 45 - "Community 45"
Cohesion: 0.13
Nodes (4): MpcTrajectoryManager, _normalize_body_name(), Tensor, Planner-owned cache manager for MPC reference trajectories.

### Community 46 - "Community 46"
Cohesion: 0.12
Nodes (30): M1RollCommandsCfg, M1RollEnvCfg, PolicyCfg, ObsGroup, M1 rolling task for the first autonomous locomotion stage., Slow forward commands for the first stable rolling curriculum., M1SmokeActionsCfg, M1SmokeCommandsCfg (+22 more)

### Community 47 - "Community 47"
Cohesion: 0.10
Nodes (29): get_m1_panda_teacher_train_cfg(), Return a fresh small-MLP PPO config for A0/A1 Teacher training., aggregate_candidate_summaries(), _finite_number(), Tensor, Pure full-scale evaluation helpers for M1 + Panda Teacher checkpoints., Reject evaluation rows that do not prove the full-scale contract., Aggregate the exact three-seed evidence for one candidate checkpoint. (+21 more)

### Community 48 - "Community 48"
Cohesion: 0.16
Nodes (26): M1PandaStudent, Tensor, Explicit temporal history and GRU estimator/actor for Student S1., GRU estimator plus bounded residual actor., Per-environment rolling history without hidden global recurrent state., StudentHistoryBuffer, StudentNetworkCfg, Validate and atomically publish a resumable Student checkpoint. (+18 more)

### Community 49 - "Community 49"
Cohesion: 0.13
Nodes (34): refresh_targeted_scanner_pose(), apply_mpc_debug_variant_cfg(), _command_frame(), compute_plane_low_small_fk_metrics(), compute_segmented_plane_low_small_fk_metrics(), main(), mpc_debug_extra_loss(), _patched_reachable_loss_for_variant() (+26 more)

### Community 50 - "Community 50"
Cohesion: 0.11
Nodes (34): reachable_cfg_for_variant(), reachable_command_frame_endpoint_metrics(), reachable_distance_window_weights(), reachable_extra_loss(), reachable_foot_height_relative_to_root_metrics(), reachable_foot_over_arc_metrics(), test_command_frame_endpoint_metrics_report_forward_swing_then_rear_touchdown(), test_foot_height_relative_to_root_metrics_report_foot_above_root() (+26 more)

### Community 51 - "Community 51"
Cohesion: 0.14
Nodes (29): AssetBaseCfg, M1RollObservationsCfg, M1RollRewardsCfg, Trainable M1 rolling environment.  This is the first long-running M1 locomotion, Low-dimensional proprioception for rolling control., Rolling rewards: move forward without tipping or scraping the body., M1SmallObstacle10mmEnvCfg, M1SmallObstacle10mmSceneCfg (+21 more)

### Community 52 - "Community 52"
Cohesion: 0.14
Nodes (28): bpearl_pattern(), BpearlPatternCfg, GridPatternCfg, LidarPatternCfg, LivoxPatternCfg, PatternBaseCfg, PinholeCameraPatternCfg, r"""Create a :class:`PinholeCameraPatternCfg` class instance from an intrinsic m (+20 more)

### Community 53 - "Community 53"
Cohesion: 0.15
Nodes (32): _body_stencil_xy(), _crossing_metrics(), _finite_ratio(), _loss_only_continuity_anchor_extra_loss(), _loss_only_high_large_avoid_extra_loss(), _loss_only_high_large_fk_semantic_extra_loss(), _loss_only_high_large_handoff_extra_loss(), _loss_only_low_small_crossing_extra_loss() (+24 more)

### Community 54 - "Community 54"
Cohesion: 0.11
Nodes (30): FkCollisionMargins, parametric_fk_body_leg_collision_loss(), parametric_plane_root_z_target_loss(), parametric_swing_foot_clearance_loss(), parametric_touchdown_keepout_loss(), parametric_trajectory_fk_consistency_loss(), Tensor, Loss helpers for parametric MPC trajectory variables. (+22 more)

### Community 55 - "Community 55"
Cohesion: 0.11
Nodes (16): get_coordinated_joint_reset_diagnostics(), Return the latest sampled joint reset deviations for one articulation., configure_coordinated_training_domain_randomization(), Set the exact approved train ranges or restore deterministic defaults., M1PandaCoordinatedEnvWrapper, Tensor, atomic_write_summary(), build_arg_parser() (+8 more)

### Community 56 - "Community 56"
Cohesion: 0.20
Nodes (30): _arrived(), _axis_angle(), _balance_score(), coordinated_base_target_error_b(), coordinated_base_tracking_reward(), coordinated_base_velocity_tracking_reward(), coordinated_desired_twist_b(), coordinated_ee_pose_error_b() (+22 more)

### Community 57 - "Community 57"
Cohesion: 0.11
Nodes (25): _atomic_json_save(), _atomic_torch_save(), _load_actual_manifest(), _load_payload(), load_student_checkpoint(), LoadedStudentCheckpoint, Module, Path (+17 more)

### Community 58 - "Community 58"
Cohesion: 0.14
Nodes (31): _actual_foot_pos_w(), apply_command_to_env(), _base_env(), build_arg_parser(), build_mpc_foot_markers(), build_mpc_foot_trajectory_markers(), command_body_source_diagnostics(), controlled_crossing_step_metrics() (+23 more)

### Community 59 - "Community 59"
Cohesion: 0.10
Nodes (24): Configuration for the combined M1 and Panda articulation., M1PandaSmokeActionsCfg, M1PandaSmokeEnvCfg, M1PandaSmokeSceneCfg, Isolated smoke environment for the combined M1 and Panda articulation., Flat smoke scene using the combined articulation., Exactly 12 M1 leg-position plus four M1 wheel-velocity actions., No-planner smoke cfg for the combined 25-DOF articulation. (+16 more)

### Community 60 - "Community 60"
Cohesion: 0.08
Nodes (19): GlobalRigidObjectCollectionCfg, RigidObjectCollectionCfg, 全局刚体对象集合的配置          Configuration for global rigid object collections., GlobalRigidObjectCollection, RigidObjectCollectionCfg, Tensor, 重置物体集合状态                  Reset the rigid object collection state., 写入物体位姿到仿真                  Write object link pose to simulation. (+11 more)

### Community 61 - "Community 61"
Cohesion: 0.18
Nodes (28): _as_cpu_float64(), _canonicalize(), _CanonicalProblem, DenseQpProblem, _diagnostic_seed(), _metrics(), _primal_feasible(), Tensor (+20 more)

### Community 62 - "Community 62"
Cohesion: 0.11
Nodes (20): configure_pvcnn_cuda(), _prepend(), Path, Runtime setup for the repository-local PVCNN CUDA extension toolchain., Use the local CUDA development prefix when CUDA_HOME is not already set., downsample_perception_maps(), grid_elevation_to_point_cloud(), logits_to_semantic_channel() (+12 more)

### Community 63 - "Community 63"
Cohesion: 0.13
Nodes (21): _as_device_tensor(), blend_reference_caches(), clone_reference_cache(), mpc_result_to_reference_cache(), dtype, Tensor, Reference-cache adapter for batch MPC planner results., result_new_ok_mask() (+13 more)

### Community 64 - "Community 64"
Cohesion: 0.14
Nodes (26): fk_feet_from_joint_angles(), fk_leg_points_from_joint_angles(), MpcLegPoints, Tensor, GPU batched IK helper for dense MPC planner outputs., Forward-kinematics foot, knee, and shank samples from planner-order joints., Forward-kinematics foot positions from world root pose and planner-order joints., Solve per-frame Go2 leg IK from world-frame root pose and foot targets. (+18 more)

### Community 65 - "Community 65"
Cohesion: 0.09
Nodes (16): Set the class type to LidarSensor after initialization., LidarSensor, Tensor, Calculates the true world position of the sensor, including the local offset., Initialize ray patterns for LiDAR sensor., Update sensor buffers with LiDAR-specific processing.                  This meth, Update ray directions for dynamic patterns (e.g., Livox sensors)., Apply noise to distance measurements.                  Simulates realistic senso (+8 more)

### Community 66 - "Community 66"
Cohesion: 0.09
Nodes (28): evaluate_obstacle_gate(), evaluate_roll_gate(), Track straight-over semantic small-obstacle crossings without counting avoidance, Evaluate the deterministic Stage 1 rolling acceptance gate., Evaluate deterministic whole-robot obstacle clearance., Track wheel lift before contact and contact-free clearance above a crossbar., Check lift thresholds only for wheels selected by the crossing task., required_axle_lift_passed() (+20 more)

### Community 67 - "Community 67"
Cohesion: 0.13
Nodes (15): BatchedRollingTeacherBank, Tensor, One-Teacher-per-environment orchestration for Student S1 collection., Keep C1a Teacher controller state isolated per simulated environment., _bank(), _fake_env(), _FakeContactSensor, _FakeMath (+7 more)

### Community 68 - "Community 68"
Cohesion: 0.15
Nodes (25): _apply_semantic_small_height_override(), _apply_semantic_small_profile_override(), _batch_size_from_tensors(), build_command_cases(), CommandCase, _constant_over_time_ratio(), _format_hard_reason_mask(), format_hard_reason_summary() (+17 more)

### Community 69 - "Community 69"
Cohesion: 0.12
Nodes (22): _load_script(), _load_smoke(), test_build_log_dir_is_stage_scoped_and_refuses_existing_directory(), test_cli_contract_forbids_run_name_when_resuming(), test_cli_contract_rejects_invalid_stage_base_checkpoint_combinations(), test_cli_contract_rejects_nonpositive_training_overrides(), test_fork_cli_requires_isolated_a1_run(), test_fork_creates_fresh_stage_directory_without_mutating_source() (+14 more)

### Community 71 - "Community 71"
Cohesion: 0.12
Nodes (22): get_m1_train_cfg(), Return a small MLP PPO config for `Isaac-M1-Walk-v0`., datetime, Hold exact teacher control first, then linearly hand control to the student., scheduled_student_rollout_weight(), M1PvcnnRslRlEnvWrapper, Replace the actor semantic map with PVCNN predictions; keep critic ground truth., build_parser() (+14 more)

### Community 72 - "Community 72"
Cohesion: 0.13
Nodes (22): Kinematic bounds for M1 + Panda coordination., _indices_for(), dtype, Tensor, Validate a finite tensor without copying or changing its device., Runtime articulation indices expressed in the canonical WBC order., require_tensor(), WbcJointMap (+14 more)

### Community 73 - "Community 73"
Cohesion: 0.10
Nodes (16): LidarRayCaster, ndarray, Tensor, Get the rigid body view for a given prim path.                  This finds the r, Get the collision groups for each rigid body in the rigid body view., Search through the xform prim and its children to find all meshes and merge them, Initialize warp meshes for ray casting.                  This method:         1., Initialize ray patterns and collision group buffers. (+8 more)

### Community 74 - "Community 74"
Cohesion: 0.14
Nodes (14): CoordinatedDisturbanceCfg, CoordinatedDisturbanceScheduler, dtype, Tensor, Seeded Panda-hand wrench curriculum for coordinated M1 + Panda training., Reset selected segment rows without rewinding the curriculum clock., Generate independent base-frame wrench segments at the control frequency., RSL-RL wrapper for the combined coordinated 23-effort task. (+6 more)

### Community 75 - "Community 75"
Cohesion: 0.11
Nodes (10): PPO, Adapt LR toward the requested KL while respecting configured bounds., Clamp policy exploration in physical standard-deviation units., Compute mean old-to-new policy KL over active action dimensions., Train PVCNN on collected point clouds and semantic labels (synchronous)., test_adaptive_lr_moves_toward_desired_kl(), test_adaptive_lr_obeys_configured_bounds(), test_invalid_optimizer_bounds_are_rejected() (+2 more)

### Community 76 - "Community 76"
Cohesion: 0.12
Nodes (25): advance_runner_after_resume(), apply_recovery_resume_train_cfg(), build_arg_parser(), build_log_dir(), load_runner_checkpoint(), main(), mark_recovery_block_completed(), preserve_recovery_resume_state() (+17 more)

### Community 77 - "Community 77"
Cohesion: 0.20
Nodes (21): DaggerSelection, DaggerStageCfg, Tensor, Deterministic DAgger action selection and supervised Student S1 losses., Return the six scalar normalized S1 DAgger loss terms., Choose per-environment Teacher/Student actions reproducibly.      Any unsafe Stu, _require_action(), _require_loss_tensor() (+13 more)

### Community 78 - "Community 78"
Cohesion: 0.09
Nodes (14): Tensor, Initialize warp meshes and build semantic classification mapping., Initialize ray patterns and semantic label buffers., Update sensor buffers including semantic classification.                  This e, Compute semantic labels for ray hits using returned mesh IDs.                  T, Get semantic class labels for each ray hit.                  Args:             e, A Semantic LiDAR sensor that classifies ray hits by object type.          This s, Get mesh prototype IDs for each ray hit.                  Args:             env_ (+6 more)

### Community 79 - "Community 79"
Cohesion: 0.19
Nodes (17): Return the immutable contract for ``name``., stage_spec(), EligibilitySnapshot, _finite(), FoldedLoadTrainingGuard, GuardDecision, Pure eligibility, catastrophe, and atomic acceptance for folded-load PPO., Evaluate exact rolling windows while enforcing always-on stop rules. (+9 more)

### Community 80 - "Community 80"
Cohesion: 0.13
Nodes (6): M1PandaFoldedLoadEnvWrapper, Tensor, Install one fixed command per environment before deterministic evaluation., Return completed records exactly once without flattening command attribution., Keep the inactive effort coordinates tied to the approved dynamic fold., Own episode commands, metrics, and the exact-zero inactive boundary.

### Community 81 - "Community 81"
Cohesion: 0.10
Nodes (14): ActorCriticCNN, get_activation(), Module, Get activation function by name., Initialize weights using orthogonal initialization., Reset for recurrent policies (not used here)., Forward pass not implemented - use act() or evaluate()., Actor-Critic网络，整合2D CNN编码器处理代价地图          输入:         - observations: (batch, nu (+6 more)

### Community 82 - "Community 82"
Cohesion: 0.13
Nodes (19): Reference generator scaffolding for planner-guided training., Configuration for future planner-backed reference generation., Build a tiny, import-safe placeholder reference trajectory., Return a populated cache with a small forward drift., ReferenceGenerator, ReferenceGeneratorConfig, Reference trajectory helpers for the active batched planner runtime., ensure_kinematic_footsteps_on_syspath() (+11 more)

### Community 83 - "Community 83"
Cohesion: 0.10
Nodes (6): _load_verifier_contract_helpers(), test_articulation_root_predicate_requires_exact_m1_base_link(), test_dependency_classification_only_allows_exact_omnipbr(), test_mount_contract_predicate_checks_targets_enabled_and_inclusion(), test_mount_plane_predicate_enforces_micrometer_tolerance(), test_surface_gap_predicate_rejects_both_air_gap_and_penetration()

### Community 84 - "Community 84"
Cohesion: 0.10
Nodes (19): array, float32, Ray caster implementation for LiDAR sensor with dynamic mesh support.  This is a, # NOTE: This mesh_prototype_ids.extend is assuming the rigid_body_view has the o, Warp utility functions for ray casting and mesh operations., dtype, Warp kernels for grouped ray casting against meshes.  This module provides CUDA, Warp kernel for ray casting against grouped meshes with transforms.          Thi (+11 more)

### Community 85 - "Community 85"
Cohesion: 0.23
Nodes (19): apply_student_residual(), _group_vector(), Tensor, Frozen deployable observation and residual-action contracts for Student S1., Convert safe Teacher position/velocity targets into normalized residual labels., Apply amplitude and physical slew limits before reconstructing safe targets., _require_positive_finite(), StudentActionCommand (+11 more)

### Community 86 - "Community 86"
Cohesion: 0.13
Nodes (20): balanced_eval_commands(), classify_command_buckets(), CommandBatch, CommandFamily, _nonzero_uniform(), IntEnum, Tensor, Pure contracts for the M1 + Panda folded-load locomotion curriculum. (+12 more)

### Community 87 - "Community 87"
Cohesion: 0.12
Nodes (6): RolloutStorage, Transition, Splits trajectories at done indices. Then concatenates them and pads with zeros, split_and_pad_trajectories(), store_code_state(), unpad_trajectories()

### Community 88 - "Community 88"
Cohesion: 0.12
Nodes (15): manager_supports_current_reference(), PlannerResultProtocol, Tensor, Backend-agnostic trajectory contracts used by manager/reward/viewer wiring., Minimal contract expected by reward/viewer consumers., Refresh internal cache from env state and return ready cache., Return current-frame reference fields., Return current frame index for each environment row. (+7 more)

### Community 89 - "Community 89"
Cohesion: 0.21
Nodes (21): active_action_l2_tensor(), active_action_rate_l2_tensor(), folded_load_active_action_l2(), folded_load_active_action_rate_l2(), folded_load_compat_base_error_b(), folded_load_compat_ee_error_b(), folded_load_desired_twist_b(), folded_load_lateral_velocity_l2() (+13 more)

### Community 90 - "Community 90"
Cohesion: 0.17
Nodes (21): binary_contact_state(), elevation_map(), elevation_map_height_scan(), elevation_semantic_dual_map(), goal_based_velocity_commands(), pvcnn_features_with_cost_map(), ManagerBasedRLEnv, SceneEntityCfg (+13 more)

### Community 91 - "Community 91"
Cohesion: 0.18
Nodes (18): _accepted_checkpoint(), _atomic_json(), build_arg_parser(), CurriculumState, _document_path(), ExecutionRequest, main(), _persist() (+10 more)

### Community 92 - "Community 92"
Cohesion: 0.15
Nodes (11): ContactSensor, ContactSensor variant for global static semantic-course objects., resolve_contact_body_paths(), _semantic_root_from_filter_expr(), SemanticGlobalContactSensor, ContactSensorCfg, _semantic_global_contact_sensor(), _install_fake_isaaclab_contact_sensor() (+3 more)

### Community 93 - "Community 93"
Cohesion: 0.19
Nodes (17): _drop_target_from_root(), DropTarget, _empty_target(), _finite_bool(), main(), Any, Path, Tensor (+9 more)

### Community 94 - "Community 94"
Cohesion: 0.19
Nodes (17): build_m1_smoke_action(), M1SmokeControllerCfg, _positive_sine(), Tensor, Open-loop M1 smoke controller actions.  The output order matches ``M1SmokeAction, Numerical parameters for the open-loop M1 smoke controller., Build a normalized M1 smoke action tensor.      The IsaacLab action terms apply, build_arg_parser() (+9 more)

### Community 95 - "Community 95"
Cohesion: 0.17
Nodes (16): RuntimeError, build_arg_parser(), _check_finite(), _check_no_reset(), _evaluate_channel(), _exact_body_id(), main(), _prepare_independent_window() (+8 more)

### Community 96 - "Community 96"
Cohesion: 0.16
Nodes (20): aggregate_controlled_crossing_rounds(), _attach_reference_manager_if_enabled(), checkpoint_path(), _close_env(), command_for_step(), _command_tuple_from_args(), _make_eval_env_wrapper(), make_run_output_dir() (+12 more)

### Community 97 - "Community 97"
Cohesion: 0.16
Nodes (15): _attach_reference_manager_if_enabled(), _attach_single_env_livestream_follow_camera(), build_arg_parser(), _compute_follow_camera_pose(), _launch_app(), _livestream_camera_update_interval(), main(), _parse_args() (+7 more)

### Community 99 - "Community 99"
Cohesion: 0.12
Nodes (20): _aggregate(), _aggregate_variants(), _candidate_variants_for_variant(), _command_path_metrics(), _low_small_foot_over_metrics(), main(), Check whether a swing foot actually passes over the low-small footprint., run_probe() (+12 more)

### Community 100 - "Community 100"
Cohesion: 0.22
Nodes (19): _aggregate_terrain_case_rows(), _aggregate_variant_rows(), _build_speed_grid_commands(), _iter_true_runs(), main(), _parse_command(), _parse_float_list(), Tensor (+11 more)

### Community 101 - "Community 101"
Cohesion: 0.17
Nodes (14): _asset_cfg(), _class(), _class_assignments(), _fake_env(), _keywords(), dict, _Scene, test_base_xy_drift_is_relative_to_each_environment_origin() (+6 more)

### Community 102 - "Community 102"
Cohesion: 0.13
Nodes (14): 初始化全局刚体对象                  Initialize the global rigid object.          Args:, Action functions for Go2 PVCNN locomotion., MDP components for Go2 PVCNN environment., m1_panda_mount_wrench_b(), Tensor, Mount-wrench observation helpers for the combined M1 and Panda articulation., Express a world-frame sensor wrench at the base-link origin in base coordinates., Return the parent-on-child mount wrench about the base origin in the base frame. (+6 more)

### Community 103 - "Community 103"
Cohesion: 0.15
Nodes (11): _cfg_use_yaw_only_rays(), Tensor, Map ray-cast face ids to semantic ids without crashing on invalid indices., Isaac Lab ``RayCaster`` with one merged mesh, face-id semantics, and elevation +, Merge all ``mesh_prim_paths`` into one warp mesh and build per-face semantic ids, Rebuild the warp mesh if startup-generated semantic ids 1/2 appeared after senso, Refresh only selected env rows without triggering SensorBase's full outdated pas, Match both legacy ``attach_yaw_only`` and newer ``ray_alignment='yaw'`` RayCaste (+3 more)

### Community 104 - "Community 104"
Cohesion: 0.24
Nodes (18): _articulation_root_errors(), _asset_path_text(), _build_parser(), _classify_unresolved_dependencies(), _independent_mount_parent_local_pos(), _inspect_dependencies(), _is_within(), main() (+10 more)

### Community 105 - "Community 105"
Cohesion: 0.22
Nodes (13): _fake_cfg(), FakeEnv, FakeRobot, _ids(), _load_configure_helper(), Tensor, reset_coordinated_joints_by_offset(), SceneEntityCfg (+5 more)

### Community 106 - "Community 106"
Cohesion: 0.16
Nodes (11): _load_probe(), test_base_frame_wrench_is_transformed_into_rotated_hand_local_axes_each_step(), test_case_table_is_the_verbatim_base_frame_contract(), test_channel_check_accepts_fraction_boundary_and_rejects_point_eighty_eight(), test_channel_check_requires_mean_expected_sign_and_strict_twenty_percent_ratio(), test_empty_clear_accepts_only_the_known_isaaclab_empty_assignment_bug(), test_empty_clear_rejects_an_unrelated_shape_mismatch(), test_independent_case_windows_reset_and_do_not_reuse_baselines() (+3 more)

### Community 107 - "Community 107"
Cohesion: 0.22
Nodes (15): clamp_row_index(), count_for_row(), count_to_dict(), layout_index_for_row(), layout_values_for_row(), Semantic obstacle curriculum configuration and state helpers., _validate_count_sequence(), _validate_float_sequence() (+7 more)

### Community 108 - "Community 108"
Cohesion: 0.26
Nodes (17): _apply_world_transform(), _collect_supported_geometry_prims(), _cube_prim_to_world_trimesh(), _geometry_prim_to_world_trimesh(), _mesh_prim_to_world_trimesh(), _orient_points_from_z_axis(), ndarray, Depth-first collect every supported geometry prim under ``root_prim``. (+9 more)

### Community 109 - "Community 109"
Cohesion: 0.22
Nodes (15): Return a valid requested PPO update count., validate_max_iterations(), _atomic_save(), atomic_write_json(), build_arg_parser(), main(), ParentLineage, prepare_empty_run_dir() (+7 more)

### Community 110 - "Community 110"
Cohesion: 0.24
Nodes (14): _atomic_copy(), _atomic_json(), AtomicStageArtifacts, Path, PathLike, Publish evaluation reports and the accepted checkpoint atomically., Aggregate fixed-seed evidence without ever publishing a parent policy., sha256_file() (+6 more)

### Community 111 - "Community 111"
Cohesion: 0.25
Nodes (17): Layer, RobotAssembler, _assemble_m1_panda(), build_asset(), mount_offset_z(), _mount_patch_top_z(), _prepare_serialized_root_layer(), ndarray (+9 more)

### Community 112 - "Community 112"
Cohesion: 0.28
Nodes (16): _balanced_records(), _episode(), _load(), _manifest(), Path, test_clean_ineligible_smoke_is_not_a_process_failure(), test_diagnostic_manifest_can_never_be_used_as_curriculum_parent(), test_diagnostic_manifest_dispatches_to_non_promoting_finalizer() (+8 more)

### Community 113 - "Community 113"
Cohesion: 0.13
Nodes (11): Agent configuration for Go2 training., RSL-RL PPO configuration for coordinated M1 + Panda training., RSL-RL PPO configuration for folded-load M1 + Panda locomotion., RSL-RL PPO configuration for M1 + Panda Teacher balance stages., RSL-RL training configuration for M1 locomotion., get_train_cfg(), Training configuration for the active semantic MPC teacher experiment., Training config for MPC semantic trajectory imitation. (+3 more)

### Community 114 - "Community 114"
Cohesion: 0.22
Nodes (15): filtered_contact_penalty_from_force_matrix(), global_semantic_contact_penalty_from_matrices(), Tensor, Rewards based on IsaacLab filtered contact sensors for semantic objects., Return negative contact penalty for semantic small/large object contacts., Aggregate one filtered contact sensor matrix into a per-env penalty., Aggregate two global semantic contact matrices into a per-env penalty., Return negative semantic collision penalty from two global contact sensors. (+7 more)

### Community 115 - "Community 115"
Cohesion: 0.27
Nodes (16): _body_relative_foot(), _command_sequences(), _command_tensor(), _leg_swing_stats(), main(), _max(), _mean(), _pair_left_right_alternation_stats() (+8 more)

### Community 116 - "Community 116"
Cohesion: 0.29
Nodes (15): _assert_action_contract(), _assert_registry_contract(), _assignment_name(), _assignment_value(), _assignments(), _class(), _keywords(), test_action_contract_rejects_an_extra_panda_term() (+7 more)

### Community 117 - "Community 117"
Cohesion: 0.23
Nodes (16): _load_script(), _source(), test_adapter_captures_initial_hand_quaternion_for_relative_pose(), test_adapter_reads_live_physx_dynamics_and_explicitly_combines_bias_force(), test_bias_reader_upgrades_legacy_joint_only_physx_forces_to_floating_base(), test_body_jacobian_offsets_body_id_when_physx_omits_root_link(), test_body_jacobian_uses_direct_body_id_when_physx_includes_root_link(), test_c0_adapter_keeps_accepted_radius_default_but_allows_c1a_injection() (+8 more)

### Community 118 - "Community 118"
Cohesion: 0.17
Nodes (8): _FakePhysxView, _FakeRobot, Tensor, _quat_rotate(), _quat_rotate_inverse(), test_adapter_rejects_missing_or_duplicate_explicit_body(), test_adapter_rotates_raw_joint_wrench_once_then_shifts_about_base(), wrench_module()

### Community 119 - "Community 119"
Cohesion: 0.30
Nodes (15): get_m1_panda_folded_load_train_cfg(), Return a fresh PPO configuration matched to the 200 Hz task., EpisodeRecord, _bucket(), build_arg_parser(), _directional_report(), evaluate_records(), _event_rate() (+7 more)

### Community 120 - "Community 120"
Cohesion: 0.12
Nodes (9): Returns the wrapper name and environment representation., Returns the string representation of the wrapper., Returns the configuration class instance of the environment., Returns the render mode., Returns the class name of the wrapper., Wraps Isaac Lab environment for RSL-RL with PVCNN integration.          This wra, Set the seed for the environment., Close the environment. (+1 more)

### Community 121 - "Community 121"
Cohesion: 0.24
Nodes (15): _source(), test_eval_records_body_command_source_diagnostics(), test_eval_records_flat_planned_direction_metrics(), test_mpc_policy_eval_collects_small_collision_env_rate_from_semantic_sensor(), test_mpc_policy_eval_collects_tracking_reference_from_runtime_manager(), test_mpc_policy_eval_follow_camera_debug_logs_viewport_camera_state(), test_mpc_policy_eval_has_controlled_crossing_metric_helpers(), test_mpc_policy_eval_livestream_draws_full_foot_trajectories_and_follows_robot() (+7 more)

### Community 122 - "Community 122"
Cohesion: 0.17
Nodes (10): BaseException, _candidate_runtime_devices(), _close_runtime_app(), _construct_runtime_launcher(), _ensure_runtime_app(), _is_runtime_resource_error(), make_real_runtime_fixture(), Shrink semantic-course runtime smoke to a 4x1 terrain grid.          The feature (+2 more)

### Community 123 - "Community 123"
Cohesion: 0.19
Nodes (9): Tensor, Cost map generation from ground truth semantic labels (Teacher Mode).  This modu, 生成2D代价地图（使用真实语义标签，无PVCNN推理）          Teacher模式特点：     1. 直接使用LiDAR语义标签（semantic_, 计算离机器人的距离代价（机器人在网格中心）                  Returns:             distance_cost: (H, W, 计算地形梯度代价                  Args:             height_map: (batch, H, W) - 高度图, 计算沿command方向的奖励（减少那些方向上的代价）                  逻辑：         - 从机器人位置开始，沿着command_ve, 初始化Teacher代价地图生成器                  Args:             grid_size: 网格尺寸 (height, wi, 从真实语义标签和高程图生成代价地图（Teacher模式）                  Args:             point_xyz: (batc (+1 more)

### Community 124 - "Community 124"
Cohesion: 0.27
Nodes (14): _canonical_joint_ids(), _finite_ordered_range(), ManagerBasedRLEnv, SceneEntityCfg, Tensor, Event functions for Go2 PVCNN environment.  This module contains event functions, Randomize only M1 leg positions and keep wheel/Panda state at defaults., Reset dynamic objects to random positions based on terrain origins.          Thi (+6 more)

### Community 125 - "Community 125"
Cohesion: 0.20
Nodes (10): LidarCfg, Configuration for the LiDAR sensor.          This configuration extends RayCaste, Configuration for the ray-cast sensor with dynamic mesh support.  This module de, Configuration for the ray-cast sensor with dynamic mesh support.          This e, Set the class type to LidarRayCaster after initialization., RayCasterCfg, Configuration for the Semantic LiDAR sensor.          This extends LidarCfg to a, Set the class type to SemanticLidarSensor after initialization. (+2 more)

### Community 126 - "Community 126"
Cohesion: 0.25
Nodes (13): filter_semantic_leaf_obstacle_paths(), ManagerBasedRLEnvCfg, TeacherElevationTrajectoryMpcSemanticEnvCfg, _counts_by_row_col(), main(), Path, run_probe(), _semantic_leaf_paths() (+5 more)

### Community 127 - "Community 127"
Cohesion: 0.14
Nodes (9): Tensor, Replay buffer for storing point cloud and semantic label data for PVCNN training, Return current buffer size., Clear all data from replay buffer., Args:             buffer_dir: Directory to store buffer data on disk, Add batch of samples to replay buffer.                  Args:             point_, Sample random batch from replay buffer.                  Args:             batch, Replay buffer for storing point cloud and semantic label pairs.     Used for asy (+1 more)

### Community 128 - "Community 128"
Cohesion: 0.26
Nodes (14): _base_command(), build_arg_parser(), checkpoint_iteration(), latest_checkpoint(), _load_manifest(), main(), ArgumentParser, Path (+6 more)

### Community 129 - "Community 129"
Cohesion: 0.31
Nodes (14): _atomic_write_json(), _audit_payload(), build_arg_parser(), build_pruning_plan(), execute_pruning(), main(), PruningItem, PruningPlan (+6 more)

### Community 131 - "Community 131"
Cohesion: 0.22
Nodes (13): base_xy_drift_l2(), Tensor, Lightweight reward helpers for M1 + Panda Teacher balance training., Penalize velocity only for the joints resolved by ``asset_cfg``., Penalize applied torque only for the joints resolved by ``asset_cfg``., Penalize the current trainable normalized residual amplitude., Penalize changes in the trainable normalized residual., Penalize squared horizontal displacement from each environment origin. (+5 more)

### Community 132 - "Community 132"
Cohesion: 0.18
Nodes (4): NeptuneLogger, NeptuneSummaryWriter, SummaryWriter, Summary writer for Neptune.

### Community 133 - "Community 133"
Cohesion: 0.27
Nodes (13): _apply_variant(), batch_forward_kinematics(), _command_sequences(), _fk_foot_from_result(), _joint_limit_stats(), main(), _max(), _mean() (+5 more)

### Community 134 - "Community 134"
Cohesion: 0.26
Nodes (10): _actor(), _FakeStorage, _ppo(), test_folded_load_ppo_configuration_is_exact_and_fresh(), test_invalid_kl_abort_threshold_is_rejected(), test_kl_uses_only_actor_active_dimensions(), test_next_update_resets_abort_diagnostics_and_can_complete(), test_none_kl_abort_threshold_preserves_legacy_behavior() (+2 more)

### Community 135 - "Community 135"
Cohesion: 0.24
Nodes (8): _cfg(), _configure_helper(), _reset_helper(), _Robot, _SceneEntityCfg, test_l0_l1_stage_config_is_deterministic(), test_leg_only_reset_writes_selected_env_and_preserves_wheel_panda_velocity(), test_stage_config_sets_exact_d1_d2_d3_ranges_and_protected_zeros()

### Community 136 - "Community 136"
Cohesion: 0.24
Nodes (10): _FakeTerminationManager, _load_script(), test_format_play_stats_rejects_invalid_wrench(), test_format_play_stats_reports_wrench_axes_and_unavailable_terms(), test_full_scale_play_requires_a1_disturbance_finite_steps_and_summary(), test_play_cli_rejects_invalid_base_checkpoint_combinations(), test_play_cli_rejects_missing_or_out_of_range_values(), test_play_script_reuses_exact_teacher_task_and_wrapper_boundaries() (+2 more)

### Community 137 - "Community 137"
Cohesion: 0.26
Nodes (9): Extension modules for trajectory-guided teacher experiments., attach_trajectory_manager(), _command_name(), _command_term(), _env_root(), install_trajectory_manager_hooks(), Attach helpers for the active MPC trajectory manager., _wrap_command_hook() (+1 more)

### Community 138 - "Community 138"
Cohesion: 0.21
Nodes (8): CostMapGenerator, Tensor, Cost map generation from point cloud semantic segmentation., 计算离机器人的距离代价（机器人在网格中心）                  Args:             H: 网格高度             W:, 计算地形梯度代价（仅计算陡峭度，不考虑符号）                  Args:             height_map: (batch, H,, 初始化代价地图生成器                  Args:             grid_size: 网格尺寸 (height, width)，默认, 从点云语义分割和高程图生成3通道代价地图                  Args:             point_xyz: (batch, num_p, 生成2D代价地图，用于2D CNN策略网络输入          从3D点云语义分割结果投影到2D网格，计算三种代价:     1. Distance cost

### Community 139 - "Community 139"
Cohesion: 0.19
Nodes (8): PVCNNWrapper, Tensor, Hook to capture layer activations., Extract features from point cloud data using batched processing., Extract features from a single batch of point cloud data.                  Args:, Wrapper for PVCNN model to extract features from point cloud data.          This, Get the dimension of extracted features., 初始化 PVCNN 包装器。                  参数:             checkpoint_path: 训练好的PVCNN check

### Community 140 - "Community 140"
Cohesion: 0.22
Nodes (7): LidarSensorData, RayCasterData, Data container for the LiDAR sensor.          This data container extends RayCas, The sensor data object., Initializes the LiDAR sensor.          Args:             cfg: The configuration, Data container for the Semantic LiDAR sensor.          This extends LidarSensorD, SemanticLidarData

### Community 141 - "Community 141"
Cohesion: 0.29
Nodes (12): discover_latest_checkpoint(), Path, Return the checkpoint with the largest numeric iteration suffix., _evaluate(), main(), Path, _run(), _stage1_running() (+4 more)

### Community 142 - "Community 142"
Cohesion: 0.17
Nodes (9): aggregate_small_collision_rounds(), build_controlled_crossing_commands(), TrackingRoundAccumulator, test_aggregate_small_collision_rounds_uses_env_denominator(), test_controlled_crossing_accumulator_records_reset_stage_after_foot_over(), test_controlled_crossing_accumulator_summarizes_success_by_speed_and_lateral(), test_controlled_crossing_commands_group_speed_and_lateral_offsets(), test_tracking_accumulator_averages_mean_and_valid_ratio_but_maxes_p95() (+1 more)

### Community 143 - "Community 143"
Cohesion: 0.33
Nodes (9): FakeExecutor, _load(), _prefix(), Path, test_non_l0_start_requires_complete_accepted_prefix(), test_parent_sha_must_match_previous_final_and_manifest(), test_rejected_stage_stops_and_keeps_previous_accepted_checkpoint(), test_successful_execution_advances_in_order_and_passes_parent_manifest() (+1 more)

### Community 144 - "Community 144"
Cohesion: 0.21
Nodes (10): Enum, str, Return inclusive-exclusive row bands for S1..S4 using quarter splits., Map a terrain row to its semantic-course stage., Choose one stable representative row per semantic stage., representative_rows(), SemanticCourseStage, stage_for_row() (+2 more)

### Community 145 - "Community 145"
Cohesion: 0.20
Nodes (3): SummaryWriter, Summary writer for Weights and Biases., WandbSummaryWriter

### Community 146 - "Community 146"
Cohesion: 0.33
Nodes (9): get_m1_panda_coordinated_train_cfg(), Return a fresh PPO config matched to the 200 Hz coordinated task., _get_cfg(), test_coordinated_cfg_freezes_200_hz_time_horizon_and_adaptive_ppo(), test_coordinated_cfg_returns_independent_objects(), _load_script(), Path, test_build_manifest_contract_freezes_ppo_dr_and_guard() (+1 more)

### Community 147 - "Community 147"
Cohesion: 0.36
Nodes (7): Tensor, One-environment deployable schedule, EE target, and nominal command., StudentS1Mission, _reset(), test_student_mission_has_five_phases_and_deployable_nominal_commands(), test_student_mission_rejects_wrong_reset_and_skipped_steps(), test_student_mission_reset_is_seed_repeatable_and_instances_are_isolated()

### Community 148 - "Community 148"
Cohesion: 0.29
Nodes (10): bad_orientation(), base_height(), ManagerBasedRLEnv, SceneEntityCfg, Tensor, Termination functions for Go2 PVCNN locomotion., Terminate when the robot's orientation is too far from upright., Terminate when the robot's base is too low. (+2 more)

### Community 149 - "Community 149"
Cohesion: 0.27
Nodes (8): Small explicit supervised-training configuration for Student S1., StudentTrainCfg, build_arg_parser(), main(), ArgumentParser, Path, _sha256(), train()

### Community 150 - "Community 150"
Cohesion: 0.22
Nodes (5): EmpiricalNormalization, Normalize mean and variance of values based on empirical values., Initialize EmpiricalNormalization module.          Args:             shape (int, Normalize mean and variance of values based on empirical values.          Args:, Learn input values without computing the output values of them

### Community 151 - "Community 151"
Cohesion: 0.31
Nodes (10): build_arg_parser(), main(), ArgumentParser, Path, rank_completed_rows(), Resolve immutable sweep inputs before creating any artifact., Run one strict Play child and require its JSON artifact., Validate, aggregate, and rank completed row artifacts. (+2 more)

### Community 152 - "Community 152"
Cohesion: 0.27
Nodes (6): _load_script(), test_c1a_summary_exposes_rolling_balance_and_qp_gates(), test_c1a_validate_args_rejects_invalid_shape_and_step_ranges(), test_formal_hard_gates_reject_good_tracking_with_bad_balance(), test_settling_gate_requires_minimum_steps_and_continuous_stability(), test_settling_gate_times_out_instead_of_scoring_unstable_state()

### Community 153 - "Community 153"
Cohesion: 0.22
Nodes (6): DirectRLEnv, ManagerBasedRLEnv, Returns the base environment of the wrapper., Modifies the action space to the clip range., Initialize the wrapper.                  Note:             The wrapper calls res, Inject PVCNN wrapper into the unwrapped environment.

### Community 154 - "Community 154"
Cohesion: 0.36
Nodes (9): attach_trajectory_manager_if_enabled(), _base_output(), _cuda_memory(), main(), _manager_counters(), _parse_args(), Namespace, Path (+1 more)

### Community 156 - "Community 156"
Cohesion: 0.27
Nodes (5): Isaac Lab :class:`RayCasterCfg` plus per-mesh semantic ids for multi-mesh height, SemanticGridRayCasterCfg, RayCasterData, Ray caster data plus elevation and semantic grids (Isaac ``height_scan``-style +, SemanticGridRayCasterData

### Community 157 - "Community 157"
Cohesion: 0.20
Nodes (6): Tensor, The episode length buffer., Set the episode length buffer.                  Note:             This is needed, Reset the environment.                  Returns:             A tuple of (obs, ex, Returns the current observations of the environment.                  Returns:, Step the environment.                  Returns:             A tuple of (obs, rew

### Community 158 - "Community 158"
Cohesion: 0.36
Nodes (9): atomic_write_json(), build_arg_parser(), build_manifest_contract(), initialize_fresh_zero_action_policy(), main(), Path, Make the safe implicit-actuator hold the fresh policy's exact baseline., Build the finite schema-2 contract before Isaac Sim is launched. (+1 more)

### Community 159 - "Community 159"
Cohesion: 0.33
Nodes (9): build_arg_parser(), main(), _parse_leg_actions(), _parse_stds(), _parse_wheel_actions(), ArgumentParser, Tensor, _run_one() (+1 more)

### Community 160 - "Community 160"
Cohesion: 0.24
Nodes (6): _collect_follow_camera_debug(), semantic_small_force_matrix_w(), SmallCollisionRoundAccumulator, update_small_collision_accumulator_from_env(), _usd_camera_world_position(), test_small_collision_accumulator_counts_each_env_once_per_round()

### Community 161 - "Community 161"
Cohesion: 0.27
Nodes (3): _attach_step_timing_probe(), Compact per-env-step timer for locating slow 4096 rollout stages., _StepTimingProbe

### Community 162 - "Community 162"
Cohesion: 0.20
Nodes (8): _fake_base_env(), _FakeSim, Tensor, test_apply_keyboard_command_overwrites_base_velocity_tensor(), test_viewer_apply_reset_snapshot_restores_root_and_joint_state(), test_viewer_ground_robot_from_scanner_shifts_root_z_to_match_ground(), test_viewer_step_mode_paused_loop_keeps_rendering_window(), test_viewer_zero_base_command_clears_command_tensor()

### Community 163 - "Community 163"
Cohesion: 0.42
Nodes (8): foot_acceleration_smoothness_loss(), foot_boundary_smoothness_loss(), foot_smoothness_loss(), Tensor, Smoothness losses for root and feet., root_smoothness_loss(), _weighted_mean(), test_mpc_foot_trajectory_regularization_penalizes_boundary_and_acceleration_spikes()

### Community 164 - "Community 164"
Cohesion: 0.25
Nodes (8): M1TransverseBarTerrainCfg, ndarray, Deterministic transverse-bar terrain for M1 crossing curricula., Create a flat tile with one full-width low bar in front of spawn., Parameters for a single transverse obstacle bar., transverse_bar_terrain(), SubTerrainBaseCfg, Trimesh

### Community 165 - "Community 165"
Cohesion: 0.44
Nodes (8): Linear, _last_actor_linear(), _model(), test_inactive_actions_means_log_prob_entropy_and_gradients_are_masked(), test_inactive_actor_rows_start_and_remain_exactly_zero_after_optimizer_step(), test_inference_and_checkpoint_round_trip_keep_inactive_rows_zero(), test_invalid_active_action_masks_are_rejected(), test_legacy_no_mask_behavior_and_checkpoint_keys_are_preserved()

### Community 166 - "Community 166"
Cohesion: 0.56
Nodes (8): _format_vec(), main(), _manual_bilinear_scanner_height(), _nearest_scanner_height(), Tensor, run_probe(), _summarize_gap(), _world_to_scanner_local()

### Community 167 - "Community 167"
Cohesion: 0.36
Nodes (8): fake_run(), _load_module(), pruner(), Path, test_dry_run_does_not_unlink_and_apply_writes_hashes(), test_plan_keeps_3500_and_selects_only_numeric_models_above_it(), test_plan_rejects_symlinks_and_paths_outside_exact_run(), test_postcondition_rejects_remaining_checkpoint_above_limit()

### Community 171 - "Community 171"
Cohesion: 0.25
Nodes (5): create_pvcnn_wrapper(), PVCNN Model Wrapper for Inference., Factory function to create PVCNN wrapper., Wrapper modules for Go2 PVCNN environment., Wrapper for Isaac Lab environments to work with RSL-RL PPO algorithm with PVCNN

### Community 172 - "Community 172"
Cohesion: 0.39
Nodes (7): configure_folded_load_stage(), Apply the exact reset/material ranges for one curriculum stage., _atomic_json(), build_arg_parser(), evaluate_probe(), main(), Path

### Community 173 - "Community 173"
Cohesion: 0.25
Nodes (3): 从扁平化观测中提取CNN特征并与本体感知拼接                  Args:             observations: (batch,, 根据观测选择动作                  Args:             observations: (batch, num_obs) - 扁平化, 评估价值函数                  Args:             critic_observations: (batch, num_criti

### Community 174 - "Community 174"
Cohesion: 0.43
Nodes (6): _assignment(), _class(), _keywords(), _sha256(), test_wbc_environment_exposes_one_exact_ordered_effort_action_and_c0_timing(), test_wbc_registration_is_lazy_and_does_not_change_legacy_teacher_sources()

### Community 175 - "Community 175"
Cohesion: 0.32
Nodes (3): _FakeEnv, _FakeSim, test_update_camera_accepts_requires_grad_tensors()

### Community 176 - "Community 176"
Cohesion: 0.57
Nodes (6): _all_asset_paths(), _independent_mount_parent_local_pos(), _load_cleanup_helper(), main(), _mount_patch_top_z(), _visible_bottom_z()

### Community 178 - "Community 178"
Cohesion: 0.38
Nodes (3): _good_diagnostics(), _load_probe(), test_probe_gate_requires_finite_exact_mask_fold_effort_margin_and_mount_response()

### Community 179 - "Community 179"
Cohesion: 0.33
Nodes (5): M1SmokeCurriculumCfg, M1SmokeTerminationsCfg, Minimal M1 smoke environment.  This config intentionally avoids the Go2-specific, Conservative smoke-test terminations., No curriculum terms for the first M1 smoke environment.

### Community 183 - "Community 183"
Cohesion: 0.60
Nodes (5): _load_script(), _summary(), test_rank_completed_rows_selects_lexicographic_winner(), test_run_play_child_rejects_nonzero_exit_and_missing_row(), test_validate_sweep_inputs_rejects_duplicates_missing_seeds_and_existing_dir()

### Community 184 - "Community 184"
Cohesion: 0.60
Nodes (5): _source(), test_train_exposes_keep_std_resume_option(), test_train_exposes_mpc_num_envs_cli_override(), test_train_preserves_livestream_flag_before_applauncher_mutates_args(), test_train_single_env_livestream_installs_follow_camera_only_for_env0()

### Community 185 - "Community 185"
Cohesion: 0.60
Nodes (4): l2_norm(), Tensor, safe_mean(), zero_per_env()

### Community 186 - "Community 186"
Cohesion: 0.40
Nodes (4): M1PandaSmokeObservationsCfg, PolicyCfg, ObsGroup, M1-only proprioception; Panda joints stay outside policy observations.

### Community 187 - "Community 187"
Cohesion: 0.50
Nodes (4): build_student_observation(), Tensor, Frozen 100-value deployable observation assembly for Student S1., StudentObservationParts

### Community 188 - "Community 188"
Cohesion: 0.40
Nodes (3): Returns the observation space., Returns the action space., Space

### Community 189 - "Community 189"
Cohesion: 0.40
Nodes (5): Quantify the user-visible final-frame foot jump and airborne stance symptom., _terminal_foot_anomaly_metrics(), test_terminal_foot_anomaly_metrics_flag_last_frame_jump_and_airborne_stance(), test_terminal_foot_anomaly_metrics_flag_mid_trajectory_jump(), test_terminal_foot_anomaly_metrics_flag_replan_boundary_jump()

### Community 190 - "Community 190"
Cohesion: 0.60
Nodes (3): _Cfg, _install_isaaclab_asset_stubs(), test_m1_asset_cfg_uses_floating_usd_and_16_controlled_joints()

### Community 194 - "Community 194"
Cohesion: 0.50
Nodes (3): ISAACLAB_SITE, PYTHONPATH, run_m1_roll_stage05_forward_long.sh script

### Community 195 - "Community 195"
Cohesion: 0.50
Nodes (3): ISAACLAB_SITE, PYTHONPATH, run_m1_roll_stage0_6s_long.sh script

### Community 200 - "Community 200"
Cohesion: 0.83
Nodes (3): _source(), test_runner_load_can_keep_checkpoint_std_when_requested(), test_runner_load_defaults_to_resetting_checkpoint_std()

## Knowledge Gaps
- **11 isolated node(s):** `create_urdf.sh script`, `ROS_PACKAGE_PATH`, `run_m1_contactfree_policy_play.sh script`, `run_m1_contactfree_policy_train.sh script`, `run_m1_front_obstacle_play.sh script` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_device()` connect `Community 9` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 6`, `Community 8`, `Community 14`, `Community 142`, `Community 18`, `Community 19`, `Community 21`, `Community 26`, `Community 30`, `Community 38`, `Community 40`, `Community 45`, `Community 49`, `Community 53`, `Community 54`, `Community 58`, `Community 63`, `Community 68`, `Community 72`, `Community 74`, `Community 86`, `Community 94`, `Community 96`, `Community 103`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `downsampled_elevation_semantic_scan()` connect `Community 5` to `Community 8`, `Community 16`, `Community 22`, `Community 90`, `Community 95`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `require_tensor()` connect `Community 72` to `Community 32`, `Community 35`, `Community 9`, `Community 12`, `Community 17`, `Community 18`, `Community 147`, `Community 23`, `Community 27`, `Community 28`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `MpcPlannerTerrain` (e.g. with `FkCollisionMargins` and `NominalCommandShapeDiagnostics`) actually correct?**
  _`MpcPlannerTerrain` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 116 inferred relationships involving `RuntimeError` (e.g. with `.current_frame_ids()` and `.current_reference()`) actually correct?**
  _`RuntimeError` has 116 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `MpcPlannerCfg` (e.g. with `MpcReferenceParticipationCfg` and `MpcTerrainDifficultyPair`) actually correct?**
  _`MpcPlannerCfg` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `SemanticObstacleCount` (e.g. with `M1PvcnnCrossing100mmEnvCfg` and `M1PvcnnCrossing60mmContactFreePlayEnvCfg`) actually correct?**
  _`SemanticObstacleCount` has 53 INFERRED edges - model-reasoned connections that need verification._