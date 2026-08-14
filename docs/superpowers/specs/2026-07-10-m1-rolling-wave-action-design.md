# M1 rolling and wave action design

## Goal

Adapt the M1 smoke environment from one 16-joint position action to a hybrid wheel-leg action contract: 12 leg joints use position targets and 4 wheel joints use velocity targets. Add explicit rolling and wave mode parameters so flat rolling and obstacle-crossing wave behavior have stable configuration names.

## Scope

This is still a smoke/control-interface adaptation, not a complete learned locomotion policy or obstacle perception stack. The environment should expose the correct action split and mode parameters. A later task can connect these parameters to terrain sensing, command sampling, and policy rewards.

## Design

`go2_pvcnn.assets` will define:

- `M1_LEG_JOINT_NAMES`: all `ABAD`, `HIP`, and `KNEE` joints, 12 total.
- `M1_WHEEL_JOINT_NAMES`: the four `FOOT_JOINT` joints.
- `M1_ROLLING_MODE` and `M1_WAVE_MODE`: string constants for mode names.

`M1SmokeActionsCfg` will expose:

- `leg_pos = mdp.JointPositionActionCfg(...)` using `M1_LEG_JOINT_NAMES`.
- `wheel_vel = mdp.JointVelocityActionCfg(...)` using `M1_WHEEL_JOINT_NAMES`.

`M1SmokeEnvCfg` will expose conservative parameters:

- `control_mode = M1_ROLLING_MODE`
- `rolling_wheel_velocity = 4.0`
- `wave_wheel_velocity = 1.5`
- `wave_amplitude = 0.08`
- `wave_frequency = 1.0`
- `wave_phase_offsets = (0.0, 0.5, 0.5, 0.0)`

The wave values are configuration hooks for a stepping/wave controller. They do not claim full obstacle autonomy.
