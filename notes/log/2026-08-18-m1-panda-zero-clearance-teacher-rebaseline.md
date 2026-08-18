# 2026-08-18 M1 + Panda 零间隙 Teacher 重基线验收

## Authority

- Accepted combined USD SHA-256: `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`
- Panda USD SHA-256: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- Source checksum manifest SHA-256: `e7c44f8cf461a9d90f981357fa36699e226e7ffba6c85bc1a51a946253af0abb`
- Generated checksum manifest SHA-256: `5057a3165a2f034dd9c7e544b62dfc7415763dfb7d3f11516242a77e5879a63e`
- Builder/runtime commit: `eba7906`
- GPU: NVIDIA GeForce RTX 5070, driver `580.159.03`; Isaac Sim `5.1`

## Asset acceptance

- Final static asset/smoke/WBC triple: `37 passed`。
- Full C0+C1a 12-file regression: `184 passed in 2.02s`。
- PXR: one root `/M1Panda/BASE_LINK`, fixed mount, local parent plane error `0.0 m`。
- Local visible mount top: `0.0780230313539505 m`。
- Panda visible bottom: `0.07802216708660126 m`。
- Surface gap: `-8.642673492431641e-07 m`，within `±1e-6 m`。
- CPU and relocated-tree verifier: `25 DOF`, no dependency violations or validation errors。
- One-step mount relative delta: `2.398501783318352e-05 m < 1e-4 m`。
- Visual: user explicitly confirmed attachment after the global-max `50.976 mm` gap was corrected。

## GPU0 C0

- Exit `0`, `steps=2000`, `exit_reason=steps_complete`, finite。
- QP `2000/2000 = 1.0`; TRACK `2000/2000`。
- Maximum EE error `0.0002551754676521742 m`; maximum lateral slip `0.0009811028139665723 m/s`。
- Minimum singular value `0.1836902975459393`。
- Limit, base contact, self collision, reset and arm snap counts all zero。

## GPU0 C1a without Panda target motion

- Exit `0`, `hard_gates_passed=true`, five phases each `800` steps。
- QP `4000/4000 = 1.0`; TRACK `4000/4000`; minimum wheel contact count `4`。
- EE error `0.0016525656448521282 m`; rolling residual `0.0016347041429006625 m/s`; lateral slip `0.0011881585264347507 m/s`。
- Forward/reverse displacement `0.5807474023089404/-0.18901617288686418 m`。
- Zero hold-or-worse, limit, base contact, reset, arm snap, wheel saturation and direction mismatch events。

## GPU0 combined C1a

- Exit `0`, `hard_gates_passed=true`, five phases each `800` steps。
- QP `4000/4000 = 1.0`; TRACK `4000/4000`; minimum wheel contact count `4`。
- EE error `0.0016529180293131272 m`; rolling residual `0.0016649047778608131 m/s`; lateral slip `0.0012495935569394058 m/s`。
- Maximum roll/pitch `0.0012477257987484336/0.0004640368861146271 rad`。
- Zero hold-or-worse, limit, base contact, reset, arm snap, wheel saturation and direction mismatch events。

## Decision

All zero-clearance asset and unchanged C0/C1a Teacher gates pass. T400.6 is complete. The exact accepted combined-asset SHA above is the only asset authority for Student S1 manifests.

Student S1 implementation is unlocked under the approved [design](../../docs/superpowers/specs/2026-08-18-m1-panda-zero-clearance-dagger-student-s1-design.md) and [plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md). Scope remains flat ground, five longitudinal phases, small Panda motion and online DAgger; no random wrench, turning, terrain, PPO, grasping or real hardware.

## Links

- [Runbook](../../docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)
- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [Runtime/visual gate](2026-08-18-m1-panda-zero-clearance-runtime-visual-gates.md)
