# O6 vendor source assets

This directory contains the original O6 vendor USD, URDF and mesh files.
Treat vendor files as read-only; normalization outputs belong to
`Go2Pvcnn/assets/m1_dual_panda_o6/`.

From the repository root, using the project's Python environment:

```bash
python Go2Pvcnn/scripts/normalize_o6_assets.py
```

Both default paths resolve relative to the script, independently of the shell
working directory. Explicit path overrides remain supported.

Simulation continues to load the assembled asset
`Go2Pvcnn/assets/m1_dual_panda_o6/m1_dual_panda_o6.usd`.
Original URDF package names and relative USD dependencies are preserved.
