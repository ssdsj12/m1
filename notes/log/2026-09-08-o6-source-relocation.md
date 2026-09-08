# O6 source relocation

- Parent: [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
- Baseline: `dbcaddb`; candidate: working tree.
- Moved the external vendor directory into the main M1 checkout's `o6asset/`.
  Copied the same vendor tree into this linked worktree's `o6asset/` so it can
  be versioned and relocated with the development branch.
- [Normalizer](../../Go2Pvcnn/scripts/normalize_o6_assets.py) now defaults to
  repository-relative source and project-relative destination paths computed
  from its own file location; explicit CLI overrides remain supported.
- Verification: temporary-directory normalization from the new path produced
  exactly the existing source manifest, including all 35 SHA256 entries.
- Existing simulation assets were not regenerated; no controller change.
- GitHub upload is not part of this relocation operation.
