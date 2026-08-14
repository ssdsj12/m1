"""Runtime setup for the repository-local PVCNN CUDA extension toolchain."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path


def _prepend(environ: MutableMapping[str, str], name: str, value: Path) -> None:
    value_str = str(value)
    current = environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if value_str not in entries:
        environ[name] = os.pathsep.join((value_str, *entries))


def configure_pvcnn_cuda(
    workspace: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Use the local CUDA development prefix when CUDA_HOME is not already set."""
    environ = os.environ if environ is None else environ
    if environ.get("CUDA_HOME"):
        return False

    cuda_root = Path(workspace).resolve() / ".cuda-nvcc-12.8"
    nvcc = cuda_root / "bin/nvcc"
    if not nvcc.is_file():
        return False

    target_root = cuda_root / "targets/x86_64-linux"
    environ["CUDA_HOME"] = str(cuda_root)
    environ["CUDACXX"] = str(nvcc)
    _prepend(environ, "PATH", cuda_root / "bin")
    _prepend(environ, "CPATH", target_root / "include")
    _prepend(environ, "LIBRARY_PATH", target_root / "lib")
    _prepend(environ, "LD_LIBRARY_PATH", cuda_root / "lib")
    _prepend(environ, "LD_LIBRARY_PATH", target_root / "lib")
    environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return True
