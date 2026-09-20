"""Private float32 CUDA RTI correctness backend and fixed contracts.

The eager coupled LQ, line-search, and planner modules are correctness-only;
production GPU Play remains explicitly fail-closed until later runtime,
benchmark, and Isaac validation gates. These types do not replace the public
CPU float64 control contracts.
"""

from .config import GpuRtiCfg
from .contracts import GpuRtiTrajectory

__all__ = ["GpuRtiCfg", "GpuRtiTrajectory"]
