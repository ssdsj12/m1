"""Detached, strict-JSON evidence; never consumed by the control path."""

from dataclasses import fields, is_dataclass
from enum import Enum
import math

import torch


def diagnostic_snapshot(value):
    """Copy tensors/dataclasses to JSON values, spelling nonfinite numbers explicitly.

    Diagnostic NaN/Inf are strings (NaN, Infinity, -Infinity), never valid
    numeric measurements. This preserves their identity without nonstandard JSON.
    """
    if isinstance(value, torch.Tensor):
        return diagnostic_snapshot(value.detach().cpu().tolist())
    if is_dataclass(value):
        return {field.name: diagnostic_snapshot(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {key: diagnostic_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_snapshot(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def capture_arm_attempt(sample, solution, planner):
    """Capture routed input and actual solver evidence before dual-arm fallback.

    Row numbers match the backend: six horizon-sized inequality blocks followed
    by finite upper then lower variable bounds. All Arm MPC bounds are finite.
    Residual is signed lhs - upper; positive values violate the <= constraint.
    Node and joint indices are zero based. A solver failure need not imply a
    violated row (e.g. nonfinite KKT output); retain raw status in that case.
    """
    problem = getattr(planner, "last_qp_problem", None)
    result = getattr(planner, "last_qp_result", None)
    rows = []
    if problem is not None and result is not None:
        dimension = problem.gradient.numel()
        lhs = torch.cat((problem.inequality_matrix @ result.solution,
                         result.solution, -result.solution))
        upper = torch.cat((problem.inequality_upper, problem.upper_bound,
                           -problem.lower_bound))
        families = ("position", "position", "velocity", "velocity",
                    "effort", "effort", "acceleration", "acceleration")
        active = set(result.active_set)
        for row in range(upper.numel()):
            block, variable = divmod(row, dimension)
            node, joint = divmod(variable, sample.q.numel())
            rows.append({
                "row": row, "family": families[block],
                "bound": "upper" if block % 2 == 0 else "lower",
                "node": node, "joint": joint, "active": row in active,
                "lhs": float(lhs[row]), "upper": float(upper[row]),
                "residual": float(lhs[row] - upper[row]),
            })
    return diagnostic_snapshot({
        "input": sample, "config": getattr(planner, "cfg", None),
        "diagnostics": solution.diagnostics,
        "qp_problem": problem, "qp_result": result,
        "constraint_rows": rows,
        "qp_evidence_available": problem is not None and result is not None,
    })
