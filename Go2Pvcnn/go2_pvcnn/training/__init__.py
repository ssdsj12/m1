"""Training-only guards and artifact helpers."""

from .m1_panda_arm_mpc_residual_promotion import (
    ENGINEERING_FLOORS,
    MetricComparison,
    PromotedCandidate,
    PromotionDecision,
    calibrate_tolerances,
    compare_with_tolerances,
    evaluate_candidate,
    select_promoted_candidate,
)

__all__ = [
    "ENGINEERING_FLOORS",
    "MetricComparison",
    "PromotedCandidate",
    "PromotionDecision",
    "calibrate_tolerances",
    "compare_with_tolerances",
    "evaluate_candidate",
    "select_promoted_candidate",
]
