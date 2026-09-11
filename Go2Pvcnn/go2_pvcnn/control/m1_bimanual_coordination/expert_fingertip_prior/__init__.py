"""Offline contracts for the frozen DexManipNet fingertip-motion prior."""

from .contracts import (
    DEXMANIPNET_REVISION,
    FINGER_ORDER,
    LEFT_REFLECTION,
    MANIPTRANS_COMMIT,
    MIXTURE_COMPONENTS,
    MODEL_INPUT_DIM,
    PHASE_ORDER,
    PRIOR_DT,
    PRIOR_HORIZON,
    STUDENT_ARTIFACT_FORMAT_VERSION,
    ExpertWindow,
    MixtureDistribution,
    PriorPhase,
    StudentArtifactMetadata,
)
from .sources import SOURCE_HANDS, SourceHandSpec

__all__ = [
    "DEXMANIPNET_REVISION",
    "FINGER_ORDER",
    "LEFT_REFLECTION",
    "MANIPTRANS_COMMIT",
    "MIXTURE_COMPONENTS",
    "MODEL_INPUT_DIM",
    "PHASE_ORDER",
    "PRIOR_DT",
    "PRIOR_HORIZON",
    "SOURCE_HANDS",
    "STUDENT_ARTIFACT_FORMAT_VERSION",
    "ExpertWindow",
    "MixtureDistribution",
    "PriorPhase",
    "SourceHandSpec",
    "StudentArtifactMetadata",
]
