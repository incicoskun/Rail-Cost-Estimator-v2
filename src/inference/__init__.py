"""Inference layer: prediction, FTA lookup, and explainability."""

from .explainer import SHAPExplainer
from .fta_lookup import FTALookup
from .service import InferenceService

__all__ = [
    "InferenceService",
    "SHAPExplainer",
    "FTALookup",
]
