"""Public API for the Turkish AI detector runtime."""

from .document import AnalysisReport, SegmentAnalysis
from .pipeline import Detector
from .segmentation import SegmentPolicy, TextSegment

__all__ = [
    "AnalysisReport",
    "Detector",
    "SegmentAnalysis",
    "SegmentPolicy",
    "TextSegment",
]
