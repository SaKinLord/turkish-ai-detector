"""High-level detector pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping, Protocol

from . import defaults
from .curvature import CurvatureScorer, ModelRuntime
from .document import AnalysisReport, SegmentAnalysis, score_document
from .probability import generated_probability, probability_band, probability_label
from .segmentation import SegmentPolicy, TextSegment, build_segments, wrap_existing_segments


class TextScorer(Protocol):
    def score(self, text: str) -> float:
        ...


class Detector:
    """Analyze Turkish text with the calibrated Config A runtime."""

    def __init__(
        self,
        model_name: str = defaults.MODEL_NAME,
        segment_policy: SegmentPolicy | None = None,
        calibration: Mapping[str, float] = defaults.CURVE_CALIBRATION,
        document_weights: Mapping[str, float] = defaults.DOCUMENT_WEIGHTS,
        samples: int = defaults.TOKEN_SAMPLE_COUNT,
        token_limit: int = defaults.TOKEN_LIMIT,
        sampling_seed: int | None = defaults.TOKEN_SAMPLING_SEED,
        scorer: TextScorer | None = None,
        lazy_model: bool = True,
    ) -> None:
        self.segment_policy = segment_policy or SegmentPolicy()
        self.calibration = dict(calibration)
        self.document_weights = dict(document_weights)

        if scorer is None:
            runtime = ModelRuntime(
                model_name=model_name,
                token_limit=token_limit,
                samples=samples,
                sampling_seed=sampling_seed,
            )
            scorer = CurvatureScorer(runtime=runtime, lazy=lazy_model)
        self.scorer = scorer

    def analyze(self, text: str, include_segments: bool = True) -> AnalysisReport:
        if not text or not text.strip():
            raise ValueError("text must contain at least one non-space character")
        return self._analyze_segment_objects(
            build_segments(text, self.segment_policy),
            include_segments=include_segments,
        )

    def analyze_segments(
        self,
        segment_texts: Iterable[str],
        include_segments: bool = True,
    ) -> AnalysisReport:
        segments = wrap_existing_segments(segment_texts)
        if not segments:
            raise ValueError("segment_texts must contain at least one segment")
        return self._analyze_segment_objects(segments, include_segments=include_segments)

    def _analyze_segment_objects(
        self,
        segments: list[TextSegment],
        include_segments: bool,
    ) -> AnalysisReport:
        analyses: list[SegmentAnalysis] = []

        for segment in segments:
            curve_score = float(self.scorer.score(segment.text))
            ai_probability = float(generated_probability(curve_score, self.calibration))
            label = probability_label(ai_probability)
            analyses.append(
                SegmentAnalysis(
                    index=segment.index,
                    text=segment.text,
                    words=segment.words,
                    sentence_count=segment.sentence_count,
                    curve_score=curve_score,
                    ai_probability=ai_probability,
                    label=label,
                    confidence=round(abs(ai_probability - 0.5) * 2.0, 4),
                    band=probability_band(ai_probability),
                )
            )

        return score_document(
            analyses,
            weights=self.document_weights,
            include_segments=include_segments,
        )
