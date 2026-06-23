"""Dataclasses and aggregation logic for document-level detector output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Mapping, Sequence

from . import defaults
from .probability import logistic


@dataclass(frozen=True)
class SegmentAnalysis:
    index: int
    text: str
    words: int
    sentence_count: int
    curve_score: float
    ai_probability: float
    label: str
    confidence: float
    band: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisReport:
    ai_score: float
    label: str
    confidence: float
    raw_curve_score: float
    segment_count: int
    ai_segment_count: int
    human_segment_count: int
    word_count: int
    features: dict[str, float]
    segments: list[SegmentAnalysis] = field(default_factory=list)

    def to_dict(self, include_segments: bool = True) -> dict:
        payload = asdict(self)
        if not include_segments:
            payload["segments"] = []
        return payload

    def to_json(self, include_segments: bool = True, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_segments=include_segments),
            ensure_ascii=False,
            indent=indent,
        )


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def extract_document_features(segments: Sequence[SegmentAnalysis]) -> dict[str, float]:
    if not segments:
        return {
            "score_mean": 0.0,
            "score_std": 0.0,
            "score_high": 0.0,
            "score_low": 0.0,
            "prob_mean": 0.5,
            "prob_std": 0.0,
            "prob_high": 0.5,
            "prob_low": 0.5,
            "generated_share": 0.0,
            "segment_total": 0.0,
        }

    scores = [segment.curve_score for segment in segments]
    probabilities = [segment.ai_probability for segment in segments]
    ai_count = sum(1 for segment in segments if segment.label == "AI")

    return {
        "score_mean": sum(scores) / len(scores),
        "score_std": _population_std(scores),
        "score_high": max(scores),
        "score_low": min(scores),
        "prob_mean": sum(probabilities) / len(probabilities),
        "prob_std": _population_std(probabilities),
        "prob_high": max(probabilities),
        "prob_low": min(probabilities),
        "generated_share": ai_count / len(segments),
        "segment_total": float(len(segments)),
    }


def score_document(
    segments: Sequence[SegmentAnalysis],
    weights: Mapping[str, float] = defaults.DOCUMENT_WEIGHTS,
    include_segments: bool = True,
) -> AnalysisReport:
    features = extract_document_features(segments)
    linear_value = weights["bias"]
    for name, value in features.items():
        linear_value += weights[name] * value

    ai_probability = logistic(linear_value)
    ai_score = round(ai_probability * 100.0, 1)
    label = "AI" if ai_score > 70 else "Mixed" if ai_score > 40 else "Human"
    confidence = round(abs(ai_probability - 0.5) * 2.0, 4)

    total_words = sum(segment.words for segment in segments)
    if total_words:
        raw_curve_score = sum(segment.curve_score * segment.words for segment in segments) / total_words
    else:
        raw_curve_score = 0.0

    ai_segments = sum(1 for segment in segments if segment.label == "AI")
    report_segments = list(segments) if include_segments else []

    return AnalysisReport(
        ai_score=ai_score,
        label=label,
        confidence=confidence,
        raw_curve_score=round(raw_curve_score, 6),
        segment_count=len(segments),
        ai_segment_count=ai_segments,
        human_segment_count=len(segments) - ai_segments,
        word_count=total_words,
        features={key: round(value, 6) for key, value in features.items()},
        segments=report_segments,
    )
