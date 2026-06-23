"""Numerical helpers used by the document classifier."""

from __future__ import annotations

import math
from typing import Mapping


def normal_density(value: float, mean: float, std: float) -> float:
    if std <= 0:
        raise ValueError("standard deviation must be positive")
    z = (value - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def generated_probability(score: float, calibration: Mapping[str, float]) -> float:
    human = normal_density(score, calibration["human_mean"], calibration["human_std"])
    generated = normal_density(
        score,
        calibration["generated_mean"],
        calibration["generated_std"],
    )
    total = human + generated
    if total == 0:
        return 0.5
    return generated / total


def logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def probability_label(probability: float) -> str:
    return "AI" if probability > 0.5 else "Human"


def probability_band(probability: float) -> str:
    if probability >= 0.9:
        return "very_likely_ai"
    if probability >= 0.7:
        return "likely_ai"
    if probability > 0.5:
        return "leaning_ai"
    if probability > 0.3:
        return "leaning_human"
    if probability > 0.1:
        return "likely_human"
    return "very_likely_human"
