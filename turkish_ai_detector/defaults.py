"""Runtime constants for the Turkish AI detector.

The values in this file are the frozen Config A runtime configuration from
private calibration work. They are intentionally plain data so lightweight
modules can import them without loading the model stack.
"""

from __future__ import annotations

MODEL_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"

SEGMENT_MIN_WORDS = 15
SEGMENT_MAX_WORDS = 50
SENTENCE_MIN_WORDS = 3

TOKEN_SAMPLE_COUNT = 5
TOKEN_LIMIT = 512
TOKEN_SAMPLING_SEED = 42

CURVE_CALIBRATION = {
    "human_mean": -0.68480255859375,
    "human_std": 0.37893215449320466,
    "generated_mean": -0.316625703125,
    "generated_std": 0.29287661943451404,
}

DOCUMENT_WEIGHTS = {
    "bias": 3.5481491416,
    "score_mean": 1.5150501062,
    "score_std": -0.2879229087,
    "score_high": 0.9270197637,
    "score_low": 1.1226946130,
    "prob_mean": 0.9966877915,
    "prob_std": -0.3059411983,
    "prob_high": 0.1748479955,
    "prob_low": 0.4761605051,
    "generated_share": 1.5745822944,
    "segment_total": -0.0817152596,
}
