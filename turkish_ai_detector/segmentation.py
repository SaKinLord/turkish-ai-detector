"""Turkish-aware text segmentation for detector inputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from . import defaults

_SPACE_RE = re.compile(r"\s+")
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞIİÖŞÜ])")


@dataclass(frozen=True)
class SegmentPolicy:
    """Word-count limits used while grouping sentence-like units."""

    min_words: int = defaults.SEGMENT_MIN_WORDS
    max_words: int = defaults.SEGMENT_MAX_WORDS
    min_sentence_words: int = defaults.SENTENCE_MIN_WORDS


@dataclass(frozen=True)
class TextSegment:
    """A model-scored unit derived from the source document."""

    index: int
    text: str
    words: int
    sentence_count: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "words": self.words,
            "sentence_count": self.sentence_count,
        }


def normalize_spacing(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip())


def split_turkish_sentences(text: str, min_words: int = defaults.SENTENCE_MIN_WORDS) -> list[str]:
    cleaned = normalize_spacing(text)
    if not cleaned:
        return []

    candidates = _BOUNDARY_RE.split(cleaned)
    return [
        sentence.strip()
        for sentence in candidates
        if sentence.strip() and len(sentence.split()) >= min_words
    ]


def _make_segment(index: int, sentences: Iterable[str]) -> TextSegment:
    sentence_list = list(sentences)
    text = " ".join(sentence_list).strip()
    return TextSegment(
        index=index,
        text=text,
        words=len(text.split()),
        sentence_count=len(sentence_list),
    )


def build_segments(text: str, policy: SegmentPolicy | None = None) -> list[TextSegment]:
    policy = policy or SegmentPolicy()
    sentences = split_turkish_sentences(text, min_words=policy.min_sentence_words)

    if not sentences:
        cleaned = normalize_spacing(text)
        return [TextSegment(index=0, text=cleaned, words=len(cleaned.split()), sentence_count=1)]

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        would_exceed = current_words + sentence_words > policy.max_words
        has_minimum = current_words >= policy.min_words

        if current and would_exceed and has_minimum:
            groups.append(current)
            current = [sentence]
            current_words = sentence_words
            continue

        current.append(sentence)
        current_words += sentence_words

    if current:
        current_words = sum(len(sentence.split()) for sentence in current)
        if groups and current_words < policy.min_words:
            groups[-1].extend(current)
        else:
            groups.append(current)

    return [_make_segment(index, group) for index, group in enumerate(groups)]


def wrap_existing_segments(texts: Iterable[str]) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for raw_text in texts:
        cleaned = normalize_spacing(raw_text)
        if not cleaned:
            continue
        sentence_count = max(1, len(split_turkish_sentences(cleaned, min_words=1)))
        segments.append(
            TextSegment(
                index=len(segments),
                text=cleaned,
                words=len(cleaned.split()),
                sentence_count=sentence_count,
            )
        )
    return segments
