"""Command-line entry point for local detector runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Detector


def _read_input(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return " ".join(args.text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Turkish text for AI-generation signals.")
    parser.add_argument("text", nargs="*", help="Text to analyze when --file is not used.")
    parser.add_argument("--file", help="UTF-8 text file to analyze.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Hide per-segment details in the top-level output.",
    )
    args = parser.parse_args()

    source_text = _read_input(args)
    if not source_text:
        parser.error("provide text arguments or --file")

    report = Detector().analyze(source_text, include_segments=not args.no_segments)
    if args.json:
        print(report.to_json(include_segments=not args.no_segments))
        return

    print(f"AI score   : {report.ai_score:.1f}%")
    print(f"Label      : {report.label}")
    print(f"Confidence : {report.confidence:.2%}")
    print(f"Segments   : {report.segment_count}")


if __name__ == "__main__":
    main()
