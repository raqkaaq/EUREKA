"""Canonical screening input shared by native CLEF and Qwen chat routing."""

from __future__ import annotations

from typing import Any

from radar.config.interests import RadarProfile
from radar.config.triage_questions import build_questions
from radar.schema.papers import CollectedWork


def build_input(work: CollectedWork, profile: RadarProfile, model: str) -> dict[str, Any]:
    """Full title/abstract and identical questions, not a rewritten task."""
    return {"model": model, "state": {
        "title": work.title or "(untitled)", "abstract": work.abstract,
        "publication_year": work.publication_year,
    }, "questions": build_questions(profile)}
