"""Shared routing questions/criteria for native CLEF and Qwen fallback."""

from __future__ import annotations

from typing import Any

from radar.config.interests import RadarProfile
from radar.prompts.catalog import screening_questions


def build_questions(profile: RadarProfile) -> dict[str, Any]:
    """The same named questions, instructions, and criteria for both models."""
    keywords = ", ".join(profile.keywords[:8]) or "machine learning"
    domains = ", ".join(profile.domains[:4]) or "behavioral science"
    return screening_questions().render(keywords=keywords, domains=domains)
