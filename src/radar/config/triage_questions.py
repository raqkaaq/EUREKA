"""Shared routing questions/criteria for native CLEF and Qwen fallback."""

from __future__ import annotations

from typing import Any

from radar.config.interests import RadarProfile


def build_questions(profile: RadarProfile) -> dict[str, Any]:
    """The same named questions, instructions, and criteria for both models."""
    keywords = ", ".join(profile.keywords[:8]) or "machine learning"
    domains = ", ".join(profile.domains[:4]) or "behavioral science"
    return {
        "ai_ml_relevance": {
            "type": "noul",
            "instructions": f"Is this paper relevant to AI/ML ({keywords})?",
            "criteria": {
                "true": "The paper is relevant to AI/ML research or methods.",
                "false": "The paper is not relevant to AI/ML.",
            },
        },
        "cross_domain_potential": {
            "type": "noul",
            "instructions": (
                "Does this paper show cross-domain transfer potential between "
                f"AI/ML and {domains} (either direction)?"
            ),
            "criteria": {
                "true": "The paper connects AI/ML with behavioral or economic ideas.",
                "false": "No such cross-domain connection.",
            },
        },
    }
