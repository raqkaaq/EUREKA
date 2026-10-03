"""Default local radar profile: AI/ML core with behavioral/economic
cross-domain discovery.

The profile centers on AI/ML methods and deliberately pairs them with
behavioral-science and economics domains so the collector surfaces
cross-domain transfer opportunities (e.g. RL ideas applied to mechanism
design, LLM evaluation borrowing from psychometrics).
"""

from __future__ import annotations

from radar.models import RadarProfile

AI_ML_KEYWORDS: tuple[str, ...] = (
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "large language models",
    "diffusion models",
)

# Behavioral / economic cross-domain lenses appended to semantic queries.
CROSS_DOMAIN_TERMS: tuple[str, ...] = (
    "behavioral science",
    "behavioral economics",
    "mechanism design",
)

DEFAULT_LOOKBACK_DAYS = 90


def default_profile(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> RadarProfile:
    """Return the default local AI/ML + behavioral/economic profile."""
    return RadarProfile(
        keywords=list(AI_ML_KEYWORDS),
        domains=list(CROSS_DOMAIN_TERMS),
        lookback_days=lookback_days,
    )
