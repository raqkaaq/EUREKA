"""Interest profile: AI/ML core with behavioral/economic cross-domain lenses.

Service-free (pydantic/stdlib only). Single home for profile construction,
keyword-override cleaning, and lookback validation.

The profile centers on AI/ML methods and deliberately pairs them with
behavioral-science and economics domains so discovery surfaces
cross-domain transfer opportunities (e.g. RL ideas applied to mechanism
design, LLM evaluation borrowing from psychometrics).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from radar.config.runtime import MAX_KEYWORDS

DEFAULT_LOOKBACK_DAYS = 90
MAX_LOOKBACK_DAYS = 3650

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


class RadarProfile(BaseModel):
    """AI/ML-centered cross-domain interest profile (validated input)."""

    keywords: list[str] = Field(min_length=1, max_length=MAX_KEYWORDS)
    domains: list[str] = Field(default_factory=list, max_length=MAX_KEYWORDS)
    lookback_days: int = Field(default=DEFAULT_LOOKBACK_DAYS, ge=1, le=MAX_LOOKBACK_DAYS)

    @field_validator("keywords", "domains", mode="before")
    @classmethod
    def _strip_nonempty(cls, v: object) -> object:
        if isinstance(v, list):
            cleaned = [t.strip() for t in v if isinstance(t, str) and t.strip()]
            return cleaned
        return v


def validate_lookback_days(value: int) -> int:
    """Recency window in days, 1..3650."""
    try:
        days = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--lookback-days must be within 1..3650") from exc
    if not 1 <= days <= MAX_LOOKBACK_DAYS:
        raise ValueError("--lookback-days must be within 1..3650")
    return days


def clean_keyword_override(keywords: list[str] | None) -> list[str] | None:
    """Clean a CLI keyword override; None when absent, error when empty."""
    if not keywords:
        return None
    cleaned = [k.strip() for k in keywords if k and k.strip()]
    if not cleaned:
        raise ValueError("--keywords must contain at least one non-empty term")
    return cleaned[:MAX_KEYWORDS]


def default_profile(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> RadarProfile:
    """Return the default local AI/ML + behavioral/economic profile."""
    return RadarProfile(
        keywords=list(AI_ML_KEYWORDS),
        domains=list(CROSS_DOMAIN_TERMS),
        lookback_days=validate_lookback_days(lookback_days),
    )


__all__ = [
    "AI_ML_KEYWORDS",
    "CROSS_DOMAIN_TERMS",
    "DEFAULT_LOOKBACK_DAYS",
    "MAX_LOOKBACK_DAYS",
    "RadarProfile",
    "clean_keyword_override",
    "default_profile",
    "validate_lookback_days",
]
