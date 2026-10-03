"""Paper contracts: OpenAlex works, locations, and query plans.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or anyother service dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from radar.config.runtime import (
    MAX_KEYWORDS,
    MAX_PER_PAGE,
    MAX_QUERIES,
    MAX_TERM_CHARS,
)


class PlannedQuery(BaseModel):
    """One bounded OpenAlex request derived from the profile."""

    kind: Literal["semantic", "recent"]
    terms: str = Field(min_length=1, max_length=MAX_TERM_CHARS)
    per_page: int = Field(default=25, ge=1, le=MAX_PER_PAGE)
    from_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("terms", mode="before")
    @classmethod
    def _strip_terms(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


class QueryPlan(BaseModel):
    """Bounded list of planned OpenAlex queries."""

    queries: list[PlannedQuery] = Field(min_length=1, max_length=MAX_QUERIES)
    profile_summary: str = Field(default="", max_length=500)


class LocationInfo(BaseModel):
    """Normalized copy of one OpenAlex location entry."""

    source_name: str = Field(default="")
    landing_url: str = Field(default="", max_length=2000)
    pdf_url: str = Field(default="", max_length=2000)
    is_oa: bool = False


class CollectedWork(BaseModel):
    """Normalized, deduplicated OpenAlex work with provenance."""

    openalex_id: str = Field(min_length=1, max_length=500)
    title: str = Field(default="")
    abstract: str = Field(default="", max_length=20000)
    publication_year: int | None = Field(default=None, ge=1500, le=2100)
    doi: str = Field(default="", max_length=500)
    primary_url: str = Field(default="", max_length=2000)
    locations: list[LocationInfo] = Field(default_factory=list, max_length=5)
    cited_by_count: int = Field(default=0, ge=0)
    matched_queries: list[str] = Field(default_factory=list, max_length=MAX_QUERIES)
    query_kinds: list[str] = Field(default_factory=list, max_length=MAX_QUERIES)
    score: float = 0.0


__all__ = [
    "CollectedWork",
    "LocationInfo",
    "PlannedQuery",
    "QueryPlan",
]
