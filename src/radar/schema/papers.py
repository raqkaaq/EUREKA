"""Paper contracts: OpenAlex works, locations, and query plans.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or anyother service dependency.
"""

from __future__ import annotations

from typing import Literal
import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from radar.config.runtime import (
    MAX_KEYWORDS,
    MAX_PER_PAGE,
    MAX_QUERIES,
    MAX_TERM_CHARS,
    MAX_SEMANTIC_CHARS,
)

SearchRole = Literal[
    "foundation", "frontier", "counterevidence", "cross_domain", "exploration"
]

CITATION_KINDS = frozenset({"references", "citations"})


class DiscoveryMatch(BaseModel):
    """Retrieval intent, not a claim about quality or conclusions."""

    question_id: str = Field(min_length=1, max_length=80)
    role: SearchRole


class PlannedQuery(BaseModel):
    """One bounded OpenAlex request derived from the profile."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["keyword", "semantic", "recent", "references", "citations"]
    terms: str = Field(default="", max_length=MAX_SEMANTIC_CHARS)
    per_page: StrictInt = Field(default=25, ge=1, le=MAX_PER_PAGE)
    from_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    question_id: str | None = Field(default=None, min_length=1, max_length=80)
    role: SearchRole | None = None
    seed_work_id: str | None = Field(
        default=None, max_length=500, pattern=r"^https://openalex\.org/W\d+$"
    )

    @model_validator(mode="after")
    def _search_contract(self) -> PlannedQuery:
        if self.from_date is not None:
            dt.date.fromisoformat(self.from_date)
        is_citation = self.kind in CITATION_KINDS
        if is_citation:
            if self.seed_work_id is None:
                raise ValueError("citation query requires seed_work_id")
        else:
            if self.seed_work_id is not None:
                raise ValueError("seed_work_id is only for references/citations")
            if not self.terms.strip():
                raise ValueError("non-citation query requires terms")
            if self.kind != "semantic" and len(self.terms) > MAX_TERM_CHARS:
                raise ValueError("keyword query exceeds its character bound")
            if self.kind == "recent" and self.from_date is None:
                raise ValueError("recent query requires from_date")
        if (self.question_id is None) != (self.role is None):
            raise ValueError("question identity and role must appear together")
        return self

    @field_validator("terms", mode="before")
    @classmethod
    def _strip_terms(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("seed_work_id", mode="before")
    @classmethod
    def _strip_seed(cls, v: object) -> object:
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return v


class QueryPlan(BaseModel):
    """Bounded list of planned OpenAlex queries."""

    model_config = ConfigDict(extra="forbid")

    queries: list[PlannedQuery] = Field(min_length=1, max_length=MAX_QUERIES)
    profile_summary: str = ""


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
    publication_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    doi: str = Field(default="", max_length=500)
    primary_url: str = Field(default="", max_length=2000)
    locations: list[LocationInfo] = Field(default_factory=list, max_length=5)
    cited_by_count: int = Field(default=0, ge=0)
    matched_queries: list[str] = Field(default_factory=list, max_length=MAX_QUERIES)
    query_kinds: list[str] = Field(default_factory=list, max_length=MAX_QUERIES)
    discovery_matches: list[DiscoveryMatch] = Field(default_factory=list, max_length=MAX_QUERIES)
    score: float = 0.0

    @field_validator("publication_date")
    @classmethod
    def _calendar_date(cls, value: str | None) -> str | None:
        if value is not None:
            dt.date.fromisoformat(value)
        return value


__all__ = [
    "CollectedWork",
    "LocationInfo",
    "PlannedQuery",
    "QueryPlan",
]
