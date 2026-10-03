"""Pydantic data models for the OpenAlex radar collector."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUERIES = 6
MAX_PER_PAGE = 50
MAX_KEYWORDS = 20
MAX_TERM_CHARS = 300


class RadarProfile(BaseModel):
    """AI/ML-centered cross-domain interest profile (validated input)."""

    keywords: list[str] = Field(min_length=1, max_length=MAX_KEYWORDS)
    domains: list[str] = Field(default_factory=list, max_length=MAX_KEYWORDS)
    lookback_days: int = Field(default=90, ge=1, le=3650)

    @field_validator("keywords", "domains", mode="before")
    @classmethod
    def _strip_nonempty(cls, v: object) -> object:
        if isinstance(v, list):
            cleaned = [t.strip() for t in v if isinstance(t, str) and t.strip()]
            return cleaned
        return v


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


# ---------------------------------------------------------------------------
# Structured analysis output (PydanticAI output_type; URLs attached later)
# ---------------------------------------------------------------------------

MAX_OPPORTUNITIES = 5
MAX_EVIDENCE_PER_OPP = 3


class OpportunityDraft(BaseModel):
    """One model-proposed opportunity. No URLs: evidence is attached
    deterministically after the model run via ``evidence`` indices."""

    title: str = Field(min_length=1, max_length=300)
    wow: str = Field(default="", max_length=2000)
    investigate: str = Field(default="", max_length=2000)
    reproduce: str = Field(default="", max_length=2000)
    evidence: list[int] = Field(default_factory=list, max_length=MAX_EVIDENCE_PER_OPP)


class RadarDraft(BaseModel):
    """Structured PydanticAI result for a batch of candidates."""

    opportunities: list[OpportunityDraft] = Field(
        default_factory=list, max_length=MAX_OPPORTUNITIES
    )
    ignore: list[str] = Field(default_factory=list, max_length=10)
    next_move: str = Field(default="", max_length=2000)

    @field_validator("ignore", mode="before")
    @classmethod
    def _coerce_ignore(cls, v: object) -> object:
        """Coerce non-string ignore items (models emit ints/nulls).

        Mirrors the downstream ``str()`` coercion in evidence attachment:
        nulls are dropped, the rest stringified. Live-evidenced 2026-10-03
        (``ignore.0: string_type`` first-try validation failures).
        """
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return v


class EvidenceLink(BaseModel):
    """Deterministically attached candidate reference (never model-written)."""

    index: int = Field(ge=0)
    title: str = Field(default="")
    url: str = Field(default="", max_length=2000)
    openalex_id: str = Field(default="", max_length=500)


class Opportunity(BaseModel):
    """Opportunity draft plus deterministically attached evidence links."""

    draft: OpportunityDraft
    evidence_links: list[EvidenceLink] = Field(default_factory=list)


class RadarReport(BaseModel):
    """Final report: analyzed opportunities with evidence + triage."""

    opportunities: list[Opportunity] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    next_move: str = Field(default="")
