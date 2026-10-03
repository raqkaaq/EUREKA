"""Opportunity contracts: analysis drafts, evidence, and final reports.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or any other service dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictInt, field_validator

from radar.config.runtime import MAX_EVIDENCE_PER_OPP, MAX_OPPORTUNITIES


class OpportunityDraft(BaseModel):
    """One model-proposed opportunity. No URLs: evidence is attached
    deterministically after the model run via ``evidence`` indices."""

    title: str = Field(min_length=1, max_length=300)
    wow: str = Field(default="", max_length=2000)
    investigate: str = Field(default="", max_length=2000)
    reproduce: str = Field(default="", max_length=2000)
    evidence: list[StrictInt] = Field(default_factory=list, max_length=MAX_EVIDENCE_PER_OPP)


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


SpecialistRole = Literal["ml_methods", "behavioral_economics", "evidence_review"]


class SpecialistContribution(BaseModel):
    """A role-attributed model hypothesis, not additional scientific evidence."""

    role: SpecialistRole
    draft: RadarDraft


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


__all__ = [
    "EvidenceLink",
    "Opportunity",
    "OpportunityDraft",
    "RadarDraft",
    "RadarReport",
]
