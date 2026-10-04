"""Opportunity contracts: analysis drafts, evidence, and final reports.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or any other service dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from radar.config.runtime import MAX_EVIDENCE_PER_OPP, MAX_OPPORTUNITIES
from radar.schema.learning import LearningDossierDraft


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


class ResolvedLearningDossier(BaseModel):
    """Primary dossier plus its deterministically attached abstract source.

    ``evidence_level`` is code-owned (``Literal["abstract"]``, never
    model-writable): the draft has no such field and ``extra="forbid"``
    rejects any model-supplied value.
    """

    model_config = ConfigDict(extra="forbid")

    dossier: LearningDossierDraft
    source: EvidenceLink
    evidence_level: Literal["abstract"] = "abstract"

    @model_validator(mode="after")
    def _matching_source(self) -> ResolvedLearningDossier:
        if self.source.index != self.dossier.paper_index or not self.source.openalex_id:
            raise ValueError("Learning dossier must reference its primary candidate.")
        return self


class LearningRadarDraft(RadarDraft):
    """Final synthesis draft: opportunity hypotheses plus one primary dossier.

    Used ONLY for final research-team synthesis. ``learning_dossiers`` holds
    at most one primary dossier; the default empty list accepts legacy
    opportunity-only drafts (including old test doubles) without requiring
    learning output.
    """

    model_config = ConfigDict(extra="forbid")

    learning_dossiers: list[LearningDossierDraft] = Field(
        default_factory=list, max_length=1
    )

    @field_validator("learning_dossiers", mode="after")
    @classmethod
    def _unique_primary(
        cls, value: list[LearningDossierDraft]
    ) -> list[LearningDossierDraft]:
        indices = [dossier.paper_index for dossier in value]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate primary dossier indices")
        return value


class Opportunity(BaseModel):
    """Opportunity draft plus deterministically attached evidence links."""

    draft: OpportunityDraft
    evidence_links: list[EvidenceLink] = Field(default_factory=list)


class RadarReport(BaseModel):
    """Final report: analyzed opportunities with evidence + triage."""

    opportunities: list[Opportunity] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    next_move: str = Field(default="")
    learning_dossiers: list[ResolvedLearningDossier] = Field(
        default_factory=list, max_length=1
    )


__all__ = [
    "EvidenceLink",
    "LearningRadarDraft",
    "Opportunity",
    "OpportunityDraft",
    "RadarDraft",
    "RadarReport",
    "ResolvedLearningDossier",
]
