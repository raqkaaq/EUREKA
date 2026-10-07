"""Opportunity contracts: analysis drafts, evidence, and final reports.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or any other service dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from radar.config.runtime import MAX_EVIDENCE_PER_OPP, MAX_OPPORTUNITIES
from radar.schema.learning import LearningDossierDraft
from radar.schema.documents import PDFSource, PDFReading, DocumentFailure, PDFPassage


class OpportunityDraft(BaseModel):
    """One model-proposed opportunity. No URLs: evidence is attached
    deterministically after the model run via ``evidence`` indices."""

    title: str = Field(min_length=1, max_length=300)
    wow: str = ""
    investigate: str = ""
    reproduce: str = ""
    evidence: list[StrictInt] = Field(default_factory=list, max_length=MAX_EVIDENCE_PER_OPP)


class RadarDraft(BaseModel):
    """Structured PydanticAI result for a batch of candidates."""

    opportunities: list[OpportunityDraft] = Field(
        default_factory=list, max_length=MAX_OPPORTUNITIES
    )
    ignore: list[str] = Field(default_factory=list, max_length=10)
    next_move: str = ""

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
    """Primary dossier plus its deterministically attached source basis.

    ``evidence_level`` is code-owned (abstract or PDF text, never
    model-writable): the draft has no such field and ``extra="forbid"``
    rejects any model-supplied value.
    """

    model_config = ConfigDict(extra="forbid")

    dossier: LearningDossierDraft
    source: EvidenceLink
    evidence_level: Literal["abstract", "pdf_text"] = "abstract"
    source_pdf: PDFSource | None = None
    source_pages: list[StrictInt] = Field(default_factory=list, max_length=8)
    source_passages: list[PDFPassage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _matching_source(self) -> ResolvedLearningDossier:
        if self.source.index != self.dossier.paper_index or not self.source.openalex_id:
            raise ValueError("Learning dossier must reference its primary candidate.")
        if [p.passage_id for p in self.source_passages] != self.dossier.source_passage_ids:
            raise ValueError("Consulted passages must match the dossier's source references.")
        if self.evidence_level == "pdf_text":
            if self.source_pdf is None or self.source_pdf.work_id != self.source.openalex_id or not self.source_pages:
                raise ValueError("PDF evidence requires its own source and page references.")
            if self.source_pages != self.dossier.supporting_pages:
                raise ValueError("Resolved PDF pages must match the model's validated references.")
            if self.source_passages:
                if any(p.work_id != self.source_pdf.work_id or p.sha256 != self.source_pdf.sha256
                       for p in self.source_passages):
                    raise ValueError("Consulted passages must belong to the primary PDF source.")
                if set(self.source_pages) != {p.page for p in self.source_passages}:
                    raise ValueError("Supporting pages must match consulted source passages.")
        elif self.source_pdf is not None or self.source_pages or self.dossier.supporting_pages or self.source_passages:
            raise ValueError("Abstract evidence cannot claim PDF page coverage.")
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
    learning_question: str = ""
    learning_dossiers: list[ResolvedLearningDossier] = Field(
        default_factory=list, max_length=1
    )
    document_readings: list[PDFReading] = Field(default_factory=list, max_length=25)
    document_failures: list[DocumentFailure] = Field(default_factory=list, max_length=25)

    @model_validator(mode="after")
    def _pdf_provenance(self) -> RadarReport:
        readings = {r.work_id: r for r in self.document_readings}
        if len(readings) != len(self.document_readings):
            raise ValueError("Duplicate full-document reading.")
        failed = [r.work_id for r in self.document_failures]
        if len(set(failed)) != len(failed) or set(failed) & set(readings):
            raise ValueError("Document outcome must be unambiguous.")
        for item in self.learning_dossiers:
            if item.evidence_level != "pdf_text":
                continue
            reading = readings.get(item.source.openalex_id)
            if reading is None or item.source_pdf != PDFSource.model_validate(
                reading.model_dump(include=set(PDFSource.model_fields))
            ):
                raise ValueError("PDF dossier requires its matching completed reading.")
            if any(p < 1 or p > reading.page_count for p in item.source_pages):
                raise ValueError("PDF reference names an unavailable page.")
            verified_pages = ({p.page for p in item.source_passages} if item.source_passages
                              else {e.page for e in reading.notes.evidence})
            if not set(item.source_pages) <= verified_pages:
                raise ValueError("PDF dossier must cite supplied verified evidence pages.")
        return self


__all__ = [
    "EvidenceLink",
    "LearningRadarDraft",
    "Opportunity",
    "OpportunityDraft",
    "RadarDraft",
    "RadarReport",
    "ResolvedLearningDossier",
]
