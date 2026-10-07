"""Learning dossier contracts: one primary source-grounded study dossier.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or any other service dependency. The model writes
:class:`LearningDossierDraft` (candidate index plus interpretation);
code attaches :class:`ResolvedLearningDossier` deterministically
(see :mod:`radar.processing.evidence`). ``evidence_level`` is code-owned:
it never appears on the model-writable draft (``extra="forbid"` rejects it)
and defaults to ``"abstract"`` on the resolved record.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

StudyTaskKind = Literal[
    "derivation", "counterexample", "replication", "comparison", "full_text_check"
]


class StudyTask(BaseModel):
    """One follow-up task with an explicit missing-evidence note."""

    model_config = ConfigDict(extra="forbid")

    kind: StudyTaskKind
    objective: str = Field(min_length=1)
    success_criterion: str = Field(min_length=1)
    missing_evidence: str = Field(min_length=1)

    @field_validator("objective", "success_criterion", "missing_evidence")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Study task content must not be blank.")
        return value


class LearningDossierDraft(BaseModel):
    """Model-proposed primary dossier for one candidate (index, not URL).

    The provider bounds output tokens, not individual narrative fields. ``connections``
    must each be explicitly labeled as hypotheses (``Hypothesis: ...`` prefix);
    ``significance`` is the model's interpretation, never a reported finding.
    No theorem, proof, or full-text results may be fabricated; source coverage
    and provenance are attached and validated by the caller.
    """

    model_config = ConfigDict(extra="forbid")

    paper_index: StrictInt = Field(ge=0)
    supporting_pages: list[StrictInt] = Field(default_factory=list, max_length=8)
    source_passage_ids: list[str] = Field(default_factory=list)
    core_problem: str = Field(min_length=1)
    reported_contribution: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    significance: str = Field(min_length=1)
    assumptions_limits: list[str] = Field(min_length=1, max_length=4)
    connections: list[str] = Field(max_length=3)
    study_tasks: list[StudyTask] = Field(min_length=1, max_length=2)
    open_questions: list[str] = Field(min_length=1, max_length=3)

    @field_validator("source_passage_ids")
    @classmethod
    def _passage_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("Source passage IDs must be nonblank and unique.")
        return value

    @field_validator("core_problem", "reported_contribution", "reasoning", "significance")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Learning dossier content must not be blank.")
        return value

    @field_validator("assumptions_limits", "open_questions")
    @classmethod
    def _nonblank_items(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("Dossier list items must not be blank.")
        return value

    @field_validator("connections")
    @classmethod
    def _labeled_hypotheses(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("Dossier list items must not be blank.")
            if not item.strip().lower().startswith("hypothesis:"):
                raise ValueError("connections must be explicitly labeled hypotheses")
        return value


__all__ = ["LearningDossierDraft", "StudyTask", "StudyTaskKind"]
