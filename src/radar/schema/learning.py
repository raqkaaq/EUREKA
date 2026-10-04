"""Learning dossier contracts: one primary abstract-level study dossier.

Service-free (pydantic only): importable without httpx, pydantic_ai,
or any other service dependency. The model writes
:class:`LearningDossierDraft` (candidate index plus compact interpretation);
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
    """One bounded follow-up task with an explicit missing-evidence note."""

    model_config = ConfigDict(extra="forbid")

    kind: StudyTaskKind
    objective: str = Field(min_length=1, max_length=300)
    success_criterion: str = Field(min_length=1, max_length=300)
    missing_evidence: str = Field(min_length=1, max_length=300)

    @field_validator("objective", "success_criterion", "missing_evidence")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Study task content must not be blank.")
        return value


class LearningDossierDraft(BaseModel):
    """Model-proposed primary dossier for one candidate (index, not URL).

    Text caps bound individual fields; the provider separately caps output tokens. ``connections``
    must each be explicitly labeled as hypotheses (``Hypothesis: ...`` prefix);
    ``significance`` is the model's interpretation, never a reported finding.
    No theorem, proof, or full-text results may be fabricated: abstract-level
    evidence only.
    """

    model_config = ConfigDict(extra="forbid")

    paper_index: StrictInt = Field(ge=0)
    core_problem: str = Field(min_length=1, max_length=500)
    reported_contribution: str = Field(min_length=1, max_length=500)
    reasoning: str = Field(min_length=1, max_length=500)
    significance: str = Field(min_length=1, max_length=500)
    assumptions_limits: list[str] = Field(min_length=1, max_length=4)
    connections: list[str] = Field(max_length=3)
    study_tasks: list[StudyTask] = Field(min_length=1, max_length=2)
    open_questions: list[str] = Field(min_length=1, max_length=3)

    @field_validator("core_problem", "reported_contribution", "reasoning", "significance")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Learning dossier content must not be blank.")
        return value

    @field_validator("assumptions_limits", "open_questions")
    @classmethod
    def _compact_items(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item) > 300:
                raise ValueError("dossier list items must be 1..300 chars")
        return value

    @field_validator("connections")
    @classmethod
    def _labeled_hypotheses(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item) > 300:
                raise ValueError("dossier list items must be 1..300 chars")
            if not item.strip().lower().startswith("hypothesis:"):
                raise ValueError("connections must be explicitly labeled hypotheses")
        return value


__all__ = ["LearningDossierDraft", "StudyTask", "StudyTaskKind"]
