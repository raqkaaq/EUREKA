"""Triage screening contracts (service-free: pydantic only).

CLEF/SystemOne per-paper screening outcomes. Probabilities are strict
0..1 floats: bools, strings, NaN/infinities, and out-of-range values are
rejected. A ``scored`` result carries both probabilities; every other
status carries none, so unknown works stay explicitly unknown.
"""

from __future__ import annotations

import math as _math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RUBRIC_VERSION = "clef-triage-v1"

TriageStatus = Literal[
    "scored", "missing_abstract", "failed", "deadline", "oversized"
]


class TriageResult(BaseModel):
    """One work's screening outcome (model-set fields only)."""

    work_id: str = Field(min_length=1, max_length=500)
    status: TriageStatus
    ai_ml_relevance: float | None = None
    cross_domain_potential: float | None = None

    @field_validator(
        "ai_ml_relevance", "cross_domain_potential", mode="before"
    )
    @classmethod
    def _strict_probability(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("probability must be a JSON number, not bool/string")
        number = float(value)
        if not _math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("probability must be finite within 0..1")
        return number

    @model_validator(mode="after")
    def _status_probability_agreement(self) -> "TriageResult":
        probs = (self.ai_ml_relevance, self.cross_domain_potential)
        if self.status == "scored":
            if any(p is None for p in probs):
                raise ValueError("scored results require both probabilities")
        elif any(p is not None for p in probs):
            raise ValueError(f"{self.status} results must not carry probabilities")
        return self


class TriageBatch(BaseModel):
    """Whole-pool screening outcome with model/rubric provenance."""

    model_id: str = Field(min_length=1, max_length=200)
    rubric_version: str = Field(min_length=1, max_length=200)
    backend: Literal["clef", "qwen"] = "clef"
    probability_kind: Literal["native_noul"] = "native_noul"
    fallback_reason: Literal[
        "missing_endpoint", "screening_failed", "deadline", "screening_error"
    ] | None = None
    results: list[TriageResult] = Field(default_factory=list)
