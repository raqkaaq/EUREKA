"""PDF text, complete reading coverage and page-grounded model notes."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from radar.config.documents import (
    DOCUMENT_MAX_CHUNKS, DOCUMENT_MAX_EXCERPTS, PDF_MAX_PAGES, PDF_MAX_TEXT_CHARS,
)


class DocumentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PDFPage(DocumentModel):
    number: StrictInt = Field(ge=1, le=PDF_MAX_PAGES)
    text: str = Field(min_length=1, max_length=PDF_MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Page has no extractable text.")
        return value


class PDFSource(DocumentModel):
    work_id: str = Field(min_length=1, max_length=500)
    source_url: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    relative_path: str = Field(pattern=r"^pdf/W[0-9]+-[a-f0-9]{64}\.pdf$")

    @field_validator("source_url")
    @classmethod
    def _source_url(cls, value: str) -> str:
        from radar.processing.link_validation import is_http_link
        if not is_http_link(value):
            raise ValueError("PDF source must be a safe HTTP link.")
        return value

    @model_validator(mode="after")
    def _hash_path(self) -> PDFSource:
        if not self.relative_path.endswith(f"-{self.sha256}.pdf"):
            raise ValueError("PDF cache path does not match its content hash.")
        if not self.relative_path.startswith(f"pdf/{self.work_id.rsplit('/', 1)[-1]}-"):
            raise ValueError("PDF cache path does not match its work identity.")
        return self


class DocumentRecord(DocumentModel):
    """Immutable per-run outcome; extraction is not itself model investigation."""

    work_id: str = Field(min_length=1, max_length=500)
    status: Literal["extracted", "read", "failed"]
    source: PDFSource | None = None
    reading: PDFReading | None = None
    failure: DocumentFailure | None = None

    @model_validator(mode="after")
    def _outcome(self) -> DocumentRecord:
        if self.status == "failed":
            if self.failure is None or self.failure.work_id != self.work_id or self.reading is not None:
                raise ValueError("Failed document requires its own failure record.")
        elif self.source is None or self.source.work_id != self.work_id or self.failure is not None:
            raise ValueError("Successful extraction requires its own source provenance.")
        if self.status == "read":
            if self.reading is None or self.reading.work_id != self.work_id:
                raise ValueError("Completed document requires its full reading record.")
            if PDFSource.model_validate(self.reading.model_dump(include=set(PDFSource.model_fields))) != self.source:
                raise ValueError("Reading must use the recorded document source.")
        elif self.reading is not None:
            raise ValueError("Extraction is not completed investigation.")
        return self


class PDFDocument(PDFSource):
    pages: list[PDFPage] = Field(min_length=1, max_length=PDF_MAX_PAGES)
    extraction_warning: str = Field(
        default="Text-layer extraction only; figures, images and equation/table fidelity are not visually verified.",
        max_length=500,
    )

    @model_validator(mode="after")
    def _complete_pages(self) -> PDFDocument:
        if [p.number for p in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("Extracted page numbering must be complete and contiguous.")
        if sum(len(p.text) for p in self.pages) > PDF_MAX_TEXT_CHARS:
            raise ValueError("Document text exceeds its bound; never truncate it.")
        return self


class PageEvidence(DocumentModel):
    page: StrictInt = Field(ge=1, le=PDF_MAX_PAGES)
    quote: str = Field(min_length=1, max_length=180)
    finding: str = Field(min_length=1)

    @field_validator("quote", "finding")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence must not be blank.")
        return value


class DocumentNoteFields(DocumentModel):
    """Required narrative fields, without arbitrary content-length limits."""

    summary: str = Field(min_length=1)
    methods: str = Field(min_length=1)
    results: str = Field(min_length=1)
    limitations: str = Field(min_length=1)

    @field_validator("summary", "methods", "results", "limitations")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reading notes must not be blank.")
        return value


class DocumentNotes(DocumentNoteFields):
    evidence: list[PageEvidence] = Field(default_factory=list, max_length=2)


class EvidenceExcerpt(DocumentModel):
    """Code-owned registry entry; never generated by the model.

    Chunk excerpts are further limited to 100 characters by their producer.
    Reduction can reference existing PageEvidence quotes up to their original
    180-character bound, preserving the final stored contract.
    """

    excerpt_id: StrictInt = Field(ge=0, lt=DOCUMENT_MAX_EXCERPTS)
    page: StrictInt = Field(ge=1, le=PDF_MAX_PAGES)
    quote: str = Field(min_length=1, max_length=180)

    @field_validator("quote")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence excerpt must not be blank.")
        return value


class EvidenceReference(DocumentModel):
    excerpt_id: StrictInt = Field(ge=0, lt=DOCUMENT_MAX_EXCERPTS)
    finding: str = Field(min_length=1)

    @field_validator("finding")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence finding must not be blank.")
        return value


class ReferencedDocumentNotes(DocumentNoteFields):
    """Model-facing notes select strict IDs, not generated pages or quotes."""

    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _unique_references(self) -> ReferencedDocumentNotes:
        ids = [item.excerpt_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence excerpt references must be unique.")
        return self


class ChunkReading(DocumentModel):
    index: StrictInt = Field(ge=0, lt=DOCUMENT_MAX_CHUNKS)
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    pages: list[StrictInt] = Field(min_length=1, max_length=PDF_MAX_PAGES)
    notes: DocumentNotes


class PDFReading(PDFSource):
    page_count: StrictInt = Field(ge=1, le=PDF_MAX_PAGES)
    text_chars: StrictInt = Field(gt=0, le=PDF_MAX_TEXT_CHARS + PDF_MAX_PAGES * 40)
    chunks: list[ChunkReading] = Field(min_length=1, max_length=DOCUMENT_MAX_CHUNKS)
    notes: DocumentNotes
    evidence_level: Literal["pdf_text"] = "pdf_text"
    extraction_warning: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _complete_coverage(self) -> PDFReading:
        end = 0
        covered: set[int] = set()
        for index, chunk in enumerate(self.chunks):
            if chunk.index != index or chunk.start != end or chunk.end <= chunk.start:
                raise ValueError("Reading chunks must cover the entire text without gaps.")
            if any(p < 1 or p > self.page_count for p in chunk.pages):
                raise ValueError("Reading chunk names an unavailable page.")
            end = chunk.end
            covered.update(chunk.pages)
        if end != self.text_chars or covered != set(range(1, self.page_count + 1)):
            raise ValueError("Every extracted character and page must be read.")
        return self


class DocumentFailure(DocumentModel):
    work_id: str = Field(min_length=1, max_length=500)
    category: str = Field(min_length=1, max_length=100)


DocumentRecord.model_rebuild()
