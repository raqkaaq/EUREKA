"""Hard bounds for PDF acquisition, extraction and complete-text investigation."""

PDF_MAX_BYTES = 20 * 1024 * 1024
PDF_MAX_PAGES = 80
PDF_MAX_TEXT_CHARS = 120_000
PDF_DOWNLOAD_TIMEOUT_S = 30.0
PDF_EXTRACTION_TIMEOUT_S = 20.0
PDF_EXTRACTION_MEMORY_BYTES = 512 * 1024 * 1024
PDF_MAX_REDIRECTS = 5
DOCUMENT_CHUNK_CHARS = 6000
DOCUMENT_MAX_CHUNKS = 24
DOCUMENT_MAX_NOTE_CHARS = 2500
DOCUMENT_MAX_EXCERPTS = 16
DOCUMENT_CHUNK_EXCERPT_CHARS = 100
DOCUMENT_REDUCTION_GROUP_SIZE = 4
DOCUMENT_ANALYSIS_TIMEOUT_S = 900.0
DOCUMENT_MAX_TIMEOUT_S = 3600.0
DOCUMENT_MAX_TOKENS = 1000
PDF_RESEARCH_PROMPT_CHARS = 48_000


def validate_document_timeout(value: float) -> float:
    import math

    message = "--document-timeout must be within (0, 3600] seconds"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(timeout) or not 0 < timeout <= DOCUMENT_MAX_TIMEOUT_S:
        raise ValueError(message)
    return timeout
