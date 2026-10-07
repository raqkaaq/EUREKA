"""Read exact page-local passages from the supplied PDF cohort, without I/O."""

from collections.abc import Sequence

from radar.config.documents import DOCUMENT_CHUNK_CHARS
from radar.schema.documents import PDFDocument, PDFPassage


class PDFPassages:
    """A run-local ledger of passages actually served to investigation."""

    def __init__(self, documents: Sequence[PDFDocument]):
        self._documents = tuple(documents)
        self._served: dict[str, PDFPassage] = {}

    def read(self, paper_index: int, page: int, offset: int = 0) -> PDFPassage:
        if any(type(value) is not int for value in (paper_index, page, offset)):
            raise ValueError("Paper index, page and offset must be strict integers.")
        if not 0 <= paper_index < len(self._documents):
            raise ValueError("Use a paper index from the supplied cohort.")
        document = self._documents[paper_index]
        if not 1 <= page <= len(document.pages):
            raise ValueError("Use a page from the supplied paper.")
        text = document.pages[page - 1].text
        if not 0 <= offset < len(text):
            raise ValueError("Use an offset within the supplied page.")
        end = min(offset + DOCUMENT_CHUNK_CHARS, len(text))
        passage = PDFPassage(
            passage_id=f"{document.work_id.rsplit('/', 1)[-1]}:{document.sha256}:p{page}:{offset}-{end}",
            work_id=document.work_id, sha256=document.sha256, page=page,
            start=offset, end=end, text=text[offset:end],
            next_offset=end if end < len(text) else None)
        self._served[passage.passage_id] = passage
        return passage

    @property
    def served(self) -> tuple[PDFPassage, ...]:
        return tuple(self._served.values())

    def resolve(self, ids: list[str], work_id: str) -> tuple[PDFPassage, ...]:
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Cite nonempty unique passage IDs returned by read_pdf_passage.")
        passages = []
        for passage_id in ids:
            passage = self._served.get(passage_id)
            if passage is None or passage.work_id != work_id:
                raise ValueError("Cite only consulted passages from the primary paper.")
            passages.append(passage)
        return tuple(passages)
