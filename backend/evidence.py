"""Deterministic Evidence Extraction and Storage Layer.

Constructs grounded evidence records (E001, E002, ...) directly from extracted
PDF text (with standard whitespace normalization) and provides lookup and
formatting utilities.
"""

from typing import List, Dict, Optional
import re

from backend.models import ExtractedDocument, EvidenceItem, EvidenceReference


def extract_paragraphs(text: str) -> List[str]:
    """Splits extracted page text into meaningful non-empty paragraphs and sections.

    Extracts text grounded in the source document without paraphrasing,
    handling paragraphs, bullet points, headers, and dialogue turns while
    standardizing line-break whitespace.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []

    def flush_current():
        if current_chunk:
            joined = " ".join(current_chunk).strip()
            if len(joined) >= 4:
                chunks.append(joined)
            current_chunk.clear()

    bullet_chars = {'•', '-', '*', '●', '○', '■', '▪', '\u2022', '\u25cf', '\u25cb', '\u25a0', '\u25aa'}
    bullet_pattern = re.compile(r"^([•\-\*\u2022\u25cf\u25cb\u25a0\u25aa]|\d+[\.\)])\s+")
    dialogue_pattern = re.compile(r"^(Q\d+(\s*\([^)]*\))?|A\d+(\s*\([^)]*\))?|Interviewer|Candidate|Panelist|Speaker):", re.IGNORECASE)
    header_pattern = re.compile(r"^(Summary|Experience|Projects|Skills|Education|Requirements|Responsibilities|About the Role|Technical Section|Culture & Work Style|Culture Section|Behavioral Section|What You'll Do|What We're Looking For|What This Role Is NOT)$", re.IGNORECASE)

    for line in lines:
        if line in bullet_chars:
            flush_current()
            current_chunk.append(line)
        elif bullet_pattern.match(line) or dialogue_pattern.match(line) or header_pattern.match(line) or (line.endswith(":") and len(line) < 40 and current_chunk):
            flush_current()
            current_chunk.append(line)
        else:
            current_chunk.append(line)

    flush_current()
    return chunks


class EvidenceStore:
    """In-memory store for grounded evidence items generated across documents."""

    def __init__(self, id_prefix: str = "E", start_index: int = 1):
        self.id_prefix = id_prefix
        self.current_index = start_index
        self._evidence: Dict[str, EvidenceItem] = {}
        self._by_source: Dict[str, List[str]] = {
            "job_description": [],
            "resume": [],
            "transcript": [],
        }
        self._by_doc_id: Dict[str, List[str]] = {}

    def _next_evidence_id(self) -> str:
        eid = f"{self.id_prefix}{self.current_index:03d}"
        self.current_index += 1
        return eid

    def add_document(self, doc: ExtractedDocument, context_prefix: Optional[str] = None) -> List[EvidenceItem]:
        """Extracts deterministic evidence items from an ExtractedDocument and registers them."""
        new_items: List[EvidenceItem] = []

        for page in doc.pages:
            paragraphs = extract_paragraphs(page.text)
            for idx, paragraph in enumerate(paragraphs):
                eid = self._next_evidence_id()
                
                # Context provides helpful location hint (e.g., 'resume (p. 1, section 2)')
                ctx = f"{doc.source_type.replace('_', ' ').title()}: {doc.filename} (p. {page.page_number})"
                if context_prefix:
                    ctx = f"{context_prefix} - {ctx}"

                item = EvidenceItem(
                    evidence_id=eid,
                    source_type=doc.source_type,
                    filename=doc.filename,
                    page_number=page.page_number,
                    quote=paragraph,
                    context=ctx,
                )

                self._evidence[eid] = item
                self._by_source.setdefault(doc.source_type, []).append(eid)
                self._by_doc_id.setdefault(doc.document_id, []).append(eid)
                new_items.append(item)

        return new_items

    def get_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Retrieves a specific evidence item by its ID."""
        return self._evidence.get(evidence_id)

    def get_by_source(self, source_type: str) -> List[EvidenceItem]:
        """Returns all evidence items for a given source type."""
        eids = self._by_source.get(source_type, [])
        return [self._evidence[eid] for eid in eids if eid in self._evidence]

    def get_by_document(self, document_id: str) -> List[EvidenceItem]:
        """Returns all evidence items belonging to a document ID."""
        eids = self._by_doc_id.get(document_id, [])
        return [self._evidence[eid] for eid in eids if eid in self._evidence]

    def get_all(self) -> List[EvidenceItem]:
        """Returns all stored evidence items in deterministic insertion order."""
        return list(self._evidence.values())

    def get_all_references(self) -> List[EvidenceReference]:
        """Returns all stored evidence as traceable EvidenceReferences."""
        return [item.to_reference() for item in self._evidence.values()]

    def format_for_prompt(self, source_type: Optional[str] = None) -> str:
        """Formats evidence items into a clean textual list suitable for LLM injection."""
        items = self.get_by_source(source_type) if source_type else self.get_all()
        if not items:
            return "No evidence recorded."

        formatted_lines = []
        for item in items:
            formatted_lines.append(
                f"[{item.evidence_id}] ({item.source_type} | {item.filename} p.{item.page_number}): \"{item.quote}\""
            )
        return "\n".join(formatted_lines)

    def clear(self):
        """Resets the store and evidence ID counter."""
        self.current_index = 1
        self._evidence.clear()
        self._by_source = {"job_description": [], "resume": [], "transcript": []}
        self._by_doc_id.clear()
