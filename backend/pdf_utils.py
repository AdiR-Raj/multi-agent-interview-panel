"""PDF Extraction Utilities using PyMuPDF (fitz).

Extracts text from PDF documents page by page while preserving page boundaries
and metadata for evidence traceability.
"""

from pathlib import Path
from typing import Union, Optional
import pymupdf

from backend.models import DocumentPage, ExtractedDocument


def extract_pdf_text(
    file_path: Union[str, Path],
    source_type: str,
    document_id: Optional[str] = None,
) -> ExtractedDocument:
    """Extracts text page by page from a PDF file using PyMuPDF.

    Args:
        file_path: Absolute or relative path to the target PDF file.
        source_type: Category ('job_description', 'resume', 'transcript').
        document_id: Optional unique identifier. Defaults to filename stem.

    Returns:
        ExtractedDocument containing all extracted pages and metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If file is not a valid PDF or is empty/corrupt.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    filename = path.name
    doc_id = document_id or f"doc_{path.stem}"

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open/parse PDF '{filename}': {exc}") from exc

    try:
        pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed

            # Extract raw text from page
            page_text = page.get_text("text") or ""
            # Normalize whitespace while preserving line structure
            normalized_text = "\n".join(
                line.strip() for line in page_text.splitlines() if line.strip()
            )

            pages.append(
                DocumentPage(
                    page_number=page_num,
                    text=normalized_text,
                )
            )

        return ExtractedDocument(
            document_id=doc_id,
            source_type=source_type,
            filename=filename,
            pages=pages,
        )
    finally:
        doc.close()


def extract_pdf_from_bytes(
    pdf_bytes: bytes,
    filename: str,
    source_type: str,
    document_id: Optional[str] = None,
) -> ExtractedDocument:
    """Extracts text from in-memory PDF bytes.

    Useful for processing uploaded files from FastAPI endpoints.
    """
    if not pdf_bytes:
        raise ValueError(f"Cannot extract from empty bytes for '{filename}'")

    doc_id = document_id or f"doc_{Path(filename).stem}"

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF bytes for '{filename}': {exc}") from exc

    try:
        pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            page_text = page.get_text("text") or ""
            normalized_text = "\n".join(
                line.strip() for line in page_text.splitlines() if line.strip()
            )

            pages.append(
                DocumentPage(
                    page_number=page_num,
                    text=normalized_text,
                )
            )

        return ExtractedDocument(
            document_id=doc_id,
            source_type=source_type,
            filename=filename,
            pages=pages,
        )
    finally:
        doc.close()
