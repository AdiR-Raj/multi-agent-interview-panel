import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

from backend.pdf_utils import extract_pdf_text, extract_pdf_from_bytes
from backend.evidence import EvidenceStore
from backend.models import ExtractedDocument, EvidenceItem


def create_sample_pdf(file_path: Path, title: str, content: str) -> Path:
    """Generates a synthetic sample PDF for testing extraction."""
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 50, 550, 750)
    page.insert_textbox(
        rect,
        f"{title}\n\n{content}",
        fontsize=11,
    )
    doc.save(str(file_path))
    doc.close()
    return file_path


def normalize_ws(text: str) -> str:
    """Normalizes whitespace sequences into single spaces for robust substring comparison."""
    return " ".join(text.split())


def run_stage2_verification():
    print("=" * 70)
    print("STAGE 2 VERIFICATION: PDF Extraction & Evidence Store")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Create 3 test PDFs
        jd_path = Path(temp_dir) / "job_description.pdf"
        create_sample_pdf(
            jd_path,
            "Job Description: Staff Backend Engineer",
            "Requirements:\n"
            "- 5+ years of experience with Python and distributed systems.\n"
            "- Strong expertise in API design and database optimizations.\n"
            "- Proven leadership and mentoring capabilities.",
        )

        resume_path = Path(temp_dir) / "resume_candidate_a.pdf"
        create_sample_pdf(
            resume_path,
            "Candidate A - Resume",
            "Experience:\n"
            "- Staff Engineer at CloudCorp (2020-Present): Scaled microservices to 50k RPS.\n"
            "- Senior Engineer at DataFlow (2017-2020): Led PostgreSQL migration.\n"
            "Skills: Python, FastAPI, Kubernetes, Distributed Systems, SQL",
        )

        transcript_path = Path(temp_dir) / "transcript_candidate_a.pdf"
        create_sample_pdf(
            transcript_path,
            "Interview Transcript - Candidate A",
            "Interviewer: Can you describe a critical production incident you resolved?\n"
            "Candidate: At CloudCorp, a deadlock caused severe connection pool exhaustion. I identified the missing composite index and mitigated latency within 20 minutes.\n"
            "Interviewer: How do you approach cross-functional alignment?\n"
            "Candidate: I run weekly RFC review sessions to align engineering and product roadmaps.",
        )

        # 2. Test extraction for all three files
        print("\n[1] Extracting documents from file paths...")
        docs = [
            extract_pdf_text(jd_path, source_type="job_description", document_id="doc_jd"),
            extract_pdf_text(resume_path, source_type="resume", document_id="doc_resume_a"),
            extract_pdf_text(transcript_path, source_type="transcript", document_id="doc_transcript_a"),
        ]
        docs_by_filename = {doc.filename: doc for doc in docs}

        store = EvidenceStore()
        for doc in docs:
            items = store.add_document(doc)
            print(f"  - Ingested '{doc.filename}' ({doc.source_type}): {len(doc.pages)} page(s), {len(items)} evidence items")

        # 3. Test in-memory byte extraction
        print("\n[2] Testing in-memory byte extraction...")
        with open(resume_path, "rb") as f:
            byte_data = f.read()
        byte_doc = extract_pdf_from_bytes(byte_data, filename="memory_resume.pdf", source_type="resume")
        assert len(byte_doc.pages) == 1
        print(f"  [OK] Successfully parsed {len(byte_data)} PDF bytes in-memory")

        # 4. Verify Evidence IDs, Traceability, and Page-Level Grounding
        print("\n[3] Verifying evidence quotes trace back to extracted page text...")
        all_evidence = store.get_all()
        print(f"  - Total Generated Evidence Records: {len(all_evidence)}")
        
        expected_prefix = "E"
        for i, item in enumerate(all_evidence, start=1):
            expected_id = f"{expected_prefix}{i:03d}"
            assert item.evidence_id == expected_id, f"Expected {expected_id}, got {item.evidence_id}"

            # Verify quote is strictly grounded in the extracted page text
            origin_doc = docs_by_filename.get(item.filename)
            assert origin_doc is not None, f"Origin doc {item.filename} not found"
            
            # Find matching page
            page = next((p for p in origin_doc.pages if p.page_number == item.page_number), None)
            assert page is not None, f"Page {item.page_number} not found in {item.filename}"

            norm_quote = normalize_ws(item.quote)
            norm_page_text = normalize_ws(page.text)
            assert norm_quote in norm_page_text, (
                f"Evidence quote '{norm_quote}' not found in page {item.page_number} text of {item.filename}"
            )

        print(f"  [OK] All {len(all_evidence)} evidence quotes verified as strictly grounded in page text")

        # 5. Verify source filtering
        print("\n[4] Verifying evidence query by source type:")
        jd_items = store.get_by_source("job_description")
        resume_items = store.get_by_source("resume")
        transcript_items = store.get_by_source("transcript")
        
        print(f"  - Job Description items: {len(jd_items)}")
        print(f"  - Resume items:          {len(resume_items)}")
        print(f"  - Transcript items:      {len(transcript_items)}")
        assert len(jd_items) > 0 and len(resume_items) > 0 and len(transcript_items) > 0

        # 6. Verify prompt formatting
        print("\n[5] Prompt representation sample:")
        prompt_text = store.format_for_prompt(source_type="transcript")
        print(prompt_text)

        print("\n" + "=" * 70)
        print("ALL STAGE 2 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)


def process_actual_data_pdfs():
    """Runs the extraction and evidence pipeline on the 5 actual PDFs in data/."""
    print("\n" + "=" * 70)
    print("PROCESSING ACTUAL PDFS IN DATA/ DIRECTORY")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data"
    pdf_configs = [
        {"filename": "job_description.pdf", "source_type": "job_description", "document_id": "doc_jd"},
        {"filename": "resume_a.pdf", "source_type": "resume", "document_id": "doc_resume_a"},
        {"filename": "resume_b.pdf", "source_type": "resume", "document_id": "doc_resume_b"},
        {"filename": "transcript_a.pdf", "source_type": "transcript", "document_id": "doc_transcript_a"},
        {"filename": "transcript_b.pdf", "source_type": "transcript", "document_id": "doc_transcript_b"},
    ]

    store = EvidenceStore()
    results = []

    for cfg in pdf_configs:
        file_path = data_dir / cfg["filename"]
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            continue

        try:
            doc = extract_pdf_text(
                file_path=file_path,
                source_type=cfg["source_type"],
                document_id=cfg["document_id"],
            )
            items = store.add_document(doc)
            total_chars = sum(len(p.text) for p in doc.pages)
            is_suspicious = total_chars < 100 or any(len(p.text.strip()) == 0 for p in doc.pages)

            # Check grounding for all generated items
            for item in items:
                page = next(p for p in doc.pages if p.page_number == item.page_number)
                assert normalize_ws(item.quote) in normalize_ws(page.text)

            res = {
                "filename": cfg["filename"],
                "source_type": cfg["source_type"],
                "num_pages": len(doc.pages),
                "total_chars": total_chars,
                "evidence_count": len(items),
                "is_suspicious": is_suspicious,
                "error": None,
            }
            results.append(res)
        except Exception as exc:
            results.append({
                "filename": cfg["filename"],
                "source_type": cfg["source_type"],
                "num_pages": 0,
                "total_chars": 0,
                "evidence_count": 0,
                "is_suspicious": True,
                "error": str(exc),
            })

    print("\nExtraction & Evidence Summary for data/ PDFs:")
    print("-" * 75)
    print(f"{'Filename':<22} | {'Source Type':<16} | {'Pages':<6} | {'Chars':<7} | {'Evidence':<8} | {'Status'}")
    print("-" * 75)
    for r in results:
        status = f"ERROR: {r['error']}" if r["error"] else ("SUSPICIOUS" if r["is_suspicious"] else "OK")
        print(f"{r['filename']:<22} | {r['source_type']:<16} | {r['num_pages']:<6} | {r['total_chars']:<7} | {r['evidence_count']:<8} | {status}")
    print("-" * 75)
    print(f"Total Evidence Items Generated Across Actual PDFs: {len(store.get_all())}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_stage2_verification()
    process_actual_data_pdfs()
