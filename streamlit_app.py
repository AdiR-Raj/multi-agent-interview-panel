"""
Multi-Agent AI Interview Panel Simulator — Streamlit Frontend

Entry point for Streamlit Community Cloud deployment.
All PDF processing, agent evaluation, debate, and final decision logic
lives in the existing backend package.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of where Streamlit starts.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from openai import OpenAI

from backend.config import settings
from backend.pdf_utils import extract_pdf_from_bytes
from backend.evidence import EvidenceStore
from backend.models import (
    CandidateProfile,
    AgentRole,
    AgentAssessment,
    DebateTurn,
    OpinionReassessment,
    FinalDecision,
    EvidenceReference,
)
from backend.agents import run_independent_agents
from backend.debate import run_debate_and_reassessment
from backend.decision import make_final_decision

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent AI Interview Panel",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    """Creates OpenAI client using settings (pulls from Streamlit secrets or .env)."""
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )


def build_candidate_profile(
    candidate_id: str,
    resume_doc,
    transcript_doc,
    evidence_store: EvidenceStore,
) -> CandidateProfile:
    """Builds a basic CandidateProfile from extracted documents.

    Extracts name from the first evidence line of the resume, collects skills
    by keyword scanning, and summarises experience from resume evidence.
    """
    resume_items = evidence_store.get_by_document(resume_doc.document_id)
    transcript_items = evidence_store.get_by_document(transcript_doc.document_id)

    # Heuristic: first non-empty evidence quote from resume is likely the name/header
    name = candidate_id
    if resume_items:
        first_quote = resume_items[0].quote.strip()
        # Use the first short line (likely a name)
        for item in resume_items:
            if len(item.quote) < 60 and "\n" not in item.quote:
                name = item.quote.strip()
                break

    # Collect all quotes as claims for agent grounding
    claims = [item.quote for item in resume_items + transcript_items]

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        extracted_skills=[],          # Agents infer from evidence
        experience_summary="See evidence store.",
        claims=claims[:30],           # Cap at 30 to avoid token overflow
        insufficient_info_flags=[],
    )


def render_evidence_reference(ref: EvidenceReference, evidence_store: EvidenceStore):
    """Renders a single evidence reference with full traceability."""
    item = evidence_store.get_by_id(ref.evidence_id)
    if item:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**{item.evidence_id}**")
                st.caption(f"{item.source_type}")
                st.caption(f"{item.filename}")
                st.caption(f"Page {item.page_number}")
            with col2:
                st.markdown(f"*\"{item.quote}\"*")
    else:
        st.caption(f"📎 {ref.evidence_id}: {ref.quote[:120]}...")


def render_evidence_list(refs: list[EvidenceReference], evidence_store: EvidenceStore, label: str = "Evidence"):
    """Renders a labelled, collapsible evidence list."""
    if not refs:
        return
    with st.expander(f"📎 {label} ({len(refs)} citation{'s' if len(refs) != 1 else ''})"):
        for ref in refs:
            render_evidence_reference(ref, evidence_store)


def render_agent_assessment(assessment: AgentAssessment, evidence_store: EvidenceStore):
    """Renders one agent's initial assessment."""
    rec = assessment.recommendation
    colour = {
        "Strong Hire": "🟢",
        "Hire": "🟩",
        "Weak Hire": "🟡",
        "Reject": "🔴",
        "Undecided": "⚪",
    }.get(rec, "⚪")

    with st.container(border=True):
        st.markdown(f"#### {colour} {assessment.agent_role.value}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Recommendation", rec)
        col2.metric("Score", f"{assessment.score:.1f}/10" if assessment.score is not None else "N/A")
        col3.metric("Confidence", f"{assessment.confidence:.0%}")

        if assessment.strengths:
            st.markdown("**Strengths**")
            for s in assessment.strengths:
                st.markdown(f"- {s}")

        if assessment.concerns:
            st.markdown("**Concerns**")
            for c in assessment.concerns:
                st.markdown(f"- {c}")

        if assessment.insufficient_info_notes:
            st.info(f"ℹ️ Insufficient info: {assessment.insufficient_info_notes}")

        render_evidence_list(assessment.evidence_citations, evidence_store, "Agent Evidence Citations")


def render_debate_turn(turn: DebateTurn, index: int, evidence_store: EvidenceStore):
    """Renders a single debate turn."""
    speaker = turn.speaker.value
    target = f" → *{turn.target_agent.value}*" if turn.target_agent else ""
    with st.container(border=True):
        st.markdown(f"**Turn {index + 1}: {speaker}**{target}")
        st.markdown(f"**Topic:** {turn.topic}")
        st.markdown(turn.argument)
        render_evidence_list(turn.cited_evidence, evidence_store, "Cited Evidence")


def render_reassessment(r: OpinionReassessment):
    """Renders one agent's post-debate reassessment."""
    changed_badge = "🔄 Changed" if r.changed else "✅ Maintained"
    with st.container(border=True):
        st.markdown(f"**{r.agent_role.value}** — {changed_badge}")
        col1, col2 = st.columns(2)
        col1.markdown(f"Before: **{r.original_recommendation}** ({r.original_confidence:.0%})")
        col2.markdown(f"After: **{r.revised_recommendation}** ({r.revised_confidence:.0%})")
        st.caption(r.reasons_for_change)


def render_final_decision(decision: FinalDecision, evidence_store: EvidenceStore):
    """Renders the final hiring decision."""
    colour = {
        "Strong Hire": "🟢",
        "Hire": "🟩",
        "Weak Hire": "🟡",
        "Reject": "🔴",
        "Undecided": "⚪",
    }.get(decision.recommendation, "⚪")

    st.markdown(f"## {colour} Final Decision: **{decision.recommendation}**")
    st.metric("Overall Confidence", f"{decision.confidence:.0%}")

    col1, col2 = st.columns(2)
    with col1:
        if decision.strengths:
            st.markdown("**Decisive Strengths**")
            for s in decision.strengths:
                st.markdown(f"- {s}")
    with col2:
        if decision.concerns:
            st.markdown("**Decisive Concerns**")
            for c in decision.concerns:
                st.markdown(f"- {c}")

    st.markdown("**Synthesis Rationale**")
    st.markdown(decision.synthesis_rationale)

    if decision.unresolved_disagreements:
        st.warning("**Unresolved Disagreements**")
        for d in decision.unresolved_disagreements:
            st.markdown(f"- {d}")

    if decision.insufficient_information_flags:
        st.error("**Insufficient Information Flags**")
        for f in decision.insufficient_information_flags:
            st.markdown(f"- {f}")

    render_evidence_list(decision.decisive_evidence, evidence_store, "Decisive Evidence")


def run_candidate_evaluation(
    candidate_id: str,
    job_description_text: str,
    resume_bytes: bytes,
    resume_name: str,
    transcript_bytes: bytes,
    transcript_name: str,
    client: OpenAI,
) -> dict:
    """Runs the complete evaluation pipeline for one candidate.

    Returns a dict with: evidence_store, profile, assessments, debate_transcript,
    reassessments, final_decision.
    """
    evidence_store = EvidenceStore()

    # --- Stage 2: Extract PDFs and build EvidenceStore ---
    resume_doc = extract_pdf_from_bytes(resume_bytes, resume_name, "resume", f"doc_resume_{candidate_id}")
    transcript_doc = extract_pdf_from_bytes(transcript_bytes, transcript_name, "transcript", f"doc_transcript_{candidate_id}")
    evidence_store.add_document(resume_doc)
    evidence_store.add_document(transcript_doc)

    # --- Build candidate profile ---
    profile = build_candidate_profile(candidate_id, resume_doc, transcript_doc, evidence_store)

    # --- Stage 3: Independent agents ---
    assessments = run_independent_agents(
        job_description=job_description_text,
        candidate_profile=profile,
        evidence_store=evidence_store,
        client=client,
    )

    # --- Stage 4: Debate + reassessment ---
    debate_transcript, reassessments = run_debate_and_reassessment(
        job_description=job_description_text,
        candidate_profile=profile,
        evidence_store=evidence_store,
        initial_assessments=assessments,
        client=client,
    )

    # --- Stage 5: Final decision ---
    final_decision = make_final_decision(
        job_description=job_description_text,
        candidate_profile=profile,
        evidence_store=evidence_store,
        initial_assessments=assessments,
        debate_transcript=debate_transcript,
        reassessments=reassessments,
        client=client,
    )

    return {
        "evidence_store": evidence_store,
        "profile": profile,
        "assessments": assessments,
        "debate_transcript": debate_transcript,
        "reassessments": reassessments,
        "final_decision": final_decision,
    }


def render_candidate_results(candidate_label: str, results: dict):
    """Renders all evaluation results for one candidate."""
    profile = results["profile"]
    evidence_store = results["evidence_store"]
    assessments = results["assessments"]
    debate_transcript = results["debate_transcript"]
    reassessments = results["reassessments"]
    final_decision = results["final_decision"]

    st.header(f"🧑 {candidate_label}: {profile.name}")

    # Final decision at the top for quick scan
    with st.container(border=True):
        render_final_decision(final_decision, evidence_store)

    st.divider()

    # Independent assessments
    st.subheader("📋 Independent Agent Assessments")
    st.caption("Each agent evaluated the candidate independently with no access to other agents' opinions.")
    for assessment in assessments:
        render_agent_assessment(assessment, evidence_store)

    st.divider()

    # Debate
    st.subheader("⚔️ Panel Debate")
    if debate_transcript:
        for i, turn in enumerate(debate_transcript):
            render_debate_turn(turn, i, evidence_store)
    else:
        st.info("No significant disagreements detected. Debate stage skipped.")

    st.divider()

    # Post-debate reassessments
    st.subheader("🔄 Post-Debate Opinion Reassessments")
    for r in reassessments:
        render_reassessment(r)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🤖 Multi-Agent AI Interview Panel")
st.caption(
    "Deterministic multi-agent candidate evaluation with grounded evidence, "
    "structured debate, and non-averaging final decision synthesis."
)

# Check API key (don't show the value)
if not settings.OPENAI_API_KEY:
    st.error(
        "⚠️ **OPENAI_API_KEY is not configured.** "
        "Add it to your `.env` file (local) or Streamlit Community Cloud secrets (deployed)."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Upload controls
# ---------------------------------------------------------------------------
st.header("📁 Upload Documents")

col_jd, col_spacer = st.columns([2, 3])
with col_jd:
    jd_file = st.file_uploader("Job Description PDF", type=["pdf"], key="jd_upload")

st.subheader("Candidate A")
col_a1, col_a2 = st.columns(2)
with col_a1:
    resume_a_file = st.file_uploader("Candidate A — Resume PDF", type=["pdf"], key="resume_a_upload")
with col_a2:
    transcript_a_file = st.file_uploader("Candidate A — Interview Transcript PDF", type=["pdf"], key="transcript_a_upload")

st.subheader("Candidate B")
col_b1, col_b2 = st.columns(2)
with col_b1:
    resume_b_file = st.file_uploader("Candidate B — Resume PDF", type=["pdf"], key="resume_b_upload")
with col_b2:
    transcript_b_file = st.file_uploader("Candidate B — Interview Transcript PDF", type=["pdf"], key="transcript_b_upload")

st.divider()

# ---------------------------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------------------------
all_uploaded = all([jd_file, resume_a_file, transcript_a_file, resume_b_file, transcript_b_file])

if not all_uploaded:
    st.info("Upload all 5 PDFs above to enable evaluation.")
else:
    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        # Store raw bytes now (Streamlit file_uploader objects reset on rerun)
        st.session_state["jd_bytes"] = jd_file.read()
        st.session_state["jd_name"] = jd_file.name
        st.session_state["resume_a_bytes"] = resume_a_file.read()
        st.session_state["resume_a_name"] = resume_a_file.name
        st.session_state["transcript_a_bytes"] = transcript_a_file.read()
        st.session_state["transcript_a_name"] = transcript_a_file.name
        st.session_state["resume_b_bytes"] = resume_b_file.read()
        st.session_state["resume_b_name"] = resume_b_file.name
        st.session_state["transcript_b_bytes"] = transcript_b_file.read()
        st.session_state["transcript_b_name"] = transcript_b_file.name
        st.session_state["results_a"] = None
        st.session_state["results_b"] = None

        client = get_openai_client()

        # Extract job description text (build a small evidence store just for JD text)
        jd_doc = extract_pdf_from_bytes(
            st.session_state["jd_bytes"],
            st.session_state["jd_name"],
            "job_description",
            "doc_job_description",
        )
        jd_text = "\n\n".join(page.text for page in jd_doc.pages)

        # --- Candidate A ---
        with st.spinner("Evaluating Candidate A…"):
            try:
                results_a = run_candidate_evaluation(
                    candidate_id="Candidate_A",
                    job_description_text=jd_text,
                    resume_bytes=st.session_state["resume_a_bytes"],
                    resume_name=st.session_state["resume_a_name"],
                    transcript_bytes=st.session_state["transcript_a_bytes"],
                    transcript_name=st.session_state["transcript_a_name"],
                    client=client,
                )
                st.session_state["results_a"] = results_a
            except Exception as e:
                st.error(f"Candidate A evaluation failed: {e}")

        # --- Candidate B ---
        with st.spinner("Evaluating Candidate B…"):
            try:
                results_b = run_candidate_evaluation(
                    candidate_id="Candidate_B",
                    job_description_text=jd_text,
                    resume_bytes=st.session_state["resume_b_bytes"],
                    resume_name=st.session_state["resume_b_name"],
                    transcript_bytes=st.session_state["transcript_b_bytes"],
                    transcript_name=st.session_state["transcript_b_name"],
                    client=client,
                )
                st.session_state["results_b"] = results_b
            except Exception as e:
                st.error(f"Candidate B evaluation failed: {e}")

# ---------------------------------------------------------------------------
# Results display (persists via session_state)
# ---------------------------------------------------------------------------
results_a = st.session_state.get("results_a")
results_b = st.session_state.get("results_b")

if results_a or results_b:
    st.divider()
    st.header("📊 Evaluation Results")

    tab_a, tab_b = st.tabs(["Candidate A", "Candidate B"])

    with tab_a:
        if results_a:
            render_candidate_results("Candidate A", results_a)
        else:
            st.warning("Candidate A evaluation did not complete.")

    with tab_b:
        if results_b:
            render_candidate_results("Candidate B", results_b)
        else:
            st.warning("Candidate B evaluation did not complete.")

