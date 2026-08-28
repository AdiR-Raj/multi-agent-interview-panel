"""Final Decision Package for Multi-Agent AI Interview Panel Simulator."""

from typing import List, Optional
from openai import OpenAI

from backend.models import (
    CandidateProfile,
    AgentAssessment,
    DebateTurn,
    OpinionReassessment,
    FinalDecision,
)
from backend.evidence import EvidenceStore
from backend.decision.final_decision import FinalDecisionEngine


def make_final_decision(
    job_description: str,
    candidate_profile: CandidateProfile,
    evidence_store: EvidenceStore,
    initial_assessments: List[AgentAssessment],
    debate_transcript: List[DebateTurn],
    reassessments: List[OpinionReassessment],
    client: Optional[OpenAI] = None,
) -> FinalDecision:
    """Entry point for executing non-averaging final decision synthesis.

    Takes the full candidate evaluation state (job description, profile, evidence,
    pre-debate assessments, debate transcript, and post-debate reassessments)
    and executes a dedicated reasoning LLM call.

    Returns:
        FinalDecision object containing qualitative synthesis, decisive evidence citations,
        and unresolved disagreements.
    """
    engine = FinalDecisionEngine(client=client)
    return engine.synthesize(
        job_description=job_description,
        candidate_profile=candidate_profile,
        evidence_store=evidence_store,
        initial_assessments=initial_assessments,
        debate_transcript=debate_transcript,
        reassessments=reassessments,
    )


__all__ = [
    "FinalDecisionEngine",
    "make_final_decision",
]

