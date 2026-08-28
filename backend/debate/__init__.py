"""Debate and Opinion Reassessment Package for Multi-Agent AI Interview Panel Simulator."""

from typing import List, Tuple, Optional
from openai import OpenAI

from backend.models import (
    CandidateProfile,
    AgentAssessment,
    DebateTurn,
    OpinionReassessment,
)
from backend.evidence import EvidenceStore
from backend.debate.coordinator import DebateCoordinator
from backend.debate.debate_agent import DebateEngine
from backend.debate.reassessment import OpinionReassessmentEngine


def run_debate_and_reassessment(
    job_description: str,
    candidate_profile: CandidateProfile,
    evidence_store: EvidenceStore,
    initial_assessments: List[AgentAssessment],
    client: Optional[OpenAI] = None,
) -> Tuple[List[DebateTurn], List[OpinionReassessment]]:
    """Orchestrates the entire Stage 4 debate and opinion reassessment pipeline.

    1. Identifies major disagreements using DebateCoordinator (max 2 topics).
    2. Generates structured debate turns using DebateEngine (max 2 turns per topic, max 4 turns total).
    3. Reassesses all 4 agents using OpinionReassessmentEngine.
    4. Preserves initial AgentAssessment objects unchanged.

    Returns:
        Tuple of (debate_transcript: List[DebateTurn], reassessments: List[OpinionReassessment])
    """
    coordinator = DebateCoordinator(client=client)
    debate_engine = DebateEngine(client=client)
    reassessment_engine = OpinionReassessmentEngine(client=client)

    # 1. Identify substantive disagreements
    topics = coordinator.identify_disagreements(
        job_description=job_description,
        candidate_profile=candidate_profile,
        evidence_store=evidence_store,
        initial_assessments=initial_assessments,
    )

    # 2. Generate debate turns (max 4 turns total)
    debate_transcript = debate_engine.generate_debate_turns(
        job_description=job_description,
        candidate_profile=candidate_profile,
        evidence_store=evidence_store,
        initial_assessments=initial_assessments,
        topics=topics,
    )

    # 3. Reassess all 4 agents
    reassessments = reassessment_engine.reassess_all(
        job_description=job_description,
        candidate_profile=candidate_profile,
        evidence_store=evidence_store,
        initial_assessments=initial_assessments,
        debate_turns=debate_transcript,
    )

    return debate_transcript, reassessments


__all__ = [
    "DebateCoordinator",
    "DebateEngine",
    "OpinionReassessmentEngine",
    "run_debate_and_reassessment",
]

