"""Independent Agent Runner Orchestration.

Executes the four panel agents in complete isolation for the initial assessment stage.
No agent receives or has access to the assessments or conclusions of other agents.
"""

from typing import List, Optional
from openai import OpenAI

from backend.models import (
    CandidateProfile,
    AgentAssessment,
    AgentRole,
)
from backend.evidence import EvidenceStore
from backend.agents.technical import TechnicalAgent
from backend.agents.hr_culture import HRCultureAgent
from backend.agents.hiring_manager import HiringManagerAgent
from backend.agents.skeptic import SkepticAgent


def run_independent_agents(
    job_description: str,
    candidate_profile: CandidateProfile,
    evidence_store: EvidenceStore,
    client: Optional[OpenAI] = None,
) -> List[AgentAssessment]:
    """Runs all four panel agents independently on candidate and evidence data.

    Enforces strict independence:
    - Each agent is invoked in a completely separate evaluation call.
    - No agent receives any other agent's output or conclusions.

    Returns:
        List of exactly 4 AgentAssessment objects (Technical, HR, Hiring Manager, Skeptic).
    """
    agents = [
        TechnicalAgent(client=client),
        HRCultureAgent(client=client),
        HiringManagerAgent(client=client),
        SkepticAgent(client=client),
    ]

    assessments: List[AgentAssessment] = []

    for agent in agents:
        # Isolated execution: only job description, candidate profile, and evidence store provided
        assessment = agent.evaluate(
            job_description=job_description,
            candidate_profile=candidate_profile,
            evidence_store=evidence_store,
        )
        assessments.append(assessment)

    return assessments

