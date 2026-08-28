"""Debate Coordinator for Multi-Agent Interview Panel.

Analyzes the four independent agent assessments, identifies substantive disagreements,
and prioritizes at most two focal debate topics that materially impact the hiring decision.
"""

import json
import logging
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import settings
from backend.models import (
    AgentRole,
    AgentAssessment,
    CandidateProfile,
    DebateTopic,
)
from backend.evidence import EvidenceStore

logger = logging.getLogger(__name__)

COORDINATOR_SYSTEM_PROMPT = """You are the Panel Debate Coordinator for an AI Interview Panel.

YOUR MANDATE:
- Analyze the independent assessments from the Technical Agent, HR / Culture Agent, Hiring Manager Agent, and Skeptic Agent.
- Identify the most substantive, high-stakes disagreements between the agents (e.g. conflicts between recommendations, disputed technical depth vs skepticism over claims, risk assessments, or differing interpretations of evidence).
- Select at most 2 important disagreement topics that could materially alter the hiring outcome.
- If all four agents are in genuine complete agreement and have no substantive conflicts, return an empty list of topics. Do NOT manufacture artificial disagreements.

REQUIREMENTS:
- For each topic, specify:
  * topic: Short title of the dispute.
  * agents_involved: The 2 primary agents in conflict (must use exact role names: 'Technical Agent', 'HR / Culture Agent', 'Hiring Manager Agent', 'Skeptic Agent').
  * disagreement_description: Concise summary of what the agents disagree on and why.
  * relevant_evidence_ids: List of relevant evidence IDs (e.g. ['E001', 'E004']) from the provided Evidence Store.

Respond strictly in JSON format:
{
  "topics": [
    {
      "topic": "string",
      "agents_involved": ["Technical Agent", "Skeptic Agent"],
      "disagreement_description": "string",
      "relevant_evidence_ids": ["E001", "E008"]
    }
  ]
}
"""


class CoordinatorLLMResponse(BaseModel):
    """Schema for coordinator LLM structured output."""
    topics: List[DebateTopic] = Field(default_factory=list)


class DebateCoordinator:
    """Orchestrates disagreement detection across panel assessments."""

    def __init__(self, client: Optional[OpenAI] = None):
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = settings.OPENAI_API_KEY or "missing-api-key"
            self._client = OpenAI(
                api_key=api_key,
                base_url=settings.OPENAI_BASE_URL,
            )
        return self._client

    def build_summary_prompt(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
    ) -> str:
        """Constructs a consolidated prompt comparing the four independent assessments."""
        assessments_text = []
        for a in initial_assessments:
            citations_str = ", ".join(c.evidence_id for c in a.evidence_citations) or "None"
            strengths_str = "; ".join(a.strengths) if a.strengths else "None"
            concerns_str = "; ".join(a.concerns) if a.concerns else "None"
            score_str = f"{a.score}/10" if a.score is not None else "N/A"

            assessments_text.append(
                f"[{a.agent_role.value}]\n"
                f"- Recommendation: {a.recommendation} (Confidence: {a.confidence})\n"
                f"- Domain Score: {score_str}\n"
                f"- Strengths: {strengths_str}\n"
                f"- Concerns / Risks: {concerns_str}\n"
                f"- Evidence Cited: {citations_str}\n"
                f"- Missing Info Notes: {a.insufficient_info_notes or 'None'}\n"
            )

        formatted_assessments = "\n".join(assessments_text)
        evidence_text = evidence_store.format_for_prompt()

        return f"""=== TARGET JOB DESCRIPTION ===
{job_description}

=== CANDIDATE PROFILE ===
Candidate ID: {candidate_profile.candidate_id}
Candidate Name: {candidate_profile.name}
Experience Summary: {candidate_profile.experience_summary}

=== FOUR INDEPENDENT AGENT ASSESSMENTS ===
{formatted_assessments}

=== AVAILABLE EVIDENCE STORE ===
{evidence_text}

=== TASK ===
Analyze the 4 assessments above. Identify at most 2 major disagreements between specific pairs of agents.
Only cite valid evidence IDs from the Evidence Store.
"""

    def identify_disagreements(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
    ) -> List[DebateTopic]:
        """Identifies at most 2 major disagreement topics among the panel agents."""
        if not initial_assessments or len(initial_assessments) < 2:
            return []

        user_prompt = self.build_summary_prompt(
            job_description,
            candidate_profile,
            evidence_store,
            initial_assessments,
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            parsed_data = json.loads(raw_content)

            # Support both direct list and dict with "topics"
            if isinstance(parsed_data, list):
                topics_raw = parsed_data
            else:
                topics_raw = parsed_data.get("topics", [])

            validated_topics: List[DebateTopic] = []
            for t_data in topics_raw[:2]:  # Enforce max 2 topics
                try:
                    # Filter relevant evidence IDs to valid ones in store
                    raw_eids = t_data.get("relevant_evidence_ids", [])
                    valid_eids = [eid for eid in raw_eids if evidence_store.get_by_id(eid) is not None]
                    t_data["relevant_evidence_ids"] = valid_eids
                    topic_obj = DebateTopic.model_validate(t_data)
                    # Must have at least 2 distinct agents involved
                    if len(topic_obj.agents_involved) >= 2:
                        validated_topics.append(topic_obj)
                except Exception as val_err:
                    logger.warning("Skipping malformed debate topic: %s (%s)", t_data, val_err)

            return validated_topics

        except Exception as exc:
            logger.error("Error in DebateCoordinator: %s", exc)
            return []

