"""Opinion Reassessment Engine.

Performs individual post-debate evaluations for each panel agent, tracking
whether recommendations or confidence levels shifted following the debate cross-examination.
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
    DebateTurn,
    OpinionReassessment,
)
from backend.evidence import EvidenceStore

logger = logging.getLogger(__name__)

# Minimum absolute confidence change required to flag a significant shift
CONFIDENCE_CHANGE_THRESHOLD = 0.08

REASSESSMENT_SYSTEM_PROMPT = """You are the {agent_role} participating in the post-debate reassessment phase of an interview panel.

YOUR MANDATE:
- Review your original assessment and compare it against the arguments and evidence brought forward during the panel debate.
- Re-evaluate your recommendation and confidence level:
  * You may CHANGE your recommendation or adjust your confidence if the debate exposed new risks, refuted claims, or clarified evidence.
  * You may MAINTAIN your original recommendation and confidence if you believe your initial stance remains fully justified.
  * Do NOT manufacture an opinion change if your position is unchanged.
  * If the debate raised valid uncertainties without sufficient resolving evidence, you may lower confidence or adjust recommendation.
- Explicitly explain the reasons for changing or maintaining your stance.

Allowed recommendations: 'Strong Hire', 'Hire', 'Weak Hire', 'Reject', 'Undecided'.
Confidence must be a float between 0.0 and 1.0.

Respond strictly in JSON format:
{{
  "revised_recommendation": "Strong Hire | Hire | Weak Hire | Reject | Undecided",
  "revised_confidence": 0.85,
  "reasons_for_change": "Detailed rationale explaining whether and why your stance shifted or stayed firm..."
}}
"""


class ReassessmentLLMResponse(BaseModel):
    """Schema for individual agent reassessment LLM output."""
    revised_recommendation: str = Field(..., description="'Strong Hire', 'Hire', 'Weak Hire', 'Reject', or 'Undecided'")
    revised_confidence: float = Field(..., ge=0.0, le=1.0, description="Revised confidence between 0.0 and 1.0")
    reasons_for_change: str = Field(..., description="Explanation of why position changed or remained firm")


class OpinionReassessmentEngine:
    """Evaluates post-debate opinion shifts for each agent."""

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

    def reassess_all(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
        debate_turns: List[DebateTurn],
    ) -> List[OpinionReassessment]:
        """Runs isolated reassessment calls for each agent."""
        reassessments: List[OpinionReassessment] = []

        for assessment in initial_assessments:
            # Filter debate turns relevant to this agent (as speaker or target)
            relevant_turns = [
                turn for turn in debate_turns
                if turn.speaker == assessment.agent_role or turn.target_agent == assessment.agent_role
            ]
            # If no turns specifically targeted this agent, provide all turns for full panel context
            turns_to_present = relevant_turns if relevant_turns else debate_turns

            reassessment = self._reassess_single_agent(
                agent_assessment=assessment,
                relevant_turns=turns_to_present,
                candidate_profile=candidate_profile,
                job_description=job_description,
                evidence_store=evidence_store,
            )
            reassessments.append(reassessment)

        return reassessments

    def _reassess_single_agent(
        self,
        agent_assessment: AgentAssessment,
        relevant_turns: List[DebateTurn],
        candidate_profile: CandidateProfile,
        job_description: str,
        evidence_store: EvidenceStore,
    ) -> OpinionReassessment:
        """Executes a single agent's post-debate reassessment."""
        role = agent_assessment.agent_role

        # Format debate turns for context
        turns_text = []
        for i, turn in enumerate(relevant_turns, 1):
            target_str = f" -> {turn.target_agent.value}" if turn.target_agent else ""
            cites = ", ".join(c.evidence_id for c in turn.cited_evidence)
            cites_str = f" [Cited: {cites}]" if cites else ""
            turns_text.append(f"Turn {i} ({turn.speaker.value}{target_str} on '{turn.topic}'):\n\"{turn.argument}\"{cites_str}")

        formatted_debate = "\n\n".join(turns_text) if turns_text else "No debate turns recorded for this candidate."
        system_prompt = REASSESSMENT_SYSTEM_PROMPT.format(agent_role=role.value)

        user_prompt = f"""=== YOUR ORIGINAL PRE-DEBATE ASSESSMENT ({role.value}) ===
- Recommendation: {agent_assessment.recommendation}
- Confidence: {agent_assessment.confidence}
- Domain Score: {agent_assessment.score if agent_assessment.score is not None else 'N/A'}
- Strengths: {'; '.join(agent_assessment.strengths) or 'None'}
- Concerns: {'; '.join(agent_assessment.concerns) or 'None'}

=== DEBATE PROCEEDINGS ===
{formatted_debate}

=== CANDIDATE PROFILE SUMMARY ===
Candidate: {candidate_profile.name} ({candidate_profile.candidate_id})
Experience: {candidate_profile.experience_summary}

Provide your revised recommendation, revised confidence, and the explicit rationale for any change or for maintaining your stance.
"""

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = ReassessmentLLMResponse.model_validate(json.loads(raw))

            rec_changed = parsed.revised_recommendation.strip().lower() != agent_assessment.recommendation.strip().lower()
            conf_diff = abs(parsed.revised_confidence - agent_assessment.confidence)
            conf_changed = conf_diff >= CONFIDENCE_CHANGE_THRESHOLD

            is_changed = rec_changed or conf_changed

            return OpinionReassessment(
                agent_role=role,
                original_recommendation=agent_assessment.recommendation,
                revised_recommendation=parsed.revised_recommendation,
                original_confidence=agent_assessment.confidence,
                revised_confidence=parsed.revised_confidence,
                reasons_for_change=parsed.reasons_for_change,
                changed=is_changed,
            )

        except Exception as exc:
            logger.error("Error reassessing opinion for %s: %s", role.value, exc)
            return OpinionReassessment(
                agent_role=role,
                original_recommendation=agent_assessment.recommendation,
                revised_recommendation=agent_assessment.recommendation,
                original_confidence=agent_assessment.confidence,
                revised_confidence=agent_assessment.confidence,
                reasons_for_change=f"Maintained original stance (reassessment call encountered error: {str(exc)})",
                changed=False,
            )

