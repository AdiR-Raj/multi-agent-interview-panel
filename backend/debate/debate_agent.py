"""Debate Engine and Direct Agent-to-Agent Cross-Examination.

Generates structured debate turns between agents on prioritized disagreement topics,
ensuring direct responses and evidence grounding.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import settings
from backend.models import (
    AgentRole,
    AgentAssessment,
    CandidateProfile,
    DebateTopic,
    DebateTurn,
    EvidenceReference,
)
from backend.evidence import EvidenceStore

logger = logging.getLogger(__name__)

DEBATE_CHALLENGE_SYSTEM_PROMPT = """You are acting as the {speaker_role} in an interview panel debate.

MANDATE:
- Directly challenge the {target_role}'s assessment on the following disputed topic: "{topic}".
- Frame your argument clearly and forcefully from your perspective as the {speaker_role}.
- Specifically address the {target_role}'s assumptions, interpretation of candidate answers, or rating.
- Cite specific evidence IDs (e.g. ['E001', 'E004']) from the provided Evidence Store to back your challenge.
- Do NOT invent evidence or evidence IDs.

Format your response strictly as JSON:
{{
  "argument": "Direct challenge statement addressing {target_role}...",
  "cited_evidence_ids": ["E001", "E005"]
}}
"""

DEBATE_RESPONSE_SYSTEM_PROMPT = """You are acting as the {speaker_role} in an interview panel debate.

MANDATE:
- Directly respond to the challenge just presented by the {target_role} on the topic: "{topic}".
- The {target_role} argued:
  "{previous_argument}"
- Address the {target_role}'s points directly. You may defend your stance with additional evidence, partially concede a point, or clarify why your interpretation is justified.
- Cite specific evidence IDs (e.g. ['E002', 'E006']) from the Evidence Store.
- Do NOT invent evidence or evidence IDs.

Format your response strictly as JSON:
{{
  "argument": "Direct response to {target_role}...",
  "cited_evidence_ids": ["E002"]
}}
"""


class TurnLLMResponse(BaseModel):
    """Schema for individual debate turn LLM response."""
    argument: str = Field(..., description="The spoken argument or counterpoint")
    cited_evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs cited in the argument")


class DebateEngine:
    """Orchestrates direct agent-to-agent debate turns."""

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

    def _resolve_citations(self, cited_eids: List[str], evidence_store: EvidenceStore) -> List[EvidenceReference]:
        """Resolves raw evidence IDs into verified EvidenceReference objects."""
        citations: List[EvidenceReference] = []
        for eid in cited_eids:
            item = evidence_store.get_by_id(eid)
            if item:
                citations.append(item.to_reference())
            else:
                logger.warning("Debate turn cited unknown evidence ID: %s", eid)
        return citations

    def generate_debate_turns(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
        topics: List[DebateTopic],
    ) -> List[DebateTurn]:
        """Generates structured debate turns for up to 2 topics (2 turns per topic).

        Enforces:
        - Maximum 4 total turns per candidate.
        - Direct response requirement: Turn 2 directly answers Turn 1.
        - Traceable evidence citations.
        """
        if not topics:
            return []

        assessment_map = {a.agent_role: a for a in initial_assessments}
        all_turns: List[DebateTurn] = []
        evidence_prompt_text = evidence_store.format_for_prompt()

        for topic_obj in topics[:2]:  # Limit to 2 topics
            if len(topic_obj.agents_involved) < 2:
                continue

            agent_a_role = topic_obj.agents_involved[0]
            agent_b_role = topic_obj.agents_involved[1]

            assessment_a = assessment_map.get(agent_a_role)
            assessment_b = assessment_map.get(agent_b_role)

            if not assessment_a or not assessment_b:
                continue

            # --- TURN 1: Agent A challenges Agent B ---
            turn_1 = self._generate_challenge_turn(
                topic_obj=topic_obj,
                speaker_role=agent_a_role,
                target_role=agent_b_role,
                speaker_assessment=assessment_a,
                target_assessment=assessment_b,
                evidence_text=evidence_prompt_text,
                evidence_store=evidence_store,
            )
            all_turns.append(turn_1)

            # --- TURN 2: Agent B directly responds to Agent A ---
            turn_2 = self._generate_response_turn(
                topic_obj=topic_obj,
                speaker_role=agent_b_role,
                target_role=agent_a_role,
                previous_argument=turn_1.argument,
                speaker_assessment=assessment_b,
                evidence_text=evidence_prompt_text,
                evidence_store=evidence_store,
            )
            all_turns.append(turn_2)

        return all_turns

    def _generate_challenge_turn(
        self,
        topic_obj: DebateTopic,
        speaker_role: AgentRole,
        target_role: AgentRole,
        speaker_assessment: AgentAssessment,
        target_assessment: AgentAssessment,
        evidence_text: str,
        evidence_store: EvidenceStore,
    ) -> DebateTurn:
        """Executes Turn 1: Challenge from speaker to target."""
        system_prompt = DEBATE_CHALLENGE_SYSTEM_PROMPT.format(
            speaker_role=speaker_role.value,
            target_role=target_role.value,
            topic=topic_obj.topic,
        )

        user_prompt = f"""DISPUTED TOPIC: {topic_obj.topic}
DISAGREEMENT CONTEXT: {topic_obj.disagreement_description}

YOUR INITIAL STANCE ({speaker_role.value}):
- Recommendation: {speaker_assessment.recommendation}
- Concerns: {'; '.join(speaker_assessment.concerns) or 'None'}
- Strengths: {'; '.join(speaker_assessment.strengths) or 'None'}

TARGET AGENT'S STANCE ({target_role.value}):
- Recommendation: {target_assessment.recommendation}
- Concerns: {'; '.join(target_assessment.concerns) or 'None'}
- Strengths: {'; '.join(target_assessment.strengths) or 'None'}

EVIDENCE STORE:
{evidence_text}

Present your direct challenge to {target_role.value} regarding "{topic_obj.topic}".
"""
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = TurnLLMResponse.model_validate(json.loads(raw))
            citations = self._resolve_citations(parsed.cited_evidence_ids, evidence_store)

            return DebateTurn(
                speaker=speaker_role,
                target_agent=target_role,
                topic=topic_obj.topic,
                argument=parsed.argument,
                cited_evidence=citations,
            )
        except Exception as exc:
            logger.error("Error generating debate challenge: %s", exc)
            return DebateTurn(
                speaker=speaker_role,
                target_agent=target_role,
                topic=topic_obj.topic,
                argument=f"I challenge the {target_role.value}'s position on {topic_obj.topic} based on the evidence.",
                cited_evidence=[],
            )

    def _generate_response_turn(
        self,
        topic_obj: DebateTopic,
        speaker_role: AgentRole,
        target_role: AgentRole,
        previous_argument: str,
        speaker_assessment: AgentAssessment,
        evidence_text: str,
        evidence_store: EvidenceStore,
    ) -> DebateTurn:
        """Executes Turn 2: Direct response from speaker back to challenger."""
        system_prompt = DEBATE_RESPONSE_SYSTEM_PROMPT.format(
            speaker_role=speaker_role.value,
            target_role=target_role.value,
            topic=topic_obj.topic,
            previous_argument=previous_argument,
        )

        user_prompt = f"""DISPUTED TOPIC: {topic_obj.topic}
THE {target_role.value} ARGUED:
"{previous_argument}"

YOUR INITIAL STANCE ({speaker_role.value}):
- Recommendation: {speaker_assessment.recommendation}
- Strengths: {'; '.join(speaker_assessment.strengths) or 'None'}
- Concerns: {'; '.join(speaker_assessment.concerns) or 'None'}

EVIDENCE STORE:
{evidence_text}

Respond directly to the {target_role.value}'s argument above.
"""
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = TurnLLMResponse.model_validate(json.loads(raw))
            citations = self._resolve_citations(parsed.cited_evidence_ids, evidence_store)

            return DebateTurn(
                speaker=speaker_role,
                target_agent=target_role,
                topic=topic_obj.topic,
                argument=parsed.argument,
                cited_evidence=citations,
            )
        except Exception as exc:
            logger.error("Error generating debate response: %s", exc)
            return DebateTurn(
                speaker=speaker_role,
                target_agent=target_role,
                topic=topic_obj.topic,
                argument=f"I note the {target_role.value}'s challenge on {topic_obj.topic} and maintain my position based on verified evidence.",
                cited_evidence=[],
            )

