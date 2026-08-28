"""Final Decision Synthesis Engine for Multi-Agent Interview Panel.

Performs qualitative non-averaging synthesis of the candidate's complete evaluation state:
job description, candidate profile, grounded evidence store, initial independent assessments,
debate cross-examination transcript, and post-debate opinion reassessments.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import settings
from backend.models import (
    CandidateProfile,
    AgentAssessment,
    DebateTurn,
    OpinionReassessment,
    FinalDecision,
    EvidenceReference,
)
from backend.evidence import EvidenceStore

logger = logging.getLogger(__name__)

FINAL_DECISION_SYSTEM_PROMPT = """You are the final hiring decision maker on an expert executive panel.

CRITICAL RULES:
- Do NOT average agent scores.
- Do NOT decide by majority vote.
- Do NOT blindly follow the most confident agent.
- Evaluate the grounded evidence, job requirements, panel disagreements, cross-examination arguments, and post-debate opinion changes.
- Every important factual claim must be traceable to provided evidence from the Evidence Store.
- If evidence is insufficient to make a responsible hiring decision, choose 'Undecided' and explicitly list what information is missing.

Allowed recommendations: 'Strong Hire', 'Hire', 'Weak Hire', 'Reject', 'Undecided'.
Confidence must be a float between 0.0 and 1.0.

Respond strictly in JSON format:
{
  "recommendation": "Strong Hire | Hire | Weak Hire | Reject | Undecided",
  "confidence": 0.85,
  "strengths": ["Key decisive strength 1", ...],
  "concerns": ["Key decisive concern or hiring risk 1", ...],
  "unresolved_disagreements": ["Point where agents remained in conflict or uncertainty", ...],
  "synthesis_rationale": "Comprehensive qualitative reasoning explaining how the evidence, debate turns, and reassessed stances led to this final decision without score averaging or voting.",
  "insufficient_information_flags": ["Missing data point 1", ...],
  "decisive_evidence_ids": ["E001", "E004"]
}
"""


class FinalDecisionLLMResponse(BaseModel):
    """Structured output schema for final decision LLM call."""
    recommendation: str = Field(..., description="'Strong Hire', 'Hire', 'Weak Hire', 'Reject', or 'Undecided'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall decision confidence 0.0-1.0")
    strengths: List[str] = Field(default_factory=list, description="Decisive strengths")
    concerns: List[str] = Field(default_factory=list, description="Decisive concerns or risks")
    unresolved_disagreements: List[str] = Field(default_factory=list, description="Unresolved panel conflicts")
    synthesis_rationale: str = Field(..., description="Qualitative reasoning connecting evidence to decision")
    insufficient_information_flags: List[str] = Field(default_factory=list, description="Missing information callouts")
    decisive_evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs directly supporting decision")


class FinalDecisionEngine:
    """Orchestrates non-averaging final decision synthesis."""

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
                logger.warning("Final decision cited unknown evidence ID: %s", eid)
        return citations

    def build_user_prompt(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
        debate_transcript: List[DebateTurn],
        reassessments: List[OpinionReassessment],
    ) -> str:
        """Constructs the comprehensive user prompt containing all stages of evaluation."""
        # 1. Initial Assessments
        init_lines = []
        for a in initial_assessments:
            cites = ", ".join(c.evidence_id for c in a.evidence_citations) or "None"
            score_str = f"{a.score}/10" if a.score is not None else "N/A"
            init_lines.append(
                f"[{a.agent_role.value}]\n"
                f"  Recommendation: {a.recommendation} | Confidence: {a.confidence} | Score: {score_str}\n"
                f"  Strengths: {'; '.join(a.strengths) or 'None'}\n"
                f"  Concerns: {'; '.join(a.concerns) or 'None'}\n"
                f"  Citations: {cites}\n"
                f"  Missing Info: {a.insufficient_info_notes or 'None'}"
            )
        initial_text = "\n\n".join(init_lines)

        # 2. Debate Transcript
        debate_lines = []
        for i, turn in enumerate(debate_transcript, 1):
            target_str = f" -> {turn.target_agent.value}" if turn.target_agent else ""
            cites = ", ".join(c.evidence_id for c in turn.cited_evidence)
            cites_str = f" [Evidence: {cites}]" if cites else ""
            debate_lines.append(
                f"Turn {i} ({turn.speaker.value}{target_str} on topic '{turn.topic}'):\n"
                f"  \"{turn.argument}\"{cites_str}"
            )
        debate_text = "\n\n".join(debate_lines) if debate_lines else "No debate turns were triggered (panel in initial consensus)."

        # 3. Post-Debate Reassessments
        reassess_lines = []
        for r in reassessments:
            changed_str = "CHANGED" if r.changed else "MAINTAINED"
            reassess_lines.append(
                f"[{r.agent_role.value}] ({changed_str})\n"
                f"  Pre-Debate: {r.original_recommendation} (conf: {r.original_confidence}) -> Post-Debate: {r.revised_recommendation} (conf: {r.revised_confidence})\n"
                f"  Rationale: {r.reasons_for_change}"
            )
        reassess_text = "\n\n".join(reassess_lines)

        # 4. Evidence Store
        evidence_text = evidence_store.format_for_prompt()

        return f"""=== TARGET JOB DESCRIPTION ===
{job_description}

=== CANDIDATE PROFILE ===
Candidate ID: {candidate_profile.candidate_id}
Candidate Name: {candidate_profile.name}
Extracted Skills: {', '.join(candidate_profile.extracted_skills) or 'None'}
Experience Summary: {candidate_profile.experience_summary}
Known Missing Info Flags: {', '.join(candidate_profile.insufficient_info_flags) or 'None'}

=== 1. PRE-DEBATE INDEPENDENT ASSESSMENTS ===
{initial_text}

=== 2. PANEL DEBATE PROCEEDINGS ===
{debate_text}

=== 3. POST-DEBATE OPINION REASSESSMENTS ===
{reassess_text}

=== 4. VERIFIED EVIDENCE STORE (AVAILABLE CITATIONS) ===
{evidence_text}

=== TASK ===
Synthesize all the above information and produce the authoritative Final Hiring Decision.
Do not calculate an average score or rely on voting. Provide rigorous qualitative reasoning.
"""

    def synthesize(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
        initial_assessments: List[AgentAssessment],
        debate_transcript: List[DebateTurn],
        reassessments: List[OpinionReassessment],
    ) -> FinalDecision:
        """Executes the separate final decision reasoning call."""
        user_prompt = self.build_user_prompt(
            job_description=job_description,
            candidate_profile=candidate_profile,
            evidence_store=evidence_store,
            initial_assessments=initial_assessments,
            debate_transcript=debate_transcript,
            reassessments=reassessments,
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": FINAL_DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = FinalDecisionLLMResponse.model_validate(json.loads(raw))
            citations = self._resolve_citations(parsed.decisive_evidence_ids, evidence_store)

            return FinalDecision(
                recommendation=parsed.recommendation,
                confidence=parsed.confidence,
                strengths=parsed.strengths,
                concerns=parsed.concerns,
                unresolved_disagreements=parsed.unresolved_disagreements,
                synthesis_rationale=parsed.synthesis_rationale,
                insufficient_information_flags=parsed.insufficient_information_flags,
                decisive_evidence=citations,
            )

        except Exception as exc:
            logger.error("Error synthesizing final decision: %s", exc)
            return FinalDecision(
                recommendation="Undecided",
                confidence=0.0,
                strengths=[],
                concerns=[f"Final decision synthesis failed: {str(exc)}"],
                unresolved_disagreements=[],
                synthesis_rationale=f"Synthesis could not be completed due to error: {str(exc)}",
                insufficient_information_flags=["Evaluation pipeline error encountered."],
                decisive_evidence=[],
            )

