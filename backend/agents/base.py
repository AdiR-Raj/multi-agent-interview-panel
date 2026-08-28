"""Base Agent Abstraction for Interview Panel Simulator.

Provides common LLM client communication, prompt building, response parsing,
and citation validation while enforcing independent execution and isolated prompts.
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
    EvidenceReference,
)
from backend.evidence import EvidenceStore

logger = logging.getLogger(__name__)


class LLMAssessmentResponse(BaseModel):
    """Schema for structured LLM assessment generation."""
    recommendation: str = Field(..., description="'Strong Hire', 'Hire', 'Weak Hire', 'Reject', or 'Undecided'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")
    score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Domain score 0-10 or null if insufficient info")
    strengths: List[str] = Field(default_factory=list, description="Key strengths identified")
    concerns: List[str] = Field(default_factory=list, description="Key risks, gaps, or concerns identified")
    cited_evidence_ids: List[str] = Field(default_factory=list, description="List of evidence IDs cited (e.g. ['E001', 'E004'])")
    insufficient_info_notes: Optional[str] = Field(None, description="Notes on missing or ambiguous data")


class BaseAgent:
    """Base class for independent interview panel agents."""

    def __init__(self, role: AgentRole, system_prompt: str, client: Optional[OpenAI] = None):
        self.role = role
        self.system_prompt = system_prompt
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            # Fall back to a dummy key if not set, so initialization doesn't crash before mock
            api_key = settings.OPENAI_API_KEY or "missing-api-key"
            self._client = OpenAI(
                api_key=api_key,
                base_url=settings.OPENAI_BASE_URL,
            )
        return self._client

    def build_user_prompt(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
    ) -> str:
        """Constructs the prompt containing only job description, candidate profile, and evidence."""
        evidence_text = evidence_store.format_for_prompt()

        skills_list = ", ".join(candidate_profile.extracted_skills) if candidate_profile.extracted_skills else "None listed"
        claims_list = "\n".join(f"- {c}" for c in candidate_profile.claims) if candidate_profile.claims else "None listed"
        missing_flags = "\n".join(f"- {m}" for m in candidate_profile.insufficient_info_flags) if candidate_profile.insufficient_info_flags else "None recorded"

        return f"""=== TARGET JOB DESCRIPTION ===
{job_description}

=== CANDIDATE PROFILE ===
Candidate ID: {candidate_profile.candidate_id}
Candidate Name: {candidate_profile.name}
Extracted Skills: {skills_list}

Experience Summary:
{candidate_profile.experience_summary}

Candidate Claims:
{claims_list}

Known Missing / Ambiguous Information:
{missing_flags}

=== VERIFIED EVIDENCE STORE (AVAILABLE CITATIONS) ===
{evidence_text}

=== INSTRUCTIONS ===
1. Evaluate this candidate STRICTLY from your perspective as the {self.role.value}.
2. You must ONLY cite evidence using existing evidence IDs from the Verified Evidence Store above (e.g. 'E001', 'E002'). Do NOT invent evidence IDs or quotes.
3. Every strength, concern, and claim should cite the relevant evidence ID(s).
4. If available evidence is insufficient to evaluate critical requirements, choose recommendation 'Undecided', set score to null, and explain in 'insufficient_info_notes'.
5. Recommendations must be exactly one of: 'Strong Hire', 'Hire', 'Weak Hire', 'Reject', 'Undecided'.
6. Confidence must be between 0.0 and 1.0. Score (if provided) must be between 0.0 and 10.0.

Respond ONLY with a valid JSON object adhering to this structure:
{{
  "recommendation": "Strong Hire | Hire | Weak Hire | Reject | Undecided",
  "confidence": 0.85,
  "score": 8.0,
  "strengths": ["string", ...],
  "concerns": ["string", ...],
  "cited_evidence_ids": ["E001", "E003", ...],
  "insufficient_info_notes": "string or null"
}}
"""

    def evaluate(
        self,
        job_description: str,
        candidate_profile: CandidateProfile,
        evidence_store: EvidenceStore,
    ) -> AgentAssessment:
        """Executes an isolated, independent LLM call to assess the candidate."""
        user_prompt = self.build_user_prompt(job_description, candidate_profile, evidence_store)

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            parsed_data = json.loads(raw_content)
            parsed_resp = LLMAssessmentResponse.model_validate(parsed_data)

            # Map cited evidence IDs to verified EvidenceReference objects
            citations: List[EvidenceReference] = []
            for eid in parsed_resp.cited_evidence_ids:
                item = evidence_store.get_by_id(eid)
                if item:
                    citations.append(item.to_reference())
                else:
                    logger.warning("Agent %s cited unknown evidence ID: %s", self.role.value, eid)

            return AgentAssessment(
                agent_role=self.role,
                score=parsed_resp.score,
                recommendation=parsed_resp.recommendation,
                confidence=parsed_resp.confidence,
                strengths=parsed_resp.strengths,
                concerns=parsed_resp.concerns,
                evidence_citations=citations,
                insufficient_info_notes=parsed_resp.insufficient_info_notes,
            )

        except Exception as exc:
            logger.error("Error running independent assessment for %s: %s", self.role.value, exc)
            return AgentAssessment(
                agent_role=self.role,
                score=None,
                recommendation="Undecided",
                confidence=0.0,
                strengths=[],
                concerns=[f"Assessment failed due to error: {str(exc)}"],
                evidence_citations=[],
                insufficient_info_notes=f"Evaluation could not complete: {str(exc)}",
            )

