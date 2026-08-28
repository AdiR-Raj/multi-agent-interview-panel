"""Unit tests for Stage 5: Final Decision Engine."""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models import (
    AgentRole,
    AgentAssessment,
    CandidateProfile,
    ExtractedDocument,
    DocumentPage,
    DebateTurn,
    OpinionReassessment,
    FinalDecision,
    EvidenceReference,
)
from backend.evidence import EvidenceStore
from backend.decision.final_decision import FinalDecisionEngine
from backend.decision import make_final_decision


class TestFinalDecisionEngine(unittest.TestCase):
    """Test suite for Stage 5 Final Decision reasoning."""

    def setUp(self):
        self.candidate_profile = CandidateProfile(
            candidate_id="Candidate_A",
            name="Rohan Malhotra",
            extracted_skills=["Python", "FastAPI", "Multi-Agent Systems", "MongoDB"],
            experience_summary="3.5 years building multi-agent systems and high throughput Python backends.",
            claims=["Designed exception handling engine for freight operations."],
            insufficient_info_flags=[],
        )

        self.job_description = "Job Description: AI Engineer - Agentic Systems for Freight Operations."

        # Setup EvidenceStore
        self.evidence_store = EvidenceStore()
        doc = ExtractedDocument(
            document_id="doc_resume_rohan",
            source_type="resume",
            filename="resume_rohan.pdf",
            pages=[
                DocumentPage(
                    page_number=1,
                    text="Rohan Malhotra. Designed exception handling engine for freight exceptions. Python, FastAPI, MongoDB.",
                )
            ],
        )
        self.evidence_store.add_document(doc)

        # 4 Pre-debate assessments with varied scores & recommendations
        self.initial_assessments = [
            AgentAssessment(
                agent_role=AgentRole.TECHNICAL,
                score=9.0,
                recommendation="Strong Hire",
                confidence=0.90,
                strengths=["Excellent multi-agent architecture background"],
                concerns=[],
                evidence_citations=[
                    EvidenceReference(evidence_id="E001", source_type="resume", quote="Designed exception handling engine")
                ],
            ),
            AgentAssessment(
                agent_role=AgentRole.HR_CULTURE,
                score=7.0,
                recommendation="Hire",
                confidence=0.80,
                strengths=["Clear communicator"],
                concerns=["Fast mover risk"],
                evidence_citations=[],
            ),
            AgentAssessment(
                agent_role=AgentRole.HIRING_MANAGER,
                score=9.5,
                recommendation="Strong Hire",
                confidence=0.95,
                strengths=["Immediate freight domain fit"],
                concerns=[],
                evidence_citations=[],
            ),
            AgentAssessment(
                agent_role=AgentRole.SKEPTIC,
                score=5.0,
                recommendation="Weak Hire",
                confidence=0.70,
                strengths=["Keywords match"],
                concerns=["Unverified production scale metrics"],
                evidence_citations=[],
            ),
        ]

        # Debate transcript
        self.debate_transcript = [
            DebateTurn(
                speaker=AgentRole.SKEPTIC,
                target_agent=AgentRole.TECHNICAL,
                topic="Production Autonomy vs Scale",
                argument="E001 demonstrates engine design but doesn't prove individual ownership of the 50k RPS scale.",
                cited_evidence=[
                    EvidenceReference(evidence_id="E001", source_type="resume", quote="Designed exception handling engine")
                ],
            ),
            DebateTurn(
                speaker=AgentRole.TECHNICAL,
                target_agent=AgentRole.SKEPTIC,
                topic="Production Autonomy vs Scale",
                argument="Agreed that 50k RPS was team-level, but the core routing and agent logic was individually designed.",
                cited_evidence=[
                    EvidenceReference(evidence_id="E001", source_type="resume", quote="Designed exception handling engine")
                ],
            ),
        ]

        # Reassessments
        self.reassessments = [
            OpinionReassessment(
                agent_role=AgentRole.TECHNICAL,
                original_recommendation="Strong Hire",
                revised_recommendation="Hire",
                original_confidence=0.90,
                revised_confidence=0.82,
                reasons_for_change="Conceded scale metric ambiguity to Skeptic.",
                changed=True,
            ),
            OpinionReassessment(
                agent_role=AgentRole.HR_CULTURE,
                original_recommendation="Hire",
                revised_recommendation="Hire",
                original_confidence=0.80,
                revised_confidence=0.80,
                reasons_for_change="No cultural risks raised during debate.",
                changed=False,
            ),
            OpinionReassessment(
                agent_role=AgentRole.HIRING_MANAGER,
                original_recommendation="Strong Hire",
                revised_recommendation="Strong Hire",
                original_confidence=0.95,
                revised_confidence=0.95,
                reasons_for_change="Domain expertise in freight remains uniquely high priority.",
                changed=False,
            ),
            OpinionReassessment(
                agent_role=AgentRole.SKEPTIC,
                original_recommendation="Weak Hire",
                revised_recommendation="Weak Hire",
                original_confidence=0.70,
                revised_confidence=0.60,
                reasons_for_change="Maintain concern over lack of independent verification.",
                changed=True,
            ),
        ]

    def _create_mock_response(self, content_dict):
        """Helper to mock OpenAI response."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(content_dict)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_final_decision_receives_all_inputs_in_separate_call(self):
        """Test 1 & 2: Receives job description, profile, evidence, initial assessments, debate, and reassessments."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Hire",
            "confidence": 0.88,
            "strengths": ["Deep domain relevance in freight agentic systems", "Strong technical foundations"],
            "concerns": ["Need close onboarding oversight on production reliability metrics"],
            "unresolved_disagreements": ["Skeptic maintains reservation over team vs individual scale metrics"],
            "synthesis_rationale": "While the Skeptic correctly challenged scale metrics during debate, the candidate's core multi-agent engineering directly solves Cargonet AI's immediate production needs.",
            "insufficient_information_flags": [],
            "decisive_evidence_ids": ["E001"],
        })

        decision = make_final_decision(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_transcript=self.debate_transcript,
            reassessments=self.reassessments,
            client=mock_client,
        )

        # 1. Exactly 1 separate LLM call made
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

        # 2. Verify all inputs were provided in prompt
        call_prompt = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("TARGET JOB DESCRIPTION", call_prompt)
        self.assertIn("Rohan Malhotra", call_prompt)
        self.assertIn("PRE-DEBATE INDEPENDENT ASSESSMENTS", call_prompt)
        self.assertIn("PANEL DEBATE PROCEEDINGS", call_prompt)
        self.assertIn("POST-DEBATE OPINION REASSESSMENTS", call_prompt)
        self.assertIn("E001", call_prompt)

    def test_no_score_averaging_or_mathematical_voting(self):
        """Test 3 & 4: Verifies the decision logic is qualitative and not an arithmetic mean or majority count."""
        # Average of scores: (9.0 + 7.0 + 9.5 + 5.0) / 4 = 7.625
        # Average of confidences: (0.90 + 0.80 + 0.95 + 0.70) / 4 = 0.8375
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Strong Hire",
            "confidence": 0.91,  # Distinct from 0.8375
            "strengths": ["Decisive freight domain competence"],
            "concerns": ["Scale metric clarity"],
            "unresolved_disagreements": ["Skeptic scale dispute"],
            "synthesis_rationale": "Qualitative synthesis prioritized Hiring Manager and Technical requirements over pure consensus.",
            "insufficient_information_flags": [],
            "decisive_evidence_ids": ["E001"],
        })

        decision = make_final_decision(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_transcript=self.debate_transcript,
            reassessments=self.reassessments,
            client=mock_client,
        )

        self.assertIsInstance(decision, FinalDecision)
        self.assertEqual(decision.recommendation, "Strong Hire")
        self.assertEqual(decision.confidence, 0.91)
        # Verify no artificial score averaging occurred
        self.assertNotEqual(decision.confidence, 0.8375)

    def test_evidence_id_validation_and_safety(self):
        """Test 5 & 6: Valid evidence IDs are resolved into EvidenceReferences, invalid IDs are filtered safely."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Hire",
            "confidence": 0.85,
            "strengths": ["Grounded skills"],
            "concerns": [],
            "unresolved_disagreements": [],
            "synthesis_rationale": "Evidence supports hire.",
            "insufficient_information_flags": [],
            "decisive_evidence_ids": ["E001", "E999_NONEXISTENT"],  # One valid, one invalid
        })

        decision = make_final_decision(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_transcript=self.debate_transcript,
            reassessments=self.reassessments,
            client=mock_client,
        )

        # Only E001 is mapped
        self.assertEqual(len(decision.decisive_evidence), 1)
        self.assertEqual(decision.decisive_evidence[0].evidence_id, "E001")
        self.assertIn("Designed exception handling engine", decision.decisive_evidence[0].quote)

    def test_unresolved_disagreements_and_insufficient_info(self):
        """Test 7 & 8: Preserves unresolved disagreements and supports Undecided for insufficient info."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Undecided",
            "confidence": 0.40,
            "strengths": ["Basic framework familiarity"],
            "concerns": ["Missing architectural and production depth"],
            "unresolved_disagreements": ["Skeptic vs Hiring Manager on readiness for standalone ownership"],
            "synthesis_rationale": "The panel could not reach conclusive verification due to complete absence of production interview testing.",
            "insufficient_information_flags": ["No data on live incident handling", "No code sample verification"],
            "decisive_evidence_ids": [],
        })

        decision = make_final_decision(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_transcript=self.debate_transcript,
            reassessments=self.reassessments,
            client=mock_client,
        )

        self.assertEqual(decision.recommendation, "Undecided")
        self.assertEqual(decision.confidence, 0.40)
        self.assertEqual(len(decision.unresolved_disagreements), 1)
        self.assertIn("Skeptic vs Hiring Manager", decision.unresolved_disagreements[0])
        self.assertEqual(len(decision.insufficient_information_flags), 2)

    def test_api_failure_handled_gracefully(self):
        """Test 9 & 10: API errors return a safe Undecided FinalDecision conforming to model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI 503 Service Unavailable")

        decision = make_final_decision(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_transcript=self.debate_transcript,
            reassessments=self.reassessments,
            client=mock_client,
        )

        self.assertIsInstance(decision, FinalDecision)
        self.assertEqual(decision.recommendation, "Undecided")
        self.assertEqual(decision.confidence, 0.0)
        self.assertTrue(any("failed" in c for c in decision.concerns))
        self.assertIn("OpenAI 503 Service Unavailable", decision.synthesis_rationale)


if __name__ == "__main__":
    unittest.main()

