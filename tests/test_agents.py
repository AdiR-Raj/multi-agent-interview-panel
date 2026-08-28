"""Unit tests for Stage 3: Four Independent AI Agents."""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    EvidenceItem,
)
from backend.evidence import EvidenceStore
from backend.agents.technical import TechnicalAgent
from backend.agents.hr_culture import HRCultureAgent
from backend.agents.hiring_manager import HiringManagerAgent
from backend.agents.skeptic import SkepticAgent
from backend.agents.runner import run_independent_agents


class TestIndependentAgents(unittest.TestCase):
    """Test suite for isolated, independent agent evaluation."""

    def setUp(self):
        # Build a sample candidate profile
        self.candidate_profile = CandidateProfile(
            candidate_id="Candidate_A",
            name="John Doe",
            extracted_skills=["Python", "FastAPI", "Kubernetes", "PostgreSQL"],
            experience_summary="5 years building high throughput backend services.",
            claims=["Scaled microservices to 50k RPS.", "Led PostgreSQL zero-downtime migration."],
            insufficient_info_flags=["No public code samples available."],
        )

        self.job_description = (
            "Job Title: Staff AI/Backend Engineer\n"
            "Requirements: 5+ years Python, distributed systems, API design, production ownership."
        )

        # Build an evidence store with sample items
        self.evidence_store = EvidenceStore()
        doc = ExtractedDocument(
            document_id="doc_sample",
            source_type="resume",
            filename="resume_john.pdf",
            pages=[
                DocumentPage(
                    page_number=1,
                    text="John Doe - Staff Engineer. Scaled microservices to 50k RPS. Led PostgreSQL migration.",
                )
            ],
        )
        self.evidence_store.add_document(doc)

    def _create_mock_response(self, content_dict):
        """Helper to create a mocked OpenAI chat completion response."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(content_dict)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_all_four_agents_invoked_separately(self):
        """Test 1: Verifies that all 4 distinct agents are executed independently."""
        mock_client = MagicMock()

        # Provide 4 distinct responses for each agent call
        responses = [
            # Technical
            {
                "recommendation": "Hire",
                "confidence": 0.9,
                "score": 8.5,
                "strengths": ["Strong Python & distributed systems"],
                "concerns": ["Limited frontend experience"],
                "cited_evidence_ids": ["E001"],
                "insufficient_info_notes": None,
            },
            # HR / Culture
            {
                "recommendation": "Hire",
                "confidence": 0.85,
                "score": 8.0,
                "strengths": ["Clear communication and team leadership"],
                "concerns": [],
                "cited_evidence_ids": ["E001"],
                "insufficient_info_notes": None,
            },
            # Hiring Manager
            {
                "recommendation": "Strong Hire",
                "confidence": 0.95,
                "score": 9.0,
                "strengths": ["Direct match for scaling requirements"],
                "concerns": [],
                "cited_evidence_ids": ["E001"],
                "insufficient_info_notes": None,
            },
            # Skeptic
            {
                "recommendation": "Weak Hire",
                "confidence": 0.75,
                "score": 6.5,
                "strengths": ["Relevant experience"],
                "concerns": ["50k RPS claim lacks independent benchmark"],
                "cited_evidence_ids": ["E001"],
                "insufficient_info_notes": "Needs deeper probing on individual vs team metrics",
            },
        ]

        mock_client.chat.completions.create.side_effect = [
            self._create_mock_response(r) for r in responses
        ]

        assessments = run_independent_agents(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            client=mock_client,
        )

        # 1. Exactly 4 assessments returned
        self.assertEqual(len(assessments), 4)

        # 2. Exactly 4 calls to the LLM client
        self.assertEqual(mock_client.chat.completions.create.call_count, 4)

        # 3. Roles must match the four required panel roles
        roles = [a.agent_role for a in assessments]
        self.assertEqual(
            roles,
            [
                AgentRole.TECHNICAL,
                AgentRole.HR_CULTURE,
                AgentRole.HIRING_MANAGER,
                AgentRole.SKEPTIC,
            ],
        )

    def test_agents_receive_candidate_info_and_evidence(self):
        """Test 2: Verifies that agents receive job description, candidate profile, and evidence."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Hire",
            "confidence": 0.8,
            "score": 8.0,
            "strengths": ["Solid background"],
            "concerns": [],
            "cited_evidence_ids": ["E001"],
            "insufficient_info_notes": None,
        })

        agent = TechnicalAgent(client=mock_client)
        assessment = agent.evaluate(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
        )

        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        user_content = messages[1]["content"]

        # Ensure job description, candidate name, and evidence citations are in prompt
        self.assertIn("Staff AI/Backend Engineer", user_content)
        self.assertIn("John Doe", user_content)
        self.assertIn("E001", user_content)
        self.assertIn("Scaled microservices to 50k RPS", user_content)

    def test_no_agent_receives_another_agent_assessment(self):
        """Test 3: Critical Independence - verifies prompts contain no other agent's output."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Hire",
            "confidence": 0.8,
            "score": 8.0,
            "strengths": [],
            "concerns": [],
            "cited_evidence_ids": [],
            "insufficient_info_notes": None,
        })

        run_independent_agents(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            client=mock_client,
        )

        # Inspect every prompt sent to LLM
        forbidden_phrases = [
            "Technical Agent assessment",
            "HR / Culture Agent concluded",
            "Hiring Manager score",
            "Skeptic Agent noted",
            "Other agents said",
        ]

        for call in mock_client.chat.completions.create.call_args_list:
            messages = call[1]["messages"]
            prompt_text = " ".join(m["content"] for m in messages)
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase.lower(), prompt_text.lower())

    def test_agent_outputs_conform_to_model_and_citations(self):
        """Test 4 & 5: Verifies AgentAssessment schema and evidence ID traceability."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Strong Hire",
            "confidence": 0.92,
            "score": 9.2,
            "strengths": ["Demonstrated production scalability"],
            "concerns": [],
            "cited_evidence_ids": ["E001"],
            "insufficient_info_notes": None,
        })

        agent = HiringManagerAgent(client=mock_client)
        assessment = agent.evaluate(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
        )

        # Check instance
        self.assertIsInstance(assessment, AgentAssessment)
        self.assertEqual(assessment.agent_role, AgentRole.HIRING_MANAGER)
        self.assertEqual(assessment.recommendation, "Strong Hire")
        self.assertEqual(assessment.confidence, 0.92)
        self.assertEqual(assessment.score, 9.2)

        # Check evidence references contain valid evidence_id
        self.assertEqual(len(assessment.evidence_citations), 1)
        ref = assessment.evidence_citations[0]
        self.assertEqual(ref.evidence_id, "E001")
        self.assertEqual(ref.source_type, "resume")
        self.assertIn("Scaled microservices", ref.quote)

    def test_insufficient_information_produces_undecided(self):
        """Test 6: Missing information produces Undecided recommendation without fake score."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "recommendation": "Undecided",
            "confidence": 0.3,
            "score": None,
            "strengths": [],
            "concerns": ["Critical data missing"],
            "cited_evidence_ids": [],
            "insufficient_info_notes": "Transcript did not cover system design or database architecture.",
        })

        agent = SkepticAgent(client=mock_client)
        assessment = agent.evaluate(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
        )

        self.assertEqual(assessment.recommendation, "Undecided")
        self.assertIsNone(assessment.score)
        self.assertEqual(assessment.confidence, 0.3)
        self.assertIn("Transcript did not cover system design", assessment.insufficient_info_notes)

    def test_api_failure_handled_gracefully(self):
        """Test 7: API network or parse failure returns a safe Undecided assessment."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI API rate limit exceeded")

        agent = TechnicalAgent(client=mock_client)
        assessment = agent.evaluate(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
        )

        self.assertIsInstance(assessment, AgentAssessment)
        self.assertEqual(assessment.recommendation, "Undecided")
        self.assertEqual(assessment.confidence, 0.0)
        self.assertIsNone(assessment.score)
        self.assertTrue(any("failed due to error" in c for c in assessment.concerns))
        self.assertIn("OpenAI API rate limit exceeded", assessment.insufficient_info_notes)


if __name__ == "__main__":
    unittest.main()

