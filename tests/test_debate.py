"""Unit tests for Stage 4: Debate + Opinion Reassessment."""

import sys
import json
import copy
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
    DebateTopic,
    DebateTurn,
    OpinionReassessment,
    EvidenceReference,
)
from backend.evidence import EvidenceStore
from backend.debate.coordinator import DebateCoordinator
from backend.debate.debate_agent import DebateEngine
from backend.debate.reassessment import OpinionReassessmentEngine
from backend.debate import run_debate_and_reassessment


class TestDebateAndReassessment(unittest.TestCase):
    """Comprehensive test suite for Stage 4 debate and reassessment."""

    def setUp(self):
        self.candidate_profile = CandidateProfile(
            candidate_id="Candidate_A",
            name="Rohan Malhotra",
            extracted_skills=["Python", "FastAPI", "Multi-Agent Systems", "MongoDB"],
            experience_summary="3.5 years building multi-agent systems and high throughput Python backends.",
            claims=["Designed exception handling engine for thousands of freight exceptions."],
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

        # Initial assessments from 4 agents
        self.initial_assessments = [
            AgentAssessment(
                agent_role=AgentRole.TECHNICAL,
                score=9.0,
                recommendation="Strong Hire",
                confidence=0.9,
                strengths=["Excellent multi-agent architecture background"],
                concerns=[],
                evidence_citations=[
                    EvidenceReference(
                        evidence_id="E001",
                        source_type="resume",
                        quote="Designed exception handling engine",
                        context="resume_rohan.pdf (p. 1)",
                    )
                ],
            ),
            AgentAssessment(
                agent_role=AgentRole.HR_CULTURE,
                score=7.5,
                recommendation="Hire",
                confidence=0.8,
                strengths=["Direct communicator"],
                concerns=["Fast mover, might rush"],
                evidence_citations=[],
            ),
            AgentAssessment(
                agent_role=AgentRole.HIRING_MANAGER,
                score=9.0,
                recommendation="Strong Hire",
                confidence=0.92,
                strengths=["Immediate domain and freight relevance"],
                concerns=[],
                evidence_citations=[],
            ),
            AgentAssessment(
                agent_role=AgentRole.SKEPTIC,
                score=5.5,
                recommendation="Weak Hire",
                confidence=0.75,
                strengths=["Relevant keywords"],
                concerns=["Only 3.5 years experience, claims massive scale with limited independent verification"],
                evidence_citations=[],
                insufficient_info_notes="Need more evidence on production fault-tolerance",
            ),
        ]

    def _create_mock_response(self, content_dict):
        """Helper to mock an OpenAI chat completion response."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(content_dict)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_coordinator_receives_all_four_opinions_and_finds_disagreements(self):
        """Test 1 & 2: Coordinator receives all 4 opinions and extracts real disagreements."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_response({
            "topics": [
                {
                    "topic": "Claimed Production Scale vs Seniority Depth",
                    "agents_involved": ["Technical Agent", "Skeptic Agent"],
                    "disagreement_description": "Technical Agent rates Strong Hire based on architecture claims, while Skeptic flags lack of production verification.",
                    "relevant_evidence_ids": ["E001"],
                }
            ]
        })

        coordinator = DebateCoordinator(client=mock_client)
        topics = coordinator.identify_disagreements(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
        )

        # 1. Verify coordinator sent all 4 assessments in prompt
        call_prompt = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
        self.assertIn("Technical Agent", call_prompt)
        self.assertIn("HR / Culture Agent", call_prompt)
        self.assertIn("Hiring Manager Agent", call_prompt)
        self.assertIn("Skeptic Agent", call_prompt)

        # 2. Verify identified topic
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0].topic, "Claimed Production Scale vs Seniority Depth")
        self.assertEqual(topics[0].agents_involved, [AgentRole.TECHNICAL, AgentRole.SKEPTIC])
        self.assertEqual(topics[0].relevant_evidence_ids, ["E001"])

    def test_debate_turns_structure_and_direct_responses(self):
        """Test 3 & 4: Turns have distinct speaker/target and direct response structure."""
        mock_client = MagicMock()

        # Turn 1: Skeptic -> Technical
        turn1_resp = {
            "argument": "Technical Agent gave Strong Hire based on E001, but E001 only claims the system was built without proving individual autonomy.",
            "cited_evidence_ids": ["E001"],
        }
        # Turn 2: Technical -> Skeptic
        turn2_resp = {
            "argument": "I acknowledge the Skeptic's point on E001. While the candidate led the design, the exact scale metrics were team-level.",
            "cited_evidence_ids": ["E001"],
        }

        mock_client.chat.completions.create.side_effect = [
            self._create_mock_response(turn1_resp),
            self._create_mock_response(turn2_resp),
        ]

        topic = DebateTopic(
            topic="Production Autonomy vs Scale",
            agents_involved=[AgentRole.SKEPTIC, AgentRole.TECHNICAL],
            disagreement_description="Skeptic questions solo contribution.",
            relevant_evidence_ids=["E001"],
        )

        engine = DebateEngine(client=mock_client)
        turns = engine.generate_debate_turns(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            topics=[topic],
        )

        # Must have exactly 2 turns for 1 topic
        self.assertEqual(len(turns), 2)

        # Turn 1
        self.assertEqual(turns[0].speaker, AgentRole.SKEPTIC)
        self.assertEqual(turns[0].target_agent, AgentRole.TECHNICAL)
        self.assertNotEqual(turns[0].speaker, turns[0].target_agent)

        # Turn 2
        self.assertEqual(turns[1].speaker, AgentRole.TECHNICAL)
        self.assertEqual(turns[1].target_agent, AgentRole.SKEPTIC)
        self.assertNotEqual(turns[1].speaker, turns[1].target_agent)

        # Verify Turn 2 LLM prompt received Turn 1's argument for direct response
        second_call_prompt = mock_client.chat.completions.create.call_args_list[1][1]["messages"][1]["content"]
        self.assertIn("Technical Agent gave Strong Hire based on E001", second_call_prompt)

    def test_evidence_citations_validated_and_invalid_ids_handled(self):
        """Test 5 & 11: Valid citations resolved, invalid IDs safely filtered."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._create_mock_response({
                "argument": "Looking at E001 and non-existent E999.",
                "cited_evidence_ids": ["E001", "E999"],  # E999 does not exist
            }),
            self._create_mock_response({
                "argument": "Direct counter-point referencing invalid E888.",
                "cited_evidence_ids": ["E888"],
            }),
        ]

        topic = DebateTopic(
            topic="Evidence Validation Test",
            agents_involved=[AgentRole.TECHNICAL, AgentRole.SKEPTIC],
            disagreement_description="Test citation parsing",
            relevant_evidence_ids=["E001"],
        )

        engine = DebateEngine(client=mock_client)
        turns = engine.generate_debate_turns(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            topics=[topic],
        )

        # E001 is kept, E999 is ignored
        self.assertEqual(len(turns[0].cited_evidence), 1)
        self.assertEqual(turns[0].cited_evidence[0].evidence_id, "E001")

        # E888 is ignored, resulting in empty list without crashing
        self.assertEqual(len(turns[1].cited_evidence), 0)

    def test_initial_assessments_remain_unmodified(self):
        """Test 6: Critical Independence - initial assessment objects remain strictly unchanged."""
        original_copy = copy.deepcopy(self.initial_assessments)

        mock_client = MagicMock()
        # Coordinator mock (no topics) + 4 reassessment responses
        mock_client.chat.completions.create.side_effect = [
            self._create_mock_response({"topics": []}),
            self._create_mock_response({"revised_recommendation": "Strong Hire", "revised_confidence": 0.9, "reasons_for_change": "Firm"}),
            self._create_mock_response({"revised_recommendation": "Hire", "revised_confidence": 0.8, "reasons_for_change": "Firm"}),
            self._create_mock_response({"revised_recommendation": "Strong Hire", "revised_confidence": 0.92, "reasons_for_change": "Firm"}),
            self._create_mock_response({"revised_recommendation": "Weak Hire", "revised_confidence": 0.75, "reasons_for_change": "Firm"}),
        ]

        run_debate_and_reassessment(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            client=mock_client,
        )

        # Check deep equality with original copy
        for original, current in zip(original_copy, self.initial_assessments):
            self.assertEqual(original.recommendation, current.recommendation)
            self.assertEqual(original.confidence, current.confidence)
            self.assertEqual(original.score, current.score)
            self.assertEqual(original.strengths, current.strengths)
            self.assertEqual(original.concerns, current.concerns)

    def test_reassessment_detects_opinion_change_and_preserves_stability(self):
        """Test 7, 8, 9, 10: Opinion shifts are detected, stable opinions remain unchanged."""
        mock_client = MagicMock()

        # Reassessment responses for: Technical (changes), HR (stays), HM (stays), Skeptic (adjusts confidence)
        mock_client.chat.completions.create.side_effect = [
            # Technical changes from Strong Hire (0.90) to Hire (0.80)
            self._create_mock_response({
                "revised_recommendation": "Hire",
                "revised_confidence": 0.80,
                "reasons_for_change": "Conceded to Skeptic that metrics were partially team-level.",
            }),
            # HR stays Hire (0.80)
            self._create_mock_response({
                "revised_recommendation": "Hire",
                "revised_confidence": 0.80,
                "reasons_for_change": "No debate points challenged communication or culture fit.",
            }),
            # Hiring Manager stays Strong Hire (0.92)
            self._create_mock_response({
                "revised_recommendation": "Strong Hire",
                "revised_confidence": 0.92,
                "reasons_for_change": "Delivery velocity and freight fit remain decisive for immediate business needs.",
            }),
            # Skeptic stays Weak Hire, but lowers confidence from 0.75 to 0.60 (delta >= 0.08)
            self._create_mock_response({
                "revised_recommendation": "Weak Hire",
                "revised_confidence": 0.60,
                "reasons_for_change": "Unresolved concerns remain on production monitoring.",
            }),
        ]

        engine = OpinionReassessmentEngine(client=mock_client)
        reassessments = engine.reassess_all(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_turns=[],
        )

        self.assertEqual(len(reassessments), 4)

        # 1. Technical Agent changed recommendation
        tech_re = reassessments[0]
        self.assertEqual(tech_re.agent_role, AgentRole.TECHNICAL)
        self.assertEqual(tech_re.original_recommendation, "Strong Hire")
        self.assertEqual(tech_re.revised_recommendation, "Hire")
        self.assertTrue(tech_re.changed)

        # 2. HR Agent maintained exact stance
        hr_re = reassessments[1]
        self.assertEqual(hr_re.original_recommendation, "Hire")
        self.assertEqual(hr_re.revised_recommendation, "Hire")
        self.assertEqual(hr_re.original_confidence, 0.80)
        self.assertEqual(hr_re.revised_confidence, 0.80)
        self.assertFalse(hr_re.changed)  # System does not force change

        # 3. Hiring Manager maintained exact stance
        hm_re = reassessments[2]
        self.assertEqual(hm_re.original_recommendation, "Strong Hire")
        self.assertEqual(hm_re.revised_recommendation, "Strong Hire")
        self.assertFalse(hm_re.changed)

        # 4. Skeptic maintained recommendation but changed confidence (0.75 -> 0.60, delta=0.15 >= 0.08)
        sk_re = reassessments[3]
        self.assertEqual(sk_re.original_recommendation, "Weak Hire")
        self.assertEqual(sk_re.revised_recommendation, "Weak Hire")
        self.assertEqual(sk_re.original_confidence, 0.75)
        self.assertEqual(sk_re.revised_confidence, 0.60)
        self.assertTrue(sk_re.changed)

    def test_api_failure_handled_gracefully(self):
        """Test 12: Network or API failure returns safe fallback without crashing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI rate limit")

        coordinator = DebateCoordinator(client=mock_client)
        topics = coordinator.identify_disagreements(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
        )
        self.assertEqual(topics, [])

        engine = OpinionReassessmentEngine(client=mock_client)
        reassessments = engine.reassess_all(
            job_description=self.job_description,
            candidate_profile=self.candidate_profile,
            evidence_store=self.evidence_store,
            initial_assessments=self.initial_assessments,
            debate_turns=[],
        )
        self.assertEqual(len(reassessments), 4)
        for r in reassessments:
            self.assertFalse(r.changed)
            self.assertEqual(r.original_recommendation, r.revised_recommendation)


if __name__ == "__main__":
    unittest.main()
