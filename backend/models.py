from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """The four distinct AI panel agents."""
    TECHNICAL = "Technical Agent"
    HR_CULTURE = "HR / Culture Agent"
    HIRING_MANAGER = "Hiring Manager Agent"
    SKEPTIC = "Skeptic Agent"


class DocumentPage(BaseModel):
    """Represents a single page extracted from a PDF document."""
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Raw text extracted from this page")


class ExtractedDocument(BaseModel):
    """Structured representation of a parsed PDF document."""
    document_id: str = Field(..., description="Unique document identifier (e.g. 'doc_job_desc', 'doc_resume_a')")
    source_type: str = Field(..., description="Document category: 'job_description', 'resume', or 'transcript'")
    filename: str = Field(..., description="Original filename of the PDF")
    pages: List[DocumentPage] = Field(default_factory=list, description="Pages extracted from the PDF")


class EvidenceItem(BaseModel):
    """Grounded evidence unit extracted directly from source document text."""
    evidence_id: str = Field(..., description="Deterministic unique identifier e.g. 'E001', 'E002'")
    source_type: str = Field(..., description="Document source: 'job_description', 'resume', or 'transcript'")
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="1-indexed page number where quote originates")
    quote: str = Field(..., description="Exact verbatim text excerpt from the document")
    context: Optional[str] = Field(None, description="Section heading, topic, or context for the excerpt")

    def to_reference(self) -> "EvidenceReference":
        """Converts to an EvidenceReference for citations."""
        return EvidenceReference(
            evidence_id=self.evidence_id,
            source_type=self.source_type,
            quote=self.quote,
            context=self.context or f"{self.filename} (p. {self.page_number})",
        )


class EvidenceReference(BaseModel):
    """Traceable citation to evidence extracted from actual source documents (e.g., E001, E002)."""
    evidence_id: str = Field(..., description="Unique evidence identifier e.g. 'E001', 'E002'")
    source_type: str = Field(..., description="Document source: 'resume', 'transcript', or 'job_description'")
    quote: str = Field(..., description="Verbatim or precise excerpt extracted from the source document")
    context: Optional[str] = Field(None, description="Section heading, page number, or context for the excerpt")


class CandidateProfile(BaseModel):
    """Structured candidate profile extracted from resume and transcript."""
    candidate_id: str = Field(..., description="Identifier e.g. 'Candidate_A' or 'Candidate_B'")
    name: str = Field(..., description="Candidate full name")
    extracted_skills: List[str] = Field(default_factory=list, description="Extracted technical and soft skills")
    experience_summary: str = Field(..., description="Summary of work history and domain background")
    claims: List[str] = Field(default_factory=list, description="Specific assertions made by candidate in resume/interview")
    supporting_evidence: List[EvidenceReference] = Field(default_factory=list, description="Traceable citations supporting profile points")
    insufficient_info_flags: List[str] = Field(default_factory=list, description="Areas where data is missing or ambiguous")


class AgentAssessment(BaseModel):
    """Independent opinion and evaluation produced by an agent before the debate."""
    agent_role: AgentRole = Field(..., description="Agent role conducting assessment")
    score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Domain score from 0 to 10 (None if insufficient info)")
    recommendation: str = Field(..., description="'Strong Hire', 'Hire', 'Weak Hire', 'Reject', or 'Undecided'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    strengths: List[str] = Field(default_factory=list, description="Identified candidate strengths with context")
    concerns: List[str] = Field(default_factory=list, description="Identified candidate red flags or risks")
    evidence_citations: List[EvidenceReference] = Field(default_factory=list, description="Direct citations backing claims")
    insufficient_info_notes: Optional[str] = Field(None, description="Notes on what required details were missing")


class DebateTurn(BaseModel):
    """Single turn or cross-examination in the agent debate."""
    speaker: AgentRole = Field(..., description="Agent presenting the argument")
    target_agent: Optional[AgentRole] = Field(None, description="Target agent being challenged or responded to")
    topic: str = Field(..., description="Focal topic or disputed claim")
    argument: str = Field(..., description="Core argument or counterpoint")
    cited_evidence: List[EvidenceReference] = Field(default_factory=list, description="Evidence used to defend or challenge")


class OpinionReassessment(BaseModel):
    """Post-debate opinion tracking visible shifts in stances."""
    agent_role: AgentRole = Field(..., description="Agent being reassessed")
    original_recommendation: str = Field(..., description="Recommendation before debate")
    revised_recommendation: str = Field(..., description="Recommendation after debate")
    original_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence before debate")
    revised_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence after debate")
    reasons_for_change: str = Field(..., description="Explicit rationale for changing or maintaining stance")
    changed: bool = Field(..., description="True if recommendation or confidence shifted significantly")


class FinalDecision(BaseModel):
    """Synthesized final decision produced by non-averaging reasoning."""
    recommendation: str = Field(..., description="Final hiring decision for candidate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall decision confidence")
    strengths: List[str] = Field(default_factory=list, description="Core decisive strengths")
    concerns: List[str] = Field(default_factory=list, description="Core decisive risks/concerns")
    unresolved_disagreements: List[str] = Field(default_factory=list, description="Points where agents remained in conflict")
    synthesis_rationale: str = Field(..., description="Qualitative reasoning connecting evidence and debate to decision")
    insufficient_information_flags: List[str] = Field(default_factory=list, description="Explicit missing information callouts")


class CandidateEvaluation(BaseModel):
    """Complete evaluation lifecycle for a single candidate."""
    candidate_id: str = Field(..., description="'Candidate_A' or 'Candidate_B'")
    candidate_name: str = Field(..., description="Candidate name")
    profile: CandidateProfile
    initial_assessments: List[AgentAssessment] = Field(default_factory=list, description="Independent opinions")
    debate_transcript: List[DebateTurn] = Field(default_factory=list, description="Debate turns")
    reassessments: List[OpinionReassessment] = Field(default_factory=list, description="Post-debate opinions")
    final_decision: FinalDecision


class PipelineReport(BaseModel):
    """Final output report encompassing both Candidate A and Candidate B."""
    job_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of target job description")
    candidate_a: CandidateEvaluation
    candidate_b: CandidateEvaluation
    comparative_summary: str = Field(..., description="Comparative analysis and final hiring recommendation between candidates")
    unresolved_disagreements: List[str] = Field(default_factory=list, description="Overall unresolved panel disagreements")
    generated_at: str = Field(..., description="ISO 8601 timestamp of pipeline completion")

