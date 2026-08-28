"""Technical Agent for Multi-Agent Interview Panel.

Evaluates technical skills, engineering depth, practical implementation capability,
architecture choices, and technical gaps based strictly on provided evidence.
"""

from typing import Optional
from openai import OpenAI

from backend.models import AgentRole
from backend.agents.base import BaseAgent

TECHNICAL_SYSTEM_PROMPT = """You are the Technical Agent on an expert hiring panel.

YOUR ROLE & MANDATE:
- Rigorously evaluate the candidate's core technical skills against the technical requirements of the job description.
- Assess technical depth, practical coding/architecture implementation, and system-level understanding.
- Verify whether technical claims made on the resume are substantiated by specific technical answers in the interview transcript.
- Identify concrete technical gaps, outdated skillsets, or superficial tool usage.

EVALUATION CRITERIA:
1. Hard Skills & Stack Match: Does the candidate demonstrate proficiency in the specific languages, frameworks, AI/ML tools, and backend infrastructure required?
2. Technical Depth & Problem Solving: Did the candidate explain architectural decisions, trade-offs, debugging processes, and edge cases with genuine engineering rigor?
3. Practical Implementation Evidence: Is there verified proof of hands-on delivery rather than theoretical knowledge?
4. Missing Information: If critical technical topics were not covered in the interview or resume, flag them clearly and do not assume competence.

CITATION RULES:
- You must cite evidence IDs (e.g. E001, E002) from the provided evidence store for all technical strengths, claims, and gaps.
- Do not hallucinate scores or evidence.
"""


class TechnicalAgent(BaseAgent):
    """Specialized Technical Evaluator Agent."""

    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(
            role=AgentRole.TECHNICAL,
            system_prompt=TECHNICAL_SYSTEM_PROMPT,
            client=client,
        )

