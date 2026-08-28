# Multi-Agent AI Interview Panel Simulator

A deterministic multi-agent evaluation platform that simulates an interview debrief panel. The system evaluates two candidates against a job description using four independent AI agents, a structured cross-examination debate stage, tracked opinion reassessments, and a qualitative final decision synthesis.

---

## 1. What the Project Does

The system processes:
- **1 Job Description PDF**
- **2 Candidate Resume PDFs** (Candidate A & Candidate B)
- **2 Interview Transcript PDFs** (Candidate A & Candidate B)

For each candidate, the simulator:
1. **Extracts & Ground Evidence:** Parses PDFs and builds structured Candidate Profiles with traceable verbatim citations.
2. **Conducts Independent Assessments:** Deploys 4 specialized AI agents that evaluate the candidate independently in separate LLM calls without seeing each other's opinions.
3. **Executes a Panel Debate:** The agents cross-examine one another, challenging claims, questioning interpretations, and presenting counter-evidence.
4. **Tracks Opinion Reassessments:** Explicitly records whether agents revised their recommendations or adjusted confidence levels after debate.
5. **Synthesizes Final Decisions:** Produces a qualitative decision and comparative ranking (without score averaging) with flagged insufficient information areas.

---

## 2. System Architecture

The architecture enforces a strictly deterministic, modular pipeline without heavy frameworks or unnecessary abstractions:

```
PDF Input Files
(1 Job Description, 2 Resumes, 2 Transcripts)
       │
       ▼
[ PyMuPDF Text Extraction & Normalization ]
       │
       ▼
[ Candidate Profile Builder & Evidence Store ]
       │
       ▼
[ 4 Independent Agent Assessments ]
  ├─ Technical Agent
  ├─ HR / Culture Agent
  ├─ Hiring Manager Agent
  └─ Skeptic Agent
       │
       ▼
[ Structured Panel Debate Stage ]
       │
       ▼
[ Opinion Reassessment Stage (Delta Tracking) ]
       │
       ▼
[ Final Decision Reasoning (Non-Averaging Synthesis) ]
       │
       ▼
[ Interactive UI & Final Comparative Report ]
```

### Core Principles
- **Genuinely Independent Agents:** Separate system prompts and isolated execution contexts ensure distinct perspectives.
- **Evidence Traceability:** Every score, strength, concern, and claim is grounded with source citations (`resume`, `transcript`, `job_description`).
- **Visible Opinion Reassessment:** Pre- and post-debate positions are explicitly compared.
- **Qualitative Final Synthesis:** Decisions are reached via logical reasoning rather than naive arithmetic averages.
- **Explicit Insufficient Information Handling:** Flags missing data rather than hallucinating details or scores.
- **Lightweight & Fast:** Built with Python 3.12, FastAPI, Pydantic, and native OpenAI-compatible API client without complex orchestration frameworks or vector DBs.

---

## 3. Document Processing & Evidence Store

The document extraction and evidence foundation provides deterministic grounding for all subsequent agent stages:
- **PyMuPDF Ingestion (`backend/pdf_utils.py`)**: Parses PDF files from disk or in-memory uploads, preserving 1-indexed page boundaries and layout text.
- **Deterministic Evidence Records (`backend/evidence.py`)**: Converts raw text into verbatim, un-paraphrased quotes assigned deterministic identifiers (`E001`, `E002`, `E003`, ...).
- **Full Traceability**: Every `EvidenceItem` stores its source type (`job_description`, `resume`, `transcript`), original filename, page number, verbatim quote, and contextual location tag for precise auditability.
- **EvidenceStore**: In-memory registry enabling fast lookup by ID, document filtering, source filtering, and formatted prompt generation.

---

## 4. Planned Agent Pipeline

### The 4 Panel Agents
1. **Technical Agent:** Assesses technical depth, architecture competence, problem-solving ability, and verified skills.
2. **HR / Culture Agent:** Evaluates communication, collaboration, values alignment, and career trajectory.
3. **Hiring Manager Agent:** Evaluates business impact, role execution, delivery speed, and overall fit for team needs.
4. **Skeptic Agent:** Proactively seeks out resume-transcript discrepancies, unverified claims, weak justifications, and hidden risks.

### Pipeline Stages
1. **Stage 1: Document Ingestion & Extraction**
   - Extract raw text from uploaded PDFs using PyMuPDF.
   - Build grounded Candidate Profiles and structured Evidence Stores.
2. **Stage 2: Independent Assessment**
   - Concurrently or sequentially run isolated evaluations for all 4 agents.
   - Restrict access so no agent sees another agent's initial verdict.
3. **Stage 3: Cross-Examination Debate**
   - Agents review other agents' arguments and challenge discrepancies or over-optimistic/pessimistic takes with evidence citations.
4. **Stage 4: Post-Debate Reassessment**
   - Each agent reviews the debate log and submits a revised stance, logging changes in recommendation and confidence.
5. **Stage 5: Final Non-Averaging Decision & Comparative Report**
   - Synthesizer reviews full profile, debate history, and final stances to output qualitative recommendations, unresolved disagreements, and comparative ranking for Candidate A vs Candidate B.

---

## 4. Setup & Getting Started

### Prerequisites
- Python 3.12+
- OpenAI-compatible API Key (e.g. OpenAI, OpenRouter, Local LLM)

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd multi-agent-interview-panel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your LLM API credentials
```

### Running the Server
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Visit `http://127.0.0.1:8000/api/health` to verify the backend is running.
