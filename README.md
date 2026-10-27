AI-Powered Answer Sheet Assessor

Short Answer Grading Engine: an NLP pipeline that reads handwritten or printed student answers, checks them against a teacher-defined rubric, and returns an explainable score.

Show Image Show Image Show Image Show Image Show Image

Why this project exists

Embedding-similarity graders (for example SBERT cosine similarity) cannot see negation:

Rubric point	Student answer	Embedding similarity	Reality
Photosynthesis occurs in the presence of sunlight	Photosynthesis does not happen in the presence of sunlight	very high	wrong answer

SAGE puts a Natural Language Inference (NLI) model in front of the LLM. NLI classifies each rubric point against the student answer as ENTAILMENT, NEUTRAL or CONTRADICTION, so direct contradictions are caught cheaply and explained. Only the remaining points go to the LLM for deeper evaluation.

Pipeline
too blurry
handwritten
printed
high-confidencecontradiction
everything else
Student answer photo
OpenCV preprocessingblur gate, shadow removal,deskew
Reject: ask for retake
Handwritten or printed?
TrOCR - ICR
EasyOCR - OCR
Answer text
NLI gatekeeperDistilRoBERTa
Score 0 for that pointreasoning kept for UI
LLM rubric evaluationspan-level output
Scoring engineGrammar / Content / Logicweights
PostgreSQL
Teacher dashboardhighlights, tooltips, reviewqueue
Phases
Phase	What it does	Main files
0	Rubric schema and sample question bank	phase0/rubric_schema.py, phase0/questions/*.json
1	Image preprocessing and OCR/ICR	backend/app/utils/image_processor.py, backend/app/services/ocr_service.py
2	Answer key to structured rubric (LLM-assisted)	backend/app/services/rubric_extractor.py
3	NLI contradiction gatekeeper	backend/app/services/nli_service.py
4	LLM rubric evaluation with supporting spans	backend/app/services/llm_service.py
5	Weighted scoring, hallucination penalty, logic floor cap	backend/app/services/scoring_service.py
6	FastAPI backend, PostgreSQL models, concurrency limiter	backend/app/main.py, models.py, schemas.py, database.py, utils/rate_limiter.py
7	Next.js teacher dashboard	frontend/src/
8	Calibration against teacher-graded answers	phase8/evaluate_accuracy.py

Supporting NLP modules: text_preprocessor.py (spaCy), grammar_checker.py (LanguageTool, textstat), keyword_extractor.py (KeyBERT, TF-IDF, SBERT baseline).

Key features
Image quality guardrail. Laplacian-variance blur check rejects unusable photos before OCR. Shadow removal, deskew and adaptive thresholding clean up the rest.
Free, local OCR/ICR. TrOCR (microsoft/trocr-base-handwritten) for handwriting, EasyOCR for printed text. No OCR API key is needed.
NLI gatekeeper. Contradictions at or above 0.85 confidence skip the LLM and still carry a human-readable reason.
Rubric-based grading. Each answer is scored per rubric point (concept, marks, keywords, partial credit rules), the way a teacher marks.
Configurable weights. Grammar, content and logic weights (presets or custom sliders) with a hard floor: if the logic score drops below 40%, the total is capped at 50% of max marks regardless of the weights.
Explainable UI. Green/yellow/red sentence highlighting, reasoning tooltips, per-point breakdown, and a review queue for flagged answers with teacher override.
Async backend. Evaluation runs as a FastAPI background task, and an asyncio.Semaphore caps concurrent LLM calls.
Rubric validation. Pydantic v2 checks that rubric marks sum to max_marks and that grammar/content/logic weights sum to 1.0.
Tech stack
Layer	Technology
Image processing	OpenCV, NumPy, Pillow
OCR / ICR	TrOCR (HuggingFace Transformers), EasyOCR
NLP	spaCy, LanguageTool, textstat, KeyBERT, sentence-transformers
NLI	cross-encoder/nli-distilroberta-base
LLM	OpenAI-compatible chat API (GPT-4o by default)
Backend	FastAPI, SQLAlchemy (async), asyncpg, Alembic, Pydantic v2
Database	PostgreSQL (JSONB for rubric and breakdown data)
Frontend	Next.js, React, Tailwind CSS, react-dropzone
Testing	pytest, pandas, NumPy
Project structure
sage/
├── phase0/
│   ├── rubric_schema.py          # Pydantic schema and batch validator
│   ├── questions/                # 10 sample rubric JSON files
│   ├── test_answers/             # sample student answers (good / average / poor)
│   └── manually_graded/          # teacher scores for calibration (CSV)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app and evaluation pipeline
│   │   ├── models.py             # SQLAlchemy tables
│   │   ├── schemas.py            # API request/response models
│   │   ├── database.py           # async engine and session
│   │   ├── services/             # ocr, nli, llm, scoring, rubric extraction, NLP helpers
│   │   └── utils/                # image_processor.py, rate_limiter.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── api/client.js
│       ├── components/           # UploadScreen, EvalResult, HighlightViewer, ReviewQueue, WeightageSettings
│       └── pages/Dashboard.jsx
├── phase8/
│   └── evaluate_accuracy.py      # writes calibration_report.md
├── notebooks/                    # manual test scripts per phase
├── tests/                        # pytest unit tests
└── check_phase1_packages.py      # verifies Phase 1 dependencies
Getting started
Prerequisites
Python 3.10+
Node.js 18+
PostgreSQL 14+
About 2 GB free disk for model downloads (TrOCR, EasyOCR, NLI model) on first run
1. Clone
bash
git clone https://github.com/<your-username>/sage.git
cd sage
2. Backend
bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env              # then fill in the values

Check the Phase 1 dependencies:

bash
python ../check_phase1_packages.py

Create the database, then start the server:

bash
createdb sage_db
uvicorn app.main:app --reload --port 8000

API docs: http://localhost:8000/docs

Tables are created automatically on startup. Alembic is included for migrations once you move past local development.

3. Frontend
bash
cd frontend
npm install
npm run dev

Open http://localhost:3000

4. Validate the sample rubrics
bash
cd phase0
python rubric_schema.py
5. Run tests
bash
pytest tests/ -v

LLM and NLI calls are mocked in the unit tests, so no API key or model download is needed.

Environment variables

backend/.env

Variable	Purpose	Example
DATABASE_URL	Async PostgreSQL URL	postgresql+asyncpg://user:password@localhost:5432/sage_db
OPENAI_API_KEY	LLM key for Phase 2 and Phase 4 (any OpenAI-compatible provider)	sk-...
LLM_MAX_CONCURRENT	Max simultaneous LLM calls	5
FRONTEND_URL	Allowed CORS origin	http://localhost:3000

frontend/.env.local

Variable	Purpose	Example
NEXT_PUBLIC_API_URL	Backend base URL	http://localhost:8000

Never commit .env.

API overview
Method	Endpoint	Description
POST	/api/questions/	Create a question with its rubric
GET	/api/questions/{question_id}	Fetch a question
POST	/api/questions/extract-rubric	Generate a rubric from raw answer-key text
POST	/api/answers/upload	Upload an answer sheet image; returns a job_id
GET	/api/answers/{job_id}/status	Quick job status
GET	/api/evaluations/{job_id}	Poll for the full evaluation result
POST	/api/evaluations/{eval_id}/review	Teacher score override
GET	/api/evaluations/flagged	Evaluations waiting for review

Typical flow: upload an image, receive a job_id, poll /api/evaluations/{job_id} until status is done, rejected or error.

Rubric format

Each question is a JSON file validated by phase0/rubric_schema.py:

json
{
  "question_id": "sci_001",
  "subject": "Science",
  "grade": "10th",
  "question_text": "Explain the process of photosynthesis.",
  "max_marks": 10,
  "rubric_points": [
    {
      "point_id": "p1",
      "concept": "Definition",
      "description": "Defines photosynthesis as a process by which plants make food using sunlight",
      "marks": 2,
      "is_required": true,
      "partial_credit_allowed": true,
      "keywords": ["sunlight", "green plants"],
      "common_mistakes": ["Says photosynthesis does not need sunlight"]
    }
  ],
  "marking_scheme": {
    "content_weight": 0.7,
    "logic_weight": 0.2,
    "grammar_weight": 0.1
  }
}

The sample questions are only fixtures for testing and calibration. Teachers can add any subject by supplying a question and answer key.

Evaluation results

Run python phase8/evaluate_accuracy.py to compare system scores with teacher-graded scores. It writes phase8/calibration_report.md.

Fill this table from your own run:

Metric	Value
Answers evaluated	N
MAE (marks)	x.xx
RMSE (marks)	x.xx
Within ±1 mark of teacher	xx%
Pearson r	0.xx
Known limitations
Region analysis (region_analyzer.py), diagram detection with CLIP (diagram_detector.py) and formula extraction (formula_extractor.py) are implemented as modules but not yet wired into the main evaluation pipeline.
OCR accuracy on poor handwriting is the main bottleneck. Low-confidence OCR results should go to teacher review.
LLM evaluation is sensitive to prompt and model choice. Calibrate on your own graded data before relying on scores.
Concurrency control uses an in-process asyncio.Semaphore, which is correct for a single worker but not for multi-worker deployments.
This is a portfolio project. Authentication, rate limiting, audit logging and data-privacy compliance are not implemented.
