"""
backend/app/services/rubric_extractor.py
Phase 2 — Answer Key → Rubric Extractor (Gemini ONLY)
========================================================
Uses Gemini API (FREE: 1,500 req/day) for rubric creation.
No OpenAI fallback.

VERSION 2.0 — IMPROVEMENTS:
  ✅ Subject-specific prompt templates
  ✅ Few-shot examples in prompt
  ✅ Chain-of-thought prompting
  ✅ Better keyword extraction
  ✅ diagram_required and formula_required flags support
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from pydantic import ValidationError

# ── Load .env FIRST — with explicit path ──────────────────────────────────
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# Debug - check if key is loaded
key = os.environ.get("GEMINI_API_KEY")
if key:
    print(f"✅ API Key loaded: {key[:15]}...")
else:
    print("❌ API Key NOT loaded! Check .env file location.")
    print(f"   Looking for .env at: {env_path}")

# ── Add project root so Phase 0 schema is importable ─────────────────────
sys.path.append(str(Path(__file__).resolve().parents[3]))

# ── Import Phase 0 schema ──────────────────────────────────────────────────
from phase0.rubric_schema import AnswerKey, RubricPoint, GENERIC_KEYWORDS

# ── Import LLM client ──────────────────────────────────────────────────────
from backend.app.services.llm_client import get_llm_client, get_model_config, get_default_model

# ── Structured logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2.rubric_extractor")

# ── Config ─────────────────────────────────────────────────────────────────
PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0
MAX_RETRIES = 3

# ── Paths ─────────────────────────────────────────────────────────────────
QUESTIONS_FOLDER = Path(__file__).resolve().parents[3] / "phase0" / "questions"
CACHE_FOLDER = Path(__file__).resolve().parents[3] / "cache" / "rubrics"


# ── Subject-specific prompt templates ─────────────────────────────────────

SUBJECT_PROMPTS = {
    "science": """
    Focus on:
    - Processes and mechanisms
    - Chemical equations and formulas (mark formula_required=true)
    - Cause-effect relationships
    - Diagrams (mark diagram_required=true where needed)
    - Experimental evidence
    """,
    
    "physics": """
    Focus on:
    - Laws and principles
    - Mathematical formulas (F=ma, etc.) - mark formula_required=true
    - Numerical values and units
    - Real-world applications
    - Diagrams showing forces/motion - mark diagram_required=true
    """,
    
    "chemistry": """
    Focus on:
    - Chemical equations (balanced) - mark formula_required=true
    - Elements and compounds
    - Reaction mechanisms
    - Properties and characteristics
    - Molecular structures - mark diagram_required=true
    """,
    
    "biology": """
    Focus on:
    - Biological processes
    - Structures and functions
    - Diagrams (cell, organ, system) - mark diagram_required=true
    - Terminology and definitions
    - Cause-effect in living systems
    """,
    
    "history": """
    Focus on:
    - Dates and timelines
    - Causes and consequences
    - Key figures and events
    - Historical significance
    - Chronological order
    """,
    
    "geography": """
    Focus on:
    - Locations and maps - mark diagram_required=true
    - Physical processes
    - Human-environment interaction
    - Climate and weather patterns
    - Statistical data
    """,
    
    "mathematics": """
    Focus on:
    - Formulas and equations - mark formula_required=true
    - Step-by-step solutions
    - Final answer with units
    - Logical reasoning
    - Proofs and derivations
    """,
    
    "english": """
    Focus on:
    - Grammar and syntax
    - Structure and organization
    - Examples and evidence
    - Clarity and style
    - Critical analysis
    """,
    
    "civics": """
    Focus on:
    - Rights and duties
    - Constitutional provisions
    - Government structures
    - Legal frameworks
    - Democratic processes
    """
}


# ── Few-shot example rubric ──────────────────────────────────────────────

EXAMPLE_RUBRIC = {
    "question_id": "sci_example",
    "subject": "Science",
    "grade": "10th",
    "question_text": "Explain the process of photosynthesis.",
    "max_marks": 10,
    "rubric_points": [
        {
            "point_id": "p1",
            "concept": "Definition",
            "description": "Defines photosynthesis as process by which plants make food using sunlight",
            "marks": 2,
            "is_required": True,
            "diagram_required": False,
            "formula_required": False,
            "keywords": ["sunlight", "food", "green plants", "photosynthesis"],
            "partial_keywords": ["sunlight", "plants"],
            "examples": ["Photosynthesis is how plants make food using sunlight"],
            "common_mistakes": ["Says photosynthesis does not need sunlight"]
        },
        {
            "point_id": "p2",
            "concept": "Raw Materials",
            "description": "States carbon dioxide and water as raw materials",
            "marks": 2,
            "is_required": True,
            "diagram_required": False,
            "formula_required": False,
            "keywords": ["carbon dioxide", "water", "CO2", "H2O"],
            "partial_keywords": ["carbon dioxide", "water"],
            "examples": ["Plants use CO2 and H2O as raw materials"],
            "common_mistakes": ["Says oxygen is a raw material"]
        },
        {
            "point_id": "p3",
            "concept": "Products",
            "description": "States glucose and oxygen as products",
            "marks": 2,
            "is_required": True,
            "diagram_required": False,
            "formula_required": False,
            "keywords": ["glucose", "oxygen", "C6H12O6", "O2"],
            "partial_keywords": ["glucose", "oxygen"],
            "examples": ["Glucose and oxygen are the products"],
            "common_mistakes": ["Says carbon dioxide is a product"]
        },
        {
            "point_id": "p4",
            "concept": "Chemical Equation",
            "description": "Writes or describes the overall balanced equation",
            "marks": 2,
            "is_required": False,
            "diagram_required": False,
            "formula_required": True,
            "keywords": ["6CO2", "6H2O", "C6H12O6", "6O2"],
            "partial_keywords": ["CO2", "H2O"],
            "examples": ["6CO2 + 6H2O + Light → C6H12O6 + 6O2"],
            "common_mistakes": ["Unbalanced equation"]
        },
        {
            "point_id": "p5",
            "concept": "Site of Reaction",
            "description": "Mentions chloroplast or chlorophyll as the site",
            "marks": 2,
            "is_required": True,
            "diagram_required": True,
            "formula_required": False,
            "keywords": ["chloroplast", "chlorophyll"],
            "partial_keywords": ["chlorophyll"],
            "examples": ["Photosynthesis occurs in chloroplasts"],
            "common_mistakes": ["Says it occurs in mitochondria"]
        }
    ]
}


# ── Return type ───────────────────────────────────────────────────────────

@dataclass
class ExtractResult:
    status: str
    rubric: Optional[AnswerKey] = None
    saved_path: Optional[str] = None
    reason: Optional[str] = None
    raw_llm_response: Optional[str] = None
    model_used: Optional[str] = None
    from_cache: bool = False


# ── Input validation ──────────────────────────────────────────────────────

def _validate_inputs(
    question_id: str,
    question_text: str,
    answer_key_text: str,
    max_marks: float,
) -> Optional[str]:
    """Returns error message string if invalid, else None."""
    if not question_id or not question_id.strip():
        return "question_id cannot be empty"

    if not question_text or not question_text.strip():
        return "question_text cannot be empty"

    if not answer_key_text or len(answer_key_text.strip()) < 10:
        return "answer_key_text must be at least 10 characters"

    try:
        marks = float(max_marks)
        if marks <= 0:
            return f"max_marks must be greater than 0, got {marks}"
    except (TypeError, ValueError):
        return f"max_marks must be a number, got '{max_marks}'"

    return None


# ── Caching ────────────────────────────────────────────────────────────────

def _cache_key(question_id: str, answer_key_text: str) -> str:
    text_hash = hashlib.md5(answer_key_text.encode()).hexdigest()
    return f"{question_id}_{text_hash}"


def _load_cache(key: str) -> Optional[dict]:
    path = CACHE_FOLDER / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cache(key: str, data: dict) -> None:
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    path = CACHE_FOLDER / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ── Advanced Prompt Builder ─────────────────────────────────────────────

def _build_advanced_prompt(
    question_id: str,
    subject: str,
    grade: str,
    question_text: str,
    answer_key_text: str,
    max_marks: float,
    difficulty: str,
    topic: str,
    created_by: str,
) -> str:
    """
    ✅ Advanced prompt with:
    - Subject-specific guidance
    - Few-shot example
    - Chain-of-thought reasoning
    """
    
    # Get subject-specific guidance
    subject_lower = subject.lower()
    subject_guidance = SUBJECT_PROMPTS.get(subject_lower, SUBJECT_PROMPTS["science"])
    
    # Format example rubric
    example_json = json.dumps(EXAMPLE_RUBRIC, indent=2)
    
    return f"""
You are an expert exam rubric designer for school and college exams.

TASK: Convert this teacher's answer key into a structured rubric.

SUBJECT: {subject}
{subject_guidance}

--- IMPORTANT GUIDELINES ---
1. Break the answer into 4-6 discrete marking points
2. Each point must be testable - clear right/wrong
3. Marks across ALL points MUST sum to {max_marks}
4. Mark 2-3 points as is_required=true (core concepts)
5. Keywords must be SPECIFIC technical terms (NOT generic words like "process", "occurs", "happens")
6. For Science questions: mark formula_required=true for equations
7. For Diagram-based questions: mark diagram_required=true where diagrams are needed
8. Include partial_keywords for partial credit
9. Include common_mistakes for each point

--- STEP-BY-STEP THINKING PROCESS (Chain of Thought) ---
Step 1: Identify the key concepts in the answer key
Step 2: Group related concepts together
Step 3: Assign marks based on importance (core concepts = more marks)
Step 4: Extract specific technical keywords from the answer
Step 5: Identify common mistakes students might make
Step 6: Determine which points need diagrams or formulas

--- EXAMPLE OF A WELL-STRUCTURED RUBRIC ---
{example_json}

--- NOW CREATE THE RUBRIC FOR THIS QUESTION ---

QUESTION METADATA:
  question_id   : {question_id}
  subject       : {subject}
  grade         : {grade}
  question_text : {question_text}
  max_marks     : {max_marks}
  difficulty    : {difficulty}
  topic         : {topic}
  created_by    : {created_by}

RAW ANSWER KEY TEXT:
{answer_key_text}

--- RESPONSE FORMAT ---
IMPORTANT: Respond with VALID JSON ONLY.
No markdown, no extra explanation - just the JSON object.
Make sure all keys match the example format exactly.
"""


# ── Response helpers ──────────────────────────────────────────────────────

def _clean_llm_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _fix_marks_sum(data: dict, max_marks: float) -> dict:
    points = data.get("rubric_points", [])
    if not points:
        return data
    current_sum = sum(p.get("marks", 0) for p in points)
    gap = round(max_marks - current_sum, 2)
    if gap == 0:
        return data
    if abs(gap) > 2:
        raise ValueError(
            f"Rubric points sum to {current_sum}, expected {max_marks}. "
            f"Gap of {gap} is too large to auto-fix."
        )
    points[-1]["marks"] = round(points[-1]["marks"] + gap, 2)
    data["rubric_points"] = points
    return data


def _validate_keyword_quality(data: dict) -> List[str]:
    """Check for generic keywords and return warnings"""
    warnings = []
    generic = GENERIC_KEYWORDS
    
    for point in data.get("rubric_points", []):
        keywords = point.get("keywords", [])
        if keywords:
            generic_keywords = [kw for kw in keywords if kw.lower() in generic]
            if generic_keywords:
                warnings.append(
                    f"Point {point.get('point_id', 'unknown')} has generic keywords: {generic_keywords}"
                )
    return warnings


# ── Core LLM call ─────────────────────────────────────────────────────────

def _call_llm(user_prompt: str, model: str) -> str:
    """Single LLM call using Gemini (Phase 2 - FREE)."""
    try:
        client = get_llm_client("gemini")
        model_config = get_model_config("gemini", model)
        
        response = client.chat.completions.create(
            model=model_config["model"],
            temperature=TEMPERATURE,
            max_tokens=model_config["max_tokens"],
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert exam rubric designer. Return valid JSON only."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


# ── Save helper ───────────────────────────────────────────────────────────

def _save_rubric(answer_key: AnswerKey, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{answer_key.question_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(answer_key.model_dump(), f, indent=2, ensure_ascii=False)
    return str(file_path)


# ── Public API ────────────────────────────────────────────────────────────

def extract_rubric(
    question_id: str,
    subject: str,
    grade: str,
    question_text: str,
    answer_key_text: str,
    max_marks: float,
    difficulty: str = "medium",
    topic: str = "general",
    created_by: str = "teacher",
    model: str = DEFAULT_MODEL,
    save: bool = True,
) -> ExtractResult:
    """
    Main Phase 2 function — uses Gemini (FREE).
    
    Features:
    - Subject-specific prompts
    - Few-shot examples
    - Chain-of-thought reasoning
    - Keyword quality validation
    """
    logger.info(f"Starting rubric extraction — question_id={question_id}  model={model}")

    # ── 1. Input validation ───────────────────────────────────────────────
    val_error = _validate_inputs(
        question_id, question_text, answer_key_text, max_marks
    )
    if val_error:
        logger.error(f"Input validation failed: {val_error}")
        return ExtractResult(status="error", reason=val_error)

    # ── 2. Cache check ────────────────────────────────────────────────────
    ck = _cache_key(question_id, answer_key_text)
    cached = _load_cache(ck)
    if cached:
        logger.info(f"Cache hit — skipping LLM call for {question_id}")
        try:
            answer_key = AnswerKey(**cached)
            saved_path = None
            if save:
                saved_path = _save_rubric(answer_key, QUESTIONS_FOLDER)
            return ExtractResult(
                status="success",
                rubric=answer_key,
                saved_path=saved_path,
                model_used=model,
                from_cache=True,
            )
        except Exception:
            logger.warning("Cache data invalid — proceeding with LLM call")

    # ── 3. Build advanced prompt ──────────────────────────────────────────
    user_prompt = _build_advanced_prompt(
        question_id, subject, grade, question_text,
        answer_key_text, max_marks, difficulty, topic, created_by,
    )
    
    last_error = ""
    raw_response = ""

    # ── 4. LLM call with retry ────────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt}/{MAX_RETRIES} — calling Gemini...")
            raw_response = _call_llm(user_prompt, model)

            cleaned = _clean_llm_response(raw_response)
            data = json.loads(cleaned)
            
            # Validate keyword quality
            warnings = _validate_keyword_quality(data)
            if warnings:
                for w in warnings:
                    logger.warning(f"Keyword quality issue: {w}")
            
            data = _fix_marks_sum(data, max_marks)
            answer_key = AnswerKey(**data)

            # Save to disk
            saved_path = None
            if save:
                saved_path = _save_rubric(answer_key, QUESTIONS_FOLDER)
                logger.info(f"Rubric saved → {saved_path}")

            # Save to cache
            _save_cache(ck, data)
            logger.info(f"Cached result for future calls")

            return ExtractResult(
                status="success",
                rubric=answer_key,
                saved_path=saved_path,
                raw_llm_response=raw_response,
                model_used=model,
                from_cache=False,
            )

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error on attempt {attempt}: {e}"
            logger.warning(last_error)

        except ValidationError as e:
            last_error = f"Pydantic schema error on attempt {attempt}: {e}"
            logger.warning(last_error)

        except ValueError as e:
            last_error = f"Marks sum error on attempt {attempt}: {e}"
            logger.warning(last_error)

        except Exception as e:
            last_error = f"Non-retryable error on attempt {attempt}: {e}"
            logger.error(last_error)
            break

    logger.error(f"All retries failed for {question_id}: {last_error}")
    return ExtractResult(
        status="error",
        reason=last_error,
        raw_llm_response=raw_response,
        model_used=model,
    )


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Phase 2: Rubric Extractor Test")
    print("=" * 60)
    
    # Test input
    result = extract_rubric(
        question_id="test_001",
        subject="Science",
        grade="10th",
        question_text="Explain photosynthesis.",
        answer_key_text="Photosynthesis is the process by which plants use sunlight to make food.",
        max_marks=10,
        save=False
    )
    
    print(f"\nStatus: {result.status}")
    if result.status == "success":
        print(f"Points: {len(result.rubric.rubric_points)}")
        for p in result.rubric.rubric_points:
            print(f"  [{p.point_id}] {p.concept}: {p.marks} marks")
    print("=" * 60)