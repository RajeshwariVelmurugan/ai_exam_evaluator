"""
backend/app/services/llm_evaluator.py
Phase 4 — LLM Answer Evaluation (Groq ONLY)
============================================
Uses Groq API (FREE: 14,400 req/day) for student answer evaluation.
No OpenAI fallback.

Flow:
  Student answer + rubric → Groq → Evaluation results with reasoning

Usage:
  from app.services.llm_evaluator import evaluate_answer

  result = evaluate_answer(
      student_answer="Photosynthesis is...",
      rubric_points=rubric.rubric_points,
      model="mixtral-8x7b-32768"
  )
"""

import json
import logging
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from backend.app.services.llm_client import get_llm_client, get_model_config, get_default_model
load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4.llm_evaluator")

# ── Config ─────────────────────────────────────────────────────────────────
PROVIDER = "groq"
DEFAULT_MODEL = get_default_model(PROVIDER)
TEMPERATURE = 0.3
MAX_RETRIES = 3


# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert exam evaluator. Grade the student's answer against the rubric points.

For each rubric point, determine:
1. satisfied: True/False (did the student cover this point?)
2. confidence: 0.0-1.0 (how sure are you?)
3. reasoning: Brief explanation of your decision
4. matched_text: Which part of the student's answer matches this point

Also check for:
- hallucination_detected: True if the student wrote something completely off-topic or nonsensical
- hallucination_note: Brief explanation if hallucination detected

Return ONLY valid JSON in this format:
{
  "evaluation": [
    {
      "point_id": "p1",
      "satisfied": true,
      "confidence": 0.95,
      "reasoning": "Student correctly defined photosynthesis",
      "matched_text": "Photosynthesis is the process..."
    }
  ],
  "hallucination_detected": false,
  "hallucination_note": null,
  "total_score": 8.5,
  "max_possible": 10
}
""".strip()


def _build_evaluation_prompt(student_answer: str, rubric_points: List) -> str:
    """Build the user prompt for evaluation."""
    
    rubric_json = json.dumps([
        {
            "point_id": p.point_id,
            "concept": p.concept,
            "description": p.description,
            "marks": p.marks,
            "is_required": p.is_required,
            "keywords": p.keywords
        }
        for p in rubric_points
    ], indent=2)
    
    return f"""
STUDENT ANSWER:
{student_answer}

RUBRIC POINTS:
{rubric_json}

Evaluate each rubric point against the student answer.
Return valid JSON only — no markdown, no explanation.
"""


# ── Helper functions ──────────────────────────────────────────────────────

def _parse_evaluation_response(raw_response: str) -> Dict:
    """Parse and clean the LLM response."""
    # Clean markdown
    raw = raw_response.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    
    # Try to find JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    
    raise ValueError("No valid JSON found in response")


def _call_groq(user_prompt: str, model: str) -> str:
    """Call Groq API for evaluation."""
    client = get_llm_client("groq")
    model_config = get_model_config("groq", model)
    
    response = client.chat.completions.create(
        model=model_config["model"],
        temperature=TEMPERATURE,
        max_tokens=model_config["max_tokens"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


# ── Public API ─────────────────────────────────────────────────────────────

def evaluate_answer(
    student_answer: str,
    rubric_points: List,
    model: str = DEFAULT_MODEL,
) -> Dict:
    """
    Evaluate a student answer using Groq (Phase 4 - FREE).
    
    Args:
        student_answer: The student's answer text
        rubric_points: List of RubricPoint objects
        model: Groq model name
    
    Returns:
        Dict with evaluation results including:
        - evaluation: List of point evaluations
        - hallucination_detected: bool
        - total_score: float
        - max_possible: float
    """
    
    logger.info(f"Starting evaluation — model={model}")
    
    # Input validation
    if not student_answer or len(student_answer.strip()) < 5:
        return {
            "error": "Student answer too short or empty",
            "evaluation": [],
            "hallucination_detected": False,
            "total_score": 0,
            "max_possible": sum(p.marks for p in rubric_points)
        }
    
    if not rubric_points or len(rubric_points) < 1:
        return {
            "error": "Rubric points cannot be empty",
            "evaluation": [],
            "hallucination_detected": False,
            "total_score": 0,
            "max_possible": 0
        }
    
    user_prompt = _build_evaluation_prompt(student_answer, rubric_points)
    last_error = ""
    raw_response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt}/{MAX_RETRIES} — calling Groq...")
            raw_response = _call_groq(user_prompt, model)
            result = _parse_evaluation_response(raw_response)
            
            # Validate response has required fields
            if "evaluation" not in result:
                raise ValueError("Response missing 'evaluation' field")
            
            # Calculate total score if not provided
            if "total_score" not in result:
                total = sum(
                    p.get("marks", 0) * (1 if p.get("satisfied", False) else 0)
                    for p in result["evaluation"]
                )
                result["total_score"] = round(total, 1)
                result["max_possible"] = sum(p.marks for p in rubric_points)
            
            logger.info(f"✅ Evaluation complete — Score: {result.get('total_score', 0)}")
            return result
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error on attempt {attempt}: {e}"
            logger.warning(last_error)
            
        except ValueError as e:
            last_error = f"Validation error on attempt {attempt}: {e}"
            logger.warning(last_error)
            
        except Exception as e:
            last_error = f"Non-retryable error on attempt {attempt}: {e}"
            logger.error(last_error)
            break
    
    logger.error(f"All retries failed: {last_error}")
    return {
        "error": last_error,
        "evaluation": [],
        "hallucination_detected": False,
        "total_score": 0,
        "max_possible": sum(p.marks for p in rubric_points),
        "raw_response": raw_response
    }


def evaluate_batch(
    student_answers: List[str],
    rubric_points: List,
    model: str = DEFAULT_MODEL,
) -> List[Dict]:
    """
    Evaluate multiple student answers in batch.
    
    Args:
        student_answers: List of student answer texts
        rubric_points: List of RubricPoint objects
        model: Groq model name
    
    Returns:
        List of evaluation results
    """
    results = []
    for i, answer in enumerate(student_answers):
        logger.info(f"Evaluating student {i+1}/{len(student_answers)}")
        result = evaluate_answer(answer, rubric_points, model)
        result["student_index"] = i
        results.append(result)
    
    return results


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Phase 4: LLM Evaluator Test")
    print("=" * 60)
    
    # Mock rubric points
    class MockRubricPoint:
        def __init__(self, point_id, concept, description, marks, is_required, keywords):
            self.point_id = point_id
            self.concept = concept
            self.description = description
            self.marks = marks
            self.is_required = is_required
            self.keywords = keywords
    
    mock_rubric = [
        MockRubricPoint("p1", "Definition", "Defines photosynthesis", 2, True, ["sunlight"]),
        MockRubricPoint("p2", "Equation", "States chemical equation", 3, True, ["6CO2", "H2O"]),
    ]
    
    mock_answer = "Photosynthesis is the process where plants use sunlight to make food. The equation is 6CO2 + 6H2O → C6H12O6 + 6O2."
    
    print("\n📝 Test Answer:")
    print(mock_answer)
    print("\n📊 Rubric:")
    for p in mock_rubric:
        print(f"  {p.point_id}: {p.concept} ({p.marks} marks)")
    
    print("\n🤖 Running evaluation...")
    result = evaluate_answer(mock_answer, mock_rubric)
    
    print("\n📋 Results:")
    print(json.dumps(result, indent=2))
    print("\n" + "=" * 60)