"""
Phase 0 deliverable — Complete Rubric Schema Definition (Master Copy)
=====================================================================
This file defines the strict structural standards for your grading system.

How it works across your project phases:
  - Phase 0: Validate your 10 manually created JSON questions.
  - Phase 2: Serves as the Pydantic parser for automated rubric generation.
  - Phase 3 & 4: NLI Gatekeeper & LLM Engine consume this structure to score answers.
  - Phase 7: UI parses concepts and points to render the teacher dashboard.

Run directly to test validation: `python phase0/rubric_schema.py`

VERSION 2.0 — IMPROVEMENTS:
  ✅ Added diagram_required and formula_required fields
  ✅ Added duplicate point_id check
  ✅ Added keyword quality validator (warns about generic keywords)
  ✅ Added comprehensive rubric quality validation function
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from datetime import datetime
import json
from pathlib import Path


class SafetyConfig(BaseModel):
    """Safety limits, multipliers, and NLI threshold boundaries"""
    
    hard_floor_score: float = Field(
        2.0, 
        description="Maximum absolute marks allowed if critical logic/contradiction fails", 
        ge=0
    )
    contradiction_penalty: float = Field(
        0.5, 
        description="Penalty multiplier for each contradiction detected", 
        ge=0, 
        le=1
    )
    hallucination_penalty: float = Field(
        2.0, 
        description="Flat penalty score deduction for hallucination detection", 
        ge=0
    )
    max_hallucination_penalty: float = Field(
        5.0, 
        description="Maximum ceiling penalty cap for multiple combined hallucinations", 
        ge=0
    )
    absolute_min_score: float = Field(
        0.0, 
        description="Absolute floor score (cannot go below 0 even with penalties)", 
        ge=0
    )
    contradiction_threshold: float = Field(
        0.80, 
        description="NLI confidence threshold limit to register a hard contradiction", 
        ge=0, 
        le=1
    )
    entailment_threshold: float = Field(
        0.70, 
        description="NLI confidence threshold limit to confirm text entailment", 
        ge=0, 
        le=1
    )
    
    @model_validator(mode="after")
    def validate_penalties(self) -> "SafetyConfig":
        """Ensure specific logic caps stay within reasonable math boundaries"""
        if self.hallucination_penalty > self.max_hallucination_penalty:
            raise ValueError(
                f"hallucination_penalty ({self.hallucination_penalty}) cannot exceed "
                f"max_hallucination_penalty ({self.max_hallucination_penalty})"
            )
        return self


class RubricPoint(BaseModel):
    """Single standalone concept point to assess inside an answer layout"""
    
    point_id: str = Field(..., description="Unique ID for this specific rubric line (e.g., p1, p2)")
    concept: str = Field(..., description="High level concept group (e.g., Definition, Equation)")
    description: str = Field(..., description="Granular requirement description in plain English")
    marks: float = Field(..., description="Marks assigned strictly to this item line", gt=0)
    
    # ── NEW: Diagram and Formula flags ──────────────────────────────────
    diagram_required: bool = Field(
        False, 
        description="Does this point require a diagram/visual representation?"
    )
    formula_required: bool = Field(
        False, 
        description="Does this point require a mathematical formula/equation?"
    )
    
    is_required: bool = Field(True, description="Is this a core mandatory requirement point for passing?")
    partial_credit_allowed: bool = Field(True, description="Can teachers or engines allocate fractional split scores?")
    
    keywords: List[str] = Field(
        default_factory=list, 
        description="Target token strings or lemmatized terms for precise match loops"
    )
    partial_keywords: List[str] = Field(
        default_factory=list, 
        description="Secondary token structures acceptable for lesser point weights"
    )
    
    examples: List[str] = Field(default_factory=list, description="Ideal target example string fragments")
    common_mistakes: List[str] = Field(default_factory=list, description="Common logical negations or errors seen here")

    @field_validator("marks")
    @classmethod
    def marks_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Marks allocation must be positively tracking above 0. Got: {v}")
        return v
    
    @field_validator("point_id")
    @classmethod
    def point_id_format(cls, v: str) -> str:
        """Validate point_id format (e.g., p1, p2, p3)"""
        if not v or not v.strip():
            raise ValueError("point_id cannot be empty")
        if not v.startswith("p"):
            raise ValueError(f"point_id should start with 'p', got {v}")
        return v


class MarkingScheme(BaseModel):
    """Overall structural weight variables for finalizing overall aggregated scoring tiers"""
    
    content_weight: float = Field(0.7, description="Content accuracy factor scaling allocation", ge=0, le=1)
    logic_weight: float = Field(0.2, description="Logical reasoning factor scaling allocation", ge=0, le=1)
    grammar_weight: float = Field(0.1, description="Syntactical grammar precision factor scale tracking", ge=0, le=1)
    
    passing_percentage: float = Field(40.0, description="Minimum percentage scale cut to qualify pass", ge=0, le=100)
    distinction_percentage: float = Field(75.0, description="Threshold minimum score to register high honor", ge=0, le=100)
    
    @field_validator("content_weight", "logic_weight", "grammar_weight")
    @classmethod
    def validate_individual_weight(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Individual structural parameter layers must scale cleanly between 0 and 1. Got: {v}")
        return v
    
    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "MarkingScheme":
        """Enforces that total evaluation dimensions resolve precisely to a balanced whole integer"""
        total = self.content_weight + self.logic_weight + self.grammar_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Combined processing factors must converge strictly to 1.0. Resolving currently to: {total}"
            )
        return self
    
    @model_validator(mode="after")
    def validate_passing_logic(self) -> "MarkingScheme":
        if self.passing_percentage >= self.distinction_percentage:
            raise ValueError("Passing qualification floor cannot scale past standard distinction boundaries.")
        return self


# ── NEW: Generic Keyword List ─────────────────────────────────────────────

GENERIC_KEYWORDS = {
    "the", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "process", "occurs", "happens", "takes", "place", "thing", "stuff", "way",
    "use", "used", "using", "make", "makes", "made", "get", "gets", "got",
    "go", "goes", "went", "come", "comes", "came", "look", "looks", "looked"
}


class AnswerKey(BaseModel):
    """The master class instance representing a target structured question entry sheet"""
    
    # Core Fields
    question_id: str = Field(..., description="Clean indexing key index reference code (e.g., sci_001)")
    subject: str = Field(..., description="Subject classification category track")
    grade: str = Field(..., description="Academic year tier level target focus")
    question_text: str = Field(..., description="The exact literal string representing problem prompt statement")
    max_marks: float = Field(..., description="Ceiling total absolute scores cap allocation tracking context", gt=0)
    
    # Rubric Array Linked Models
    rubric_points: List[RubricPoint] = Field(..., description="The comprehensive granular matrix data points")
    marking_scheme: MarkingScheme = Field(default_factory=MarkingScheme, description="Dimensional evaluation parameters")
    safety_config: SafetyConfig = Field(default_factory=SafetyConfig, description="Fault boundaries configurations mapping")
    
    # Extra Text Meta Parameters
    answer_key_text: Optional[str] = Field(None, description="Full string sequence text serving as baseline master blueprint")
    difficulty: str = Field("medium", description="Question difficulty metric tracking rank context")
    topic: str = Field("general", description="Granular module specific reference path classification")
    
    # Structural Context Controls
    version: str = Field("1.0", description="Schema revision mapping sequence identifier tracking index")
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Autogenerated execution tracking date logs")
    created_by: str = Field("teacher", description="Role source tracking context identifier")
    
    min_word_count: int = Field(30, description="Minimum length threshold floor standard constraint")
    max_word_count: int = Field(200, description="Ceiling buffer expansion threshold boundary max limit context")
    allow_extra_points: bool = Field(True, description="Toggle override options if high depth creative inputs register validation flags")
    penalty_for_hallucination: float = Field(2.0, description="Deprecated standalone track baseline - redirected to SafetyConfig layer structures")

    @field_validator("question_id")
    @classmethod
    def question_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Target structural validation identification codes cannot process null fields.")
        return v
    
    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        """Validate grade format"""
        valid_grades = ["6th", "7th", "8th", "9th", "10th", "11th", "12th"]
        if v not in valid_grades:
            raise ValueError(f"Grade must be one of {valid_grades}, got {v}")
        return v
    
    @field_validator("question_text")
    @classmethod
    def question_text_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 5:
            raise ValueError("Question text must be at least 5 characters")
        return v
    
    @field_validator("rubric_points")
    @classmethod
    def rubric_not_empty(cls, v: List[RubricPoint]) -> List[RubricPoint]:
        if not v:
            raise ValueError("Rubric points cannot be empty - at least 1 point required")
        return v
    
    # ── NEW: Duplicate point_id check ─────────────────────────────────────
    @field_validator("rubric_points")
    @classmethod
    def check_duplicate_point_ids(cls, points: List[RubricPoint]) -> List[RubricPoint]:
        """Check for duplicate point_id values"""
        point_ids = [p.point_id for p in points]
        duplicates = [pid for pid in point_ids if point_ids.count(pid) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate point_id found: {set(duplicates)}. "
                f"Each point_id must be unique."
            )
        return points
    
    # ── NEW: Keyword quality check (warns, doesn't fail) ──────────────────
    @field_validator("rubric_points")
    @classmethod
    def validate_keyword_quality(cls, points: List[RubricPoint]) -> List[RubricPoint]:
        """Check if keywords are too generic (warn but don't fail)"""
        for point in points:
            if point.keywords:
                generic = [kw for kw in point.keywords if kw.lower() in GENERIC_KEYWORDS]
                if generic:
                    print(f"⚠️  Warning: Point {point.point_id} has generic keywords: {generic}")
                    print(f"   Consider using more specific terms like 'chloroplast', 'photosynthesis'")
        return points
    
    @field_validator("difficulty")
    @classmethod
    def difficulty_valid(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if v not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}")
        return v
    
    @model_validator(mode="after")
    def marks_must_sum_correctly(self) -> "AnswerKey":
        """Calculates internal list contents array balances directly against the parent allocation standard"""
        total = sum(p.marks for p in self.rubric_points)
        if abs(total - self.max_marks) > 0.001:
            raise ValueError(
                f"Structural Error: Rubric collection adds up to {total} score limits. "
                f"Parent question setup enforces precisely a maximum capacity ceiling limit of {self.max_marks}."
            )
        return self
    
    @model_validator(mode="after")
    def validate_required_marks(self) -> "AnswerKey":
        """Validate required points don't exceed max_marks"""
        required_marks = sum(p.marks for p in self.get_required_points())
        if required_marks > self.max_marks:
            raise ValueError(
                f"Required points sum to {required_marks}, but max_marks is {self.max_marks}"
            )
        return self
    
    @model_validator(mode="after")
    def validate_word_counts(self) -> "AnswerKey":
        """Validate min_word_count < max_word_count"""
        if self.min_word_count >= self.max_word_count:
            raise ValueError(
                f"min_word_count ({self.min_word_count}) must be less than "
                f"max_word_count ({self.max_word_count})"
            )
        return self
    
    # Core Data Retrieval Helpers
    def get_required_points(self) -> List[RubricPoint]:
        return [p for p in self.rubric_points if p.is_required]
    
    def get_optional_points(self) -> List[RubricPoint]:
        return [p for p in self.rubric_points if not p.is_required]
    
    def get_total_marks_for_required(self) -> float:
        return sum(p.marks for p in self.get_required_points())
    
    def get_hard_floor_score(self) -> float:
        """Get hard-floor safety cap for contradictions"""
        return self.safety_config.hard_floor_score
    
    def get_contradiction_threshold(self) -> float:
        """Get NLI contradiction threshold"""
        return self.safety_config.contradiction_threshold


# ── NEW: Rubric Quality Validator ────────────────────────────────────────

def validate_rubric_quality(answer_key: AnswerKey) -> dict:
    """
    Comprehensive rubric quality validation.
    
    Returns:
        dict with quality score and suggestions
    """
    quality_report = {
        "score": 100,
        "warnings": [],
        "suggestions": []
    }
    
    for point in answer_key.rubric_points:
        # Check keyword quality
        if point.keywords:
            generic = [kw for kw in point.keywords if kw.lower() in GENERIC_KEYWORDS]
            if generic:
                quality_report["warnings"].append(
                    f"Point {point.point_id} has generic keywords: {generic}"
                )
                quality_report["score"] -= 2
                quality_report["suggestions"].append(
                    f"Replace generic keywords in {point.point_id} with specific terms"
                )
        
        # Check description length
        if len(point.description.split()) < 5:
            quality_report["warnings"].append(
                f"Point {point.point_id} description is too short"
            )
            quality_report["score"] -= 1
        
        # Check examples
        if not point.examples:
            quality_report["warnings"].append(
                f"Point {point.point_id} has no examples"
            )
            quality_report["score"] -= 1
        
        # Check common mistakes
        if not point.common_mistakes:
            quality_report["warnings"].append(
                f"Point {point.point_id} has no common mistakes listed"
            )
            quality_report["score"] -= 0.5
    
    quality_report["score"] = max(0, quality_report["score"])
    return quality_report


# ============================================
# Folder Validation Utilities
# ============================================

def validate_rubric_file(file_path: str) -> bool:
    """Parses a specific target local file structure directly through model engine guards"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        answer_key = AnswerKey(**data)
        print(f"✅ Valid File Structure: [{answer_key.question_id}] in {file_path}")
        print(f"   Marks Balance: Verified ({answer_key.max_marks} max marks mapping exactly)")
        print(f"   Matrix Arrays: Mapped {len(answer_key.rubric_points)} parameters.")
        print(f"   Hard Floor: {answer_key.get_hard_floor_score()}")
        
        # Check quality
        quality = validate_rubric_quality(answer_key)
        if quality["score"] < 100:
            print(f"   ⚠️  Quality Score: {quality['score']}/100")
            for warning in quality["warnings"]:
                print(f"      - {warning}")
        
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Syntax File Failure in {file_path}: Unparseable text layout context. Details: {e}")
        return False
    except ValueError as e:
        print(f"❌ Schema Validation Rule Rejection in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Structural Execution Error parsing target sheet {file_path}: {e}")
        return False


def validate_all_rubrics(folder_path: str = "phase0/questions") -> dict:
    """Iterates through complete subdirectory arrays tracking file validation trends"""
    results = {'total': 0, 'valid': 0, 'invalid': 0, 'files': []}
    path = Path(folder_path)
    
    if not path.exists():
        print(f"❌ Missing Target Execution Path: Folder system not located at directory point: {folder_path}")
        return results
        
    json_files = list(path.glob("*.json"))
    if not json_files:
        print(f"⚠️ Search Sweep Warning: No system database files detected inside structural target space [{folder_path}].")
        return results
        
    print(f"🔍 Sweeping execution target space... Scanning {len(json_files)} target metadata profiles...")
    print("=" * 70)
    
    for file_path in json_files:
        results['total'] += 1
        is_valid = validate_rubric_file(str(file_path))
        if is_valid:
            results['valid'] += 1
            results['files'].append({'name': file_path.name, 'status': 'valid'})
        else:
            results['invalid'] += 1
            results['files'].append({'name': file_path.name, 'status': 'invalid'})
            
    print("=" * 70)
    print(f"📋 Verification Run Log Totals: Complete | Clean Pass: {results['valid']} | Errors Stopped: {results['invalid']}")
    return results


# ============================================
# Main Dev Framework Execution Logic Test
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("📋 Testing System Execution Layer: Phase 0 Rubric Structural Validation Framework")
    print("=" * 70)
    
    # 1. Generate perfect local mock tracking dictionary match setup
    print("\n[Step 1]: Compiling valid dynamic structural map testing layout matrix...")
    valid_mock_data = {
        "question_id": "sci_010",
        "subject": "Science",
        "grade": "10th",
        "question_text": "Explain the process of photosynthesis with explicit component structures.",
        "max_marks": 4.0,
        "rubric_points": [
            {
                "point_id": "p1",
                "concept": "Definition",
                "description": "Explains how green foliage processes raw light frequencies successfully",
                "marks": 2.0,
                "is_required": True,
                "keywords": ["sunlight", "light energy", "photosynthesis"],
                "diagram_required": False,
                "formula_required": False
            },
            {
                "point_id": "p2",
                "concept": "Site Locations",
                "description": "Pins specific biological processing locations to standard cellular chloroplast targets",
                "marks": 2.0,
                "is_required": True,
                "keywords": ["chlorophyll", "chloroplast", "thylakoid"],
                "diagram_required": True,
                "formula_required": False
            }
        ]
    }
    
    # 2. Fire parsing testing sequences
    try:
        tested_instance = AnswerKey(**valid_mock_data)
        print("✅ Core Framework Structure Sanity Check Passed! Engine compiles objects perfectly.")
        print(f"   Validated Default Fallback Cap Value -> Hard Floor Penalty Bound: {tested_instance.safety_config.hard_floor_score} Marks Max.")
        print(f"   Contradiction Threshold: {tested_instance.get_contradiction_threshold()}")
        print(f"   Required Points: {len(tested_instance.get_required_points())}")
        
        # Check quality
        quality = validate_rubric_quality(tested_instance)
        print(f"   Quality Score: {quality['score']}/100")
        
    except Exception as e:
        print(f"❌ Internal Mock Integrity Compiler Failure: Check runtime core constraints logic loops: {e}")
        
    # 3. Automated folder integration logic sequence sweep check execution
    print("\n[Step 2]: Attempting dynamic directory file scanning tracking array loops...")
    validate_all_rubrics("phase0/questions")
    
    # 4. Test invalid data with duplicate point_id
    print("\n[Step 3]: Testing duplicate point_id rejection...")
    invalid_mock_data = {
        "question_id": "sci_011",
        "subject": "Science",
        "grade": "10th",
        "question_text": "Photosynthesis?",
        "max_marks": 5.0,
        "rubric_points": [
            {
                "point_id": "p1",
                "concept": "Definition",
                "description": "Explains photosynthesis",
                "marks": 3.0,
                "is_required": True
            },
            {
                "point_id": "p1",  # ← DUPLICATE!
                "concept": "Equation",
                "description": "Writes equation",
                "marks": 2.0,
                "is_required": True
            }
        ]
    }
    
    try:
        invalid_instance = AnswerKey(**invalid_mock_data)
        print("❌ Should have failed! Duplicate point_id not detected")
    except ValueError as e:
        print(f"✅ Correctly rejected invalid data: {str(e)[:80]}...")