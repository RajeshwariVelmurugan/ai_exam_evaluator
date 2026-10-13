"""
tests/test_phase2_rubric_extractor.py
Unit tests for Phase 2 — Rubric Extractor
==========================================
Tests the non-LLM parts of the extractor (parsing, cleaning, marks fixing,
schema validation) without making real API calls.

Run:
  pytest tests/test_phase2_rubric_extractor.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.services.rubric_extractor import (
    _clean_llm_response,
    _fix_marks_sum,
    extract_rubric,
)


# ── Helper: build a valid rubric dict ─────────────────────────────────────

def _sample_rubric_dict(max_marks: float = 10.0) -> dict:
    return {
        "question_id": "test_001",
        "subject": "Science",
        "grade": "10th",
        "question_text": "Explain photosynthesis.",
        "max_marks": max_marks,
        "answer_key_text": "Photosynthesis is...",
        "difficulty": "medium",
        "topic": "Plant Biology",
        "created_by": "teacher",
        "min_word_count": 30,
        "max_word_count": 200,
        "allow_extra_points": True,
        "penalty_for_hallucination": 2.0,
        "marking_scheme": {
            "content_weight": 0.7,
            "logic_weight": 0.2,
            "grammar_weight": 0.1,
            "passing_percentage": 40,
            "distinction_percentage": 75,
        },
        "rubric_points": [
            {
                "point_id": "p1",
                "concept": "Definition",
                "description": "States that photosynthesis needs sunlight",
                "marks": 4.0,
                "is_required": True,
                "partial_credit_allowed": True,
                "keywords": ["sunlight", "light energy"],
                "partial_keywords": ["sunlight"],
                "examples": ["Plants need sunlight for photosynthesis"],
                "common_mistakes": ["Says photosynthesis doesn't need sunlight"],
            },
            {
                "point_id": "p2",
                "concept": "Products",
                "description": "States glucose and oxygen as products",
                "marks": 6.0,
                "is_required": True,
                "partial_credit_allowed": True,
                "keywords": ["glucose", "oxygen"],
                "partial_keywords": ["glucose"],
                "examples": [],
                "common_mistakes": [],
            },
        ],
    }


# ── _clean_llm_response ────────────────────────────────────────────────────

class TestCleanLLMResponse:
    def test_strips_json_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert _clean_llm_response(raw) == '{"key": "value"}'

    def test_strips_plain_fence(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        assert _clean_llm_response(raw) == '{"key": "value"}'

    def test_no_fence_unchanged(self):
        raw = '{"key": "value"}'
        assert _clean_llm_response(raw) == '{"key": "value"}'

    def test_strips_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        assert _clean_llm_response(raw) == '{"key": "value"}'


# ── _fix_marks_sum ─────────────────────────────────────────────────────────

class TestFixMarksSum:
    def test_no_change_when_sum_correct(self):
        data = _sample_rubric_dict(10.0)
        result = _fix_marks_sum(data, 10.0)
        assert sum(p["marks"] for p in result["rubric_points"]) == 10.0

    def test_fixes_small_rounding_gap(self):
        data = _sample_rubric_dict(10.0)
        data["rubric_points"][-1]["marks"] = 5.9   # sum = 9.9 instead of 10
        result = _fix_marks_sum(data, 10.0)
        total = sum(p["marks"] for p in result["rubric_points"])
        assert abs(total - 10.0) < 0.01

    def test_raises_on_large_gap(self):
        data = _sample_rubric_dict(10.0)
        data["rubric_points"][-1]["marks"] = 3.0   # sum = 7.0 — gap = 3
        with pytest.raises(ValueError, match="too large"):
            _fix_marks_sum(data, 10.0)

    def test_empty_points_returns_unchanged(self):
        data = {"rubric_points": []}
        result = _fix_marks_sum(data, 10.0)
        assert result == {"rubric_points": []}


# ── extract_rubric (mocked LLM) ────────────────────────────────────────────

class TestExtractRubric:
    """Mock the OpenAI call so tests run without a real API key."""

    def _make_mock_llm(self, response_dict: dict):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(response_dict)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @patch("backend.app.services.rubric_extractor.get_llm_client")
    def test_success(self, mock_get_llm_client, tmp_path):
        data = _sample_rubric_dict(10.0)
        mock_get_llm_client.return_value = self._make_mock_llm(data)

        with patch("backend.app.services.rubric_extractor.QUESTIONS_FOLDER", tmp_path), \
             patch("backend.app.services.rubric_extractor.CACHE_FOLDER", tmp_path):
            result = extract_rubric(
                question_id="test_001",
                subject="Science",
                grade="10th",
                question_text="Explain photosynthesis.",
                answer_key_text="Photosynthesis is...",
                max_marks=10.0,
                save=True,
            )

        assert result.status == "success"
        assert result.rubric is not None
        assert result.rubric.question_id == "test_001"
        assert (tmp_path / "test_001.json").exists()

    @patch("backend.app.services.rubric_extractor.get_llm_client")
    def test_retries_on_invalid_json(self, mock_get_llm_client, tmp_path):
        good_data = _sample_rubric_dict(10.0)
        bad_response  = MagicMock()
        bad_response.choices[0].message.content = "not valid json {{"
        good_response = MagicMock()
        good_response.choices[0].message.content = json.dumps(good_data)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            bad_response,
            good_response,
        ]
        mock_get_llm_client.return_value = mock_client

        with patch("backend.app.services.rubric_extractor.QUESTIONS_FOLDER", tmp_path), \
             patch("backend.app.services.rubric_extractor.CACHE_FOLDER", tmp_path):
            result = extract_rubric(
                question_id="test_001",
                subject="Science",
                grade="10th",
                question_text="Explain photosynthesis.",
                answer_key_text="Photosynthesis is...",
                max_marks=10.0,
                save=False,
            )

        assert result.status == "success"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("backend.app.services.rubric_extractor.get_llm_client")
    def test_error_after_max_retries(self, mock_get_llm_client, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "still bad json {{"
        mock_get_llm_client.return_value = mock_client

        with patch("backend.app.services.rubric_extractor.QUESTIONS_FOLDER", tmp_path), \
             patch("backend.app.services.rubric_extractor.CACHE_FOLDER", tmp_path):
            result = extract_rubric(
                question_id="test_001",
                subject="Science",
                grade="10th",
                question_text="Explain photosynthesis.",
                answer_key_text="Photosynthesis is...",
                max_marks=10.0,
                save=False,
            )

        assert result.status == "error"
        assert "JSON parse error" in result.reason