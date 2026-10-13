# backend/app/services/__init__.py
"""
backend/app/services/__init__.py
Phase 2 + Phase 4 — Services Package
======================================
"""

# Change from:
# from app.services.ocr_service import run_ocr

# To:
from backend.app.services.ocr_service import run_ocr
from backend.app.services.rubric_extractor import extract_rubric

__all__ = [
    "run_ocr",
    "extract_rubric",
]