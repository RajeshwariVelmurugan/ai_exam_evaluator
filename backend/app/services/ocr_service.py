"""
backend/app/services/ocr_service.py
Phase 1 — OCR Service (Google Cloud Vision)
============================================
Receives a raw image path, runs the preprocessing guardrail from
image_processor.py, then calls the Vision API.

Returns a clean dict:
  {
    "status":     "success" | "rejected" | "error",
    "text":       str | None,
    "confidence": float | None,     # 0.0 – 1.0
    "blur_score": float | None,
    "reason":     str | None,       # set on rejected / error
  }

Downstream consumers (Phase 3 NLI, Phase 4 LLM) only need to check
status == "success" and then use the `text` field.

Setup
-----
1. pip install google-cloud-vision opencv-python numpy
2. Create a Google Cloud project → enable Vision API
3. Create a service-account key (JSON) and set:
       export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   Or add it to your .env and load with python-dotenv.
"""

from __future__ import annotations

import os
from typing import TypedDict

from google.cloud import vision

from backend.app.utils.image_processor import preprocess_image, encode_image_to_bytes


# ── Return shape ───────────────────────────────────────────────────────────

class OCRResult(TypedDict):
    status: str           # "success" | "rejected" | "error"
    text: str | None
    confidence: float | None
    blur_score: float | None
    reason: str | None


# ── Internal Vision API call ───────────────────────────────────────────────

def _call_vision_api(image_bytes: bytes) -> dict:
    """
    Calls document_text_detection (better for handwriting than
    plain text_detection — handles cursive, mixed print/handwriting).

    Returns {"text": str, "confidence": float}.
    Raises RuntimeError on API-level errors.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text or ""

    # Vision returns per-word confidence, not one global score.
    # Aggregate across all detected words.
    confidences: list[float] = [
        word.confidence
        for page in response.full_text_annotation.pages
        for block in page.blocks
        for paragraph in block.paragraphs
        for word in paragraph.words
        if word.confidence is not None
    ]
    avg_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    return {"text": full_text, "confidence": round(avg_confidence, 4)}


# ── Public API ─────────────────────────────────────────────────────────────

def run_ocr(image_path: str) -> OCRResult:
    """
    Full Phase 1 pipeline for one uploaded answer-sheet image.

    Steps:
      1. Preprocess + blur gate  (image_processor.py)
      2. Encode processed image to bytes
      3. Call Google Vision API
      4. Return structured OCRResult

    This is the function Phase 6 (FastAPI background task) calls.
    """

    # ── Step 1: preprocess ─────────────────────────────────────────────────
    preprocess = preprocess_image(image_path)

    if not preprocess.success:
        return OCRResult(
            status="rejected",
            text=None,
            confidence=None,
            blur_score=preprocess.blur_score,
            reason=preprocess.rejection_reason,
        )

    # ── Step 2: encode ─────────────────────────────────────────────────────
    image_bytes = encode_image_to_bytes(preprocess.image)
    if image_bytes is None:
        return OCRResult(
            status="error",
            text=None,
            confidence=None,
            blur_score=preprocess.blur_score,
            reason="Failed to encode processed image to bytes",
        )

    # ── Step 3: OCR API call ───────────────────────────────────────────────
    try:
        ocr = _call_vision_api(image_bytes)
    except RuntimeError as e:
        return OCRResult(
            status="error",
            text=None,
            confidence=None,
            blur_score=preprocess.blur_score,
            reason=str(e),
        )

    if not ocr["text"].strip():
        return OCRResult(
            status="error",
            text=None,
            confidence=ocr["confidence"],
            blur_score=preprocess.blur_score,
            reason="Vision API returned empty text — image may be blank or upside-down",
        )

    return OCRResult(
        status="success",
        text=ocr["text"],
        confidence=ocr["confidence"],
        blur_score=preprocess.blur_score,
        reason=None,
    )