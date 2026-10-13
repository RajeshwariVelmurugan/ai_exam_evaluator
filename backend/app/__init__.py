# backend/app/utils/__init__.py
"""
backend/app/utils/__init__.py
Phase 1 — Utilities Package
============================
"""

# Change from:
# from app.utils.image_processor import preprocess_image, encode_image_to_bytes

# To:
from backend.app.utils.image_processor import preprocess_image, encode_image_to_bytes

__all__ = [
    "preprocess_image",
    "encode_image_to_bytes",
]