# backend/app/utils/image_processor.py (Updated)

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

BLUR_REJECT_THRESHOLD: float = 100.0


@dataclass
class PreprocessResult:
    success: bool
    image: Optional[np.ndarray]
    blur_score: Optional[float]
    rejection_reason: Optional[str]
    pages: Optional[List[np.ndarray]] = None  # ✅ NEW: For multi-page support


# ── Step helpers ───────────────────────────────────────────────────────────

def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Shadow removal using background estimation"""
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, background)
    
    # ✅ IMPROVED: Use CLAHE instead of global normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    diff_normalized = diff_normalized.astype(np.uint8)
    enhanced = clahe.apply(diff_normalized)
    
    return enhanced


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten a page that was photographed at a slight angle"""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return gray

    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _perspective_correction(img: np.ndarray) -> np.ndarray:
    """✅ NEW: Perspective correction for angled phone photos"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    # Find the largest contour (the page)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to a quadrilateral
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    if len(approx) != 4:
        return img  # Not a rectangle, skip
    
    # Get the four corners
    pts = approx.reshape(4, 2)
    
    # Sort corners: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Get width and height
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(tr - tl)
    width_b = np.linalg.norm(br - bl)
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(bl - tl)
    height_b = np.linalg.norm(br - tr)
    max_height = max(int(height_a), int(height_b))
    
    # Warp to rectangle
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    
    return warped


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize the image using Gaussian adaptive thresholding"""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15
    )


def _split_pages(image: np.ndarray) -> List[np.ndarray]:
    """✅ NEW: Split multi-page answer sheets into individual pages"""
    # Simple detection: if height > 2x width, split in half
    h, w = image.shape[:2]
    if h > w * 1.5:
        mid = h // 2
        page1 = image[:mid, :]
        page2 = image[mid:, :]
        return [page1, page2]
    return [image]


# ── Public API ─────────────────────────────────────────────────────────────

def preprocess_image(image_path: str, split_pages: bool = False) -> PreprocessResult:
    """
    Run the full preprocessing guardrail with improvements.
    
    Args:
        image_path: Path to the image file
        split_pages: If True, split multi-page images
    """
    img = cv2.imread(image_path)
    if img is None:
        return PreprocessResult(
            success=False, image=None, blur_score=None,
            rejection_reason="Could not read image file"
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Gate 1: blur check ─────────────────────────────────────────────────
    score = _blur_score(gray)
    if score < BLUR_REJECT_THRESHOLD:
        return PreprocessResult(
            success=False, image=None, blur_score=score,
            rejection_reason=f"Image too blurry (score={score:.1f})"
        )

    # ── Gate 2: perspective correction ────────────────────────────────────
    img = _perspective_correction(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Steps 3-5: processing ──────────────────────────────────────────────
    gray = _remove_shadows(gray)
    gray = _deskew(gray)
    processed = _adaptive_threshold(gray)

    # ── Step 6: multi-page split ──────────────────────────────────────────
    pages = None
    if split_pages:
        pages = _split_pages(processed)

    return PreprocessResult(
        success=True, image=processed, 
        blur_score=score, rejection_reason=None,
        pages=pages
    )


def encode_image_to_bytes(image: np.ndarray) -> Optional[bytes]:
    """Encode a numpy image to PNG bytes"""
    success, buf = cv2.imencode(".png", image)
    if not success:
        return None
    return buf.tobytes()