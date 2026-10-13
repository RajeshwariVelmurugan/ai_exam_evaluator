"""
test_ocr.py - Phase 1 Test Script with Formatted Output
========================================================
Tests the OCR pipeline with a sample image and displays
beautiful formatted output.

Usage:
    python test_ocr.py path/to/your/image.jpg
"""

import sys
import os
import json

# Force UTF-8 encoding for stdout and stderr to support emoji output in Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from backend.app.services.ocr_service import run_ocr


def print_header(text: str, char: str = "="):
    """Print a formatted header"""
    print("\n" + char * 60)
    print(f"📷 {text}")
    print(char * 60)


def print_section(text: str):
    """Print a section divider"""
    print("-" * 60)
    print(text)
    print("-" * 60)


def print_result(result: dict):
    """Display OCR result in a beautiful formatted way"""
    
    status = result['status']
    
    # Print header based on status
    if status == 'success':
        print_header("Phase 1: OCR Pipeline Test - SUCCESS ✅", "=")
    elif status == 'rejected':
        print_header("Phase 1: OCR Pipeline Test - REJECTED ❌", "=")
    else:  # error
        print_header("Phase 1: OCR Pipeline Test - ERROR ⚠️", "=")
    
    # Print results table
    print("\n📊 RESULTS:")
    print_section("")
    
    # Status with emoji
    status_emoji = "✅" if status == 'success' else "❌" if status == 'rejected' else "⚠️"
    print(f"  Status     : {status_emoji} {status}")
    
    # Confidence
    if result['confidence'] is not None:
        print(f"  Confidence : {result['confidence']*100:.2f}%")
    else:
        print(f"  Confidence : None")
    
    # Blur Score
    if result['blur_score'] is not None:
        threshold = 100.0
        passed = result['blur_score'] >= threshold
        status_text = "✅ Passed" if passed else "❌ Failed"
        print(f"  Blur Score : {result['blur_score']:.1f} (Threshold: 100.0) {status_text}")
    else:
        print(f"  Blur Score : None")
    
    print("-" * 60)
    
    # Print appropriate content based on status
    if status == 'success':
        # Success - Show extracted text
        print("\n📝 EXTRACTED TEXT:")
        print_section("")
        
        # Display text with word wrap
        text = result['text']
        if text:
            # Print first 500 chars with formatting
            if len(text) > 500:
                print(text[:500] + "\n... (truncated)")
            else:
                print(text)
        else:
            print("No text extracted")
        
        print("-" * 60)
        print("\n✅ Pipeline: Preprocessing → Vision API → Text Extraction → Success")
        
    elif status == 'rejected':
        # Rejected - Show rejection reason
        print("\n❌ REJECTION REASON:")
        print_section("")
        print(result['reason'])
        print("-" * 60)
        print("\n❌ Pipeline: Preprocessing → Blur Check Failed → Rejected")
        print("💡 Tip: Retake photo with better lighting and steady hands")
        
    else:  # error
        # Error - Show error details
        print("\n⚠️ ERROR DETAILS:")
        print_section("")
        print(result['reason'])
        print("-" * 60)
        print("\n⚠️ Pipeline: Preprocessing → Vision API → API Error")
        print("🔧 Fix: Check your Google Cloud Vision API credentials and enable the API")
    
    print("=" * 60 + "\n")
    
    # Print raw JSON for debugging
    print("📋 RAW JSON OUTPUT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + "=" * 60 + "\n")


def run_ocr_test(image_path: str):
    """Test OCR on a single image"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"📷 Processing: {image_path}")
    
    # Run OCR
    result = run_ocr(image_path)
    
    # Display formatted output
    print_result(result)
    
    return result


if __name__ == "__main__":
    # Check command line args
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: ask user for path
        image_path = input("📷 Enter path to image file: ")

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Usage: python test_ocr.py path/to/image.jpg")
        sys.exit(1)

    run_ocr_test(image_path)