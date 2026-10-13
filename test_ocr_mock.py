"""
test_ocr_mock.py - Phase 1 Mock Test Script
============================================
Demonstrates Phase 1 output without calling Google Vision API.
Use this for testing the output format and UI.

Usage:
    python test_ocr_mock.py
"""

import random
import json


def mock_ocr_success():
    """Mock a successful OCR result"""
    return {
        "status": "success",
        "text": "Photosynthesis is the process by which green plants manufacture food using sunlight. It occurs in the chloroplasts which contain chlorophyll. The raw materials are carbon dioxide and water. The products are glucose and oxygen. The equation is: 6CO2 + 6H2O + Light Energy → C6H12O6 + 6O2.",
        "confidence": round(random.uniform(0.85, 0.98), 4),
        "blur_score": round(random.uniform(150, 500), 1),
        "reason": None
    }


def mock_ocr_rejected():
    """Mock a rejected OCR result (blurry)"""
    return {
        "status": "rejected",
        "text": None,
        "confidence": None,
        "blur_score": round(random.uniform(20, 90), 1),
        "reason": "Image too blurry (score={:.1f}, threshold=100.0). Please retake the photo in better lighting, holding the phone steady.".format(random.uniform(20, 90))
    }


def mock_ocr_error():
    """Mock an error OCR result (API failure)"""
    errors = [
        "Vision API error: Project not enabled for Vision API. Please enable the API in Google Cloud Console.",
        "Vision API error: Invalid API key. Please check your credentials.",
        "Vision API error: Quota exceeded. Please try again later."
    ]
    return {
        "status": "error",
        "text": None,
        "confidence": None,
        "blur_score": round(random.uniform(120, 300), 1),
        "reason": random.choice(errors)
    }


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


def run_mock_test(case: str = "all"):
    """Run mock tests for different cases"""
    
    cases = {
        "success": mock_ocr_success,
        "rejected": mock_ocr_rejected,
        "error": mock_ocr_error
    }
    
    if case == "all":
        print("\n" + "=" * 70)
        print("📷 PHASE 1 OCR PIPELINE - ALL OUTPUT CASES")
        print("=" * 70)
        
        for case_name, case_func in cases.items():
            result = case_func()
            print_result(result)
            
            # Wait for user input between cases
            if case_name != "error":
                input("\n⏸️  Press Enter to continue to next case...")
                print("\n" + "=" * 70)
    else:
        result = cases[case]()
        print_result(result)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📷 PHASE 1 OCR PIPELINE - MOCK TEST DEMO")
    print("=" * 70)
    
    print("\nChoose an option:")
    print("  1. Show all three cases (Success, Rejected, Error)")
    print("  2. Show only Success case")
    print("  3. Show only Rejected case")
    print("  4. Show only Error case")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_mock_test("all")
    elif choice == "2":
        run_mock_test("success")
    elif choice == "3":
        run_mock_test("rejected")
    elif choice == "4":
        run_mock_test("error")
    else:
        print("Invalid choice. Running all cases...")
        run_mock_test("all")