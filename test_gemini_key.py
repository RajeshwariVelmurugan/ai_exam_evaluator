# test_gemini_3_flash.py
"""
Testing Gemini 3 Flash with OpenAI SDK
=======================================
Gemini provides an OpenAI-compatible endpoint,
so we can use the OpenAI Python library!

Why this works:
- OpenAI SDK is just an HTTP client
- Changing base_url redirects to Gemini
- Gemini accepts OpenAI-style requests
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get Gemini API key from .env
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    print("❌ GEMINI_API_KEY not found in .env!")
    exit()

print(f"✅ API Key found: {gemini_api_key[:15]}...")

# ── Create client with Gemini endpoint ──────────────────────────────────
# We're using OpenAI client but pointing to Gemini's API
client = OpenAI(
    api_key=gemini_api_key,  # ← Gemini API key (not OpenAI!)
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # ← Gemini URL
)

print("📡 Calling Gemini 3 Flash...")

# ── Make the call ──────────────────────────────────────────────────────
response = client.chat.completions.create(
    model="gemini-3-flash-preview",  # ← Gemini model name
    messages=[
        {"role": "user", "content": "Hello, Gemini 3 Flash! Tell me a fun fact."}
    ],
    max_tokens=100
)

print("\n✅ Response:")
print("=" * 50)
print(response.choices[0].message.content)
print("=" * 50)