"""
backend/app/services/llm_client.py
LLM Client Factory — Gemini for Phase 2, Groq for Phase 4
==========================================================
Uses the same OpenAI SDK with different base_url.

Supported providers:
  - Gemini (Phase 2): Rubric creation (FREE: 1,500 req/day)
  - Groq (Phase 4): Student evaluation (FREE: 14,400 req/day)

No OpenAI fallback — uses Gemini + Groq only.
"""

import os
import sys
from openai import OpenAI
from typing import Literal, Optional
from dotenv import load_dotenv
from pathlib import Path

# Force UTF-8 encoding for stdout and stderr to support emoji output in Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Load .env file from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)


# ── Model configurations ────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # Phase 2: Gemini (Rubric creation)
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": {
            "gemini-2.0-flash": {
                "model": "gemini-2.0-flash",
                "max_tokens": 2000,
                "note": "Fast, free, good for rubric creation"
            },
            "gemini-1.5-flash": {
                "model": "gemini-1.5-flash",
                "max_tokens": 2000,
                "note": "Slightly older, still good"
            }
        },
        "default_model": "gemini-2.0-flash",
        "rate_limit": {
            "requests_per_minute": 15,
            "requests_per_day": 1500
        },
        "note": "Phase 2: Rubric creation (5-10 calls/day)"
    },
    
    # Phase 4: Groq (Student evaluation)
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": {
            "llama-3.3-70b-versatile": {
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 4096,
                "note": "Best quality on Groq (de facto standard)"
            },
            "llama-3.1-8b-instant": {
                "model": "llama-3.1-8b-instant",
                "max_tokens": 4096,
                "note": "Fast, cheap, good for simple eval"
            }
        },
        "default_model": "llama-3.3-70b-versatile",
        "rate_limit": {
            "requests_per_minute": 30,
            "requests_per_day": 14400
        },
        "note": "Phase 4: Student evaluation (50-100 calls/day)"
    }
}


# ── Client factory ─────────────────────────────────────────────────────────

def get_llm_client(provider: Literal["gemini", "groq"]) -> OpenAI:
    """
    Returns an OpenAI-compatible client for the specified provider.
    
    Args:
        provider: "gemini" (Phase 2) or "groq" (Phase 4)
    
    Returns:
        OpenAI client configured with appropriate API key and base_url
    
    Raises:
        ValueError: If API key is missing or provider is invalid
    """
    
    if provider not in MODEL_CONFIGS:
        raise ValueError(
            f"Invalid provider: {provider}. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[provider]
    api_key = os.environ.get(config["api_key_env"])
    
    if not api_key:
        raise ValueError(
            f"{config['api_key_env']} not set. "
            f"Add it to your .env file.\n"
            f"Example: {config['api_key_env']}=your-api-key-here"
        )
    
    # ✅ REMOVED format validation - AQ.Ab... keys are valid for Gemini!
    # No need to check for "AIzaSy" prefix
    
    return OpenAI(
        api_key=api_key,
        base_url=config["base_url"]
    )


def get_model_config(provider: Literal["gemini", "groq"], model_name: Optional[str] = None) -> dict:
    """
    Get model configuration for a specific provider and model.
    """
    config = MODEL_CONFIGS[provider]
    
    if not model_name:
        model_name = config["default_model"]
    
    if model_name not in config["models"]:
        available = list(config["models"].keys())
        raise ValueError(
            f"Model '{model_name}' not found for {provider}. "
            f"Available: {available}"
        )
    
    return config["models"][model_name]


def get_default_model(provider: Literal["gemini", "groq"]) -> str:
    """Get the default model for a provider."""
    return MODEL_CONFIGS[provider]["default_model"]


def get_provider_note(provider: str) -> str:
    """Get the note/description for a provider."""
    return MODEL_CONFIGS.get(provider, {}).get("note", "Unknown provider")


def get_rate_limit(provider: str) -> dict:
    """Get rate limit information for a provider."""
    return MODEL_CONFIGS.get(provider, {}).get("rate_limit", {})


def list_models(provider: str) -> list:
    """List all available models for a provider."""
    config = MODEL_CONFIGS.get(provider, {})
    return list(config.get("models", {}).keys())


def validate_api_key(provider: Literal["gemini", "groq"]) -> tuple[bool, str]:
    """
    Validate if API key exists and has correct format.
    Returns: (is_valid, message)
    """
    try:
        client = get_llm_client(provider)
        return True, f"✅ {provider.upper()} API key is present"
    except ValueError as e:
        return False, str(e)


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 LLM Client Test")
    print("=" * 60)
    
    # Test Gemini
    print("\n📌 Testing Gemini...")
    try:
        client = get_llm_client("gemini")
        print(f"✅ Gemini client created successfully")
        print(f"   Base URL: {client.base_url}")
        print(f"   Default model: {get_default_model('gemini')}")
    except ValueError as e:
        print(f"❌ Gemini error: {e}")
    
    # Test Groq
    print("\n📌 Testing Groq...")
    try:
        client = get_llm_client("groq")
        print(f"✅ Groq client created successfully")
        print(f"   Base URL: {client.base_url}")
        print(f"   Default model: {get_default_model('groq')}")
    except ValueError as e:
        print(f"❌ Groq error: {e}")
    
    # Show available models
    print("\n📌 Available Models:")
    print(f"   Gemini: {list_models('gemini')}")
    print(f"   Groq: {list_models('groq')}")
    
    print("\n" + "=" * 60)