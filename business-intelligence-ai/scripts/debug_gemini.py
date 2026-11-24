"""
Standalone script to test your Gemini API key directly, bypassing the app's
silent fallback, so you can see the actual error if something's wrong.

Usage:
    pip install python-dotenv google-genai
    python scripts/debug_gemini.py
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("(python-dotenv not installed — reading OS environment only; "
          "run `pip install python-dotenv` or set GOOGLE_API_KEY manually)")

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

print(f"Key found: {bool(api_key)}")
if api_key:
    print(f"Key prefix: {api_key[:8]}...")
    if not api_key.startswith("AIza"):
        print("⚠️  WARNING: Gemini API keys normally start with 'AIzaSy'. "
              "This key does not match that pattern — it may not be a valid "
              "Gemini API key. Get one at https://aistudio.google.com/apikey")

if not api_key:
    print("No GOOGLE_API_KEY or GEMINI_API_KEY found in environment or .env file. Stopping.")
    raise SystemExit(1)

print("\nAttempting a live call to Gemini...")
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model="gemini-2.0-flash", contents="Say hello in one word.")
    print("SUCCESS. Response:", resp.text)
except Exception as e:
    print("FAILED. Full error below:\n")
    print(repr(e))
