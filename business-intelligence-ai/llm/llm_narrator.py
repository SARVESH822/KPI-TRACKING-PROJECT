"""
Optional LLM enrichment. Receives ONLY the pre-computed structured evidence
object (never raw transaction rows) and rewrites/expands it into polished
natural language. If no key is configured, the call fails, times out, hits a
quota/rate limit, or the response fails schema validation, this module
returns None and the caller falls back to llm.deterministic_narrator — the
app never crashes and never shows an unvalidated LLM claim.

Supports two providers, tried in this order based on configured keys:
  1. Gemini (GOOGLE_API_KEY) — default choice, generous free tier
  2. Anthropic (ANTHROPIC_API_KEY) — used if Gemini key absent or Gemini fails
"""
import json
import os
from typing import Optional

SYSTEM_PROMPT = """You are a business analytics narrator. You will be given a JSON \
"evidence object" containing already-computed KPI values, drivers, and \
retrieved supporting documents. Your job is ONLY to write natural-language \
narrative around these numbers - you must NOT invent, recompute, or alter any \
numeric value. You must NOT claim causation; use language like "is associated \
with", "coincides with", "may be a contributing factor". If evidence is weak, \
say so plainly rather than inventing a cause.

Respond with ONLY a JSON object (no markdown fences, no preamble) matching this schema:
{
  "executive_summary": string,
  "what_changed": string,
  "why_it_changed": [string, ...],
  "supporting_evidence": [string, ...],
  "recommendations": [
    {"action": string, "reason": string, "evidence": string, "priority": "High"|"Medium"|"Low", "confidence": "High"|"Medium"|"Low"}
  ]
}
"""


def _validate_schema(obj: dict) -> bool:
    required_top = ["executive_summary", "what_changed", "why_it_changed",
                     "supporting_evidence", "recommendations"]
    if not all(k in obj for k in required_top):
        return False
    if not isinstance(obj["why_it_changed"], list) or not isinstance(obj["supporting_evidence"], list):
        return False
    if not isinstance(obj["recommendations"], list):
        return False
    for r in obj["recommendations"]:
        if not all(k in r for k in ["action", "reason", "evidence", "priority", "confidence"]):
            return False
        if r["priority"] not in ("High", "Medium", "Low"):
            return False
        if r["confidence"] not in ("High", "Medium", "Low"):
            return False
    return True


def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not _validate_schema(obj):
        return None
    return obj


def _try_gemini(evidence_object: dict, model: str, timeout_s: float) -> Optional[dict]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=json.dumps(evidence_object),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1200,
                temperature=0.3,
            ),
        )
        text = resp.text or ""
        obj = _parse_json_response(text)
        if obj:
            obj["generated_by"] = "llm"
            obj["provider"] = "gemini"
        return obj
    except Exception:
        # Covers quota exhaustion (429), auth errors, timeouts, malformed output
        return None


def _try_anthropic(evidence_object: dict, model: str, timeout_s: float) -> Optional[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(evidence_object)}],
            timeout=timeout_s,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        obj = _parse_json_response(text)
        if obj:
            obj["generated_by"] = "llm"
            obj["provider"] = "anthropic"
        return obj
    except Exception:
        return None


def generate_story_with_llm(evidence_object: dict, gemini_model: str = "gemini-2.0-flash",
                             anthropic_model: str = "claude-sonnet-4-6",
                             timeout_s: float = 20.0) -> Optional[dict]:
    """Tries Gemini first (free-tier friendly), then Anthropic, then gives up."""
    result = _try_gemini(evidence_object, gemini_model, timeout_s)
    if result:
        return result
    result = _try_anthropic(evidence_object, anthropic_model, timeout_s)
    if result:
        return result
    return None
