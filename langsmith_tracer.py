"""
langsmith_tracer.py – LangSmith Token Monitoring & Execution Tracing Helper

Provides utilities for:
1. LangSmith traceable decorators with graceful fallback.
2. Extracting token usage metrics (prompt_tokens, completion_tokens, total_tokens) from LLM responses (Gemini & OpenAI-style APIs).
3. Structuring run metadata and token telemetry logging.
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if LangSmith tracing is active and valid API key is present
def is_langsmith_enabled() -> bool:
    tracing_env = (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1")
        or os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1")
    )
    api_key = (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
    is_valid_key = bool(api_key) and not api_key.startswith("your_")
    return tracing_env and is_valid_key


# Dynamic import of traceable decorator from langsmith
try:
    from langsmith import traceable as _langsmith_traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    _langsmith_traceable = None


# --- PRICING CONSTANTS (USD $) ---
# Gemini 2.5 Flash / Gemini 2.0 Flash pricing: $0.075 / 1M prompt tokens, $0.30 / 1M completion tokens
GEMINI_PROMPT_PRICE_PER_TOKEN     = 0.075 / 1_000_000
GEMINI_COMPLETION_PRICE_PER_TOKEN = 0.30 / 1_000_000

# ElevenLabs Scribe v1 pricing: ~$0.006 / minute = $0.0001 / second
ELEVENLABS_SCRIBE_PRICE_PER_SEC   = 0.0001


def calculate_token_cost_usd(prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    """Calculates LLM cost in USD ($) based on prompt and completion token counts."""
    p_cost = (prompt_tokens or 0) * GEMINI_PROMPT_PRICE_PER_TOKEN
    c_cost = (completion_tokens or 0) * GEMINI_COMPLETION_PRICE_PER_TOKEN
    return round(p_cost + c_cost, 7)


def calculate_stt_cost_usd(duration_secs: float = 0.0) -> float:
    """Calculates ElevenLabs Scribe STT audio cost in USD ($)."""
    return round((duration_secs or 0) * ELEVENLABS_SCRIBE_PRICE_PER_SEC, 6)


def get_traceable_decorator(
    name: Optional[str] = None,
    run_type: str = "chain",
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
):
    """
    Returns a @traceable decorator if langsmith is installed and configured with an API key,
    otherwise returns a pass-through decorator.
    """
    if LANGSMITH_AVAILABLE and _langsmith_traceable and is_langsmith_enabled():
        return _langsmith_traceable(
            name=name,
            run_type=run_type,
            tags=tags or ["speech-to-text-summarisation"],
            metadata=metadata
        )
    else:
        def decorator(func: Callable):
            return func
        return decorator


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""

    @property
    def cost_usd(self) -> float:
        return calculate_token_cost_usd(self.prompt_tokens, self.completion_tokens)

    def to_dict(self) -> Dict[str, Any]:
        def _safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        p = _safe_int(self.prompt_tokens)
        c = _safe_int(self.completion_tokens)
        t = _safe_int(self.total_tokens)
        return {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
            "model_name": str(self.model_name),
            "cost_usd": calculate_token_cost_usd(p, c)
        }


def extract_token_usage(response: Any, model_name: str = "") -> TokenUsage:
    """
    Extracts token usage metrics from OpenAI-like or Google Gemini API responses.
    """
    usage = TokenUsage(model_name=model_name)

    if not response:
        return usage

    try:
        # 1. Standard OpenAI-style response object
        if hasattr(response, "usage") and response.usage:
            raw_usage = response.usage
            prompt = getattr(raw_usage, "prompt_tokens", 0) or 0
            completion = getattr(raw_usage, "completion_tokens", 0) or 0
            total = getattr(raw_usage, "total_tokens", 0) or (prompt + completion if isinstance(prompt, int) and isinstance(completion, int) else 0)
            usage.prompt_tokens = prompt
            usage.completion_tokens = completion
            usage.total_tokens = total
            return usage

        # 2. Google Gemini response (GenerateContentResponse)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            prompt = getattr(meta, "prompt_token_count", 0) or getattr(meta, "prompt_tokens", 0) or 0
            completion = getattr(meta, "candidates_token_count", 0) or getattr(meta, "completion_tokens", 0) or 0
            total = getattr(meta, "total_token_count", 0) or (prompt + completion if isinstance(prompt, int) and isinstance(completion, int) else 0)
            usage.prompt_tokens = prompt
            usage.completion_tokens = completion
            usage.total_tokens = total
            return usage

        # 3. Dict-style usage (e.g. JSON responses)
        if isinstance(response, dict) and "usage" in response:
            u = response["usage"]
            usage.prompt_tokens = u.get("prompt_tokens", 0)
            usage.completion_tokens = u.get("completion_tokens", 0)
            usage.total_tokens = u.get("total_tokens", usage.prompt_tokens + usage.completion_tokens)
            return usage

    except Exception as e:
        logger.debug(f"Could not extract token usage: {e}")

    return usage


def log_token_usage(usage: TokenUsage, context_label: str = "LLM Call") -> None:
    """
    Logs token metrics cleanly to standard application output.
    """
    def _safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    p_tok = _safe_int(usage.prompt_tokens)
    c_tok = _safe_int(usage.completion_tokens)
    t_tok = _safe_int(usage.total_tokens)

    logger.info(
        f"📊 [{context_label}] Model: {usage.model_name or 'unknown'} | "
        f"Prompt Tokens: {p_tok:,} | "
        f"Completion Tokens: {c_tok:,} | "
        f"Total Tokens: {t_tok:,}"
    )

