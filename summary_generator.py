"""
summary_generator.py – Executive Summary & Multilingual Translation Generator
Utilizes Google GenAI (Gemini 2.5 Flash / Gemini 2.0 Flash) for zero-hallucination summaries and translation.
"""

import os
import re
import time
import sys
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from langsmith_tracer import (
    get_traceable_decorator,
    extract_token_usage,
    log_token_usage,
    TokenUsage
)

DEFAULT_PRIMARY_MODEL = os.getenv("SUMMARY_MODEL", os.getenv("PRIMARY_MODEL", "gemini-2.5-flash"))
DEFAULT_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

LANGUAGE_NAMES: dict = {
    "en": "English", "eng": "English",
    "hi": "हिंदी", "hin": "हिंदी",
    "te": "తెలుగు", "tel": "తెలుగు",
    "ta": "தமிழ்", "tam": "தமிழ்",
    "kn": "ಕನ್ನಡ", "kan": "ಕನ್ನಡ",
    "ml": "മലയാളം", "mal": "മലയാളം",
    "mr": "मराठी", "mar": "मराठी",
    "gu": "ગુજરાતી", "guj": "ગુજરાતી",
    "pa": "ਪੰਜਾਬੀ", "pan": "ਪੰਜਾਬੀ",
    "bn": "বাংলা", "ben": "বাংলা",
    "ur": "اردو", "urd": "اردو",
    "or": "ଓଡ଼ିଆ", "ori": "ଓଡ଼ିଆ",
    "es": "Español", "spa": "Español",
    "fr": "Français", "fra": "Français", "fre": "Français",
    "de": "Deutsch", "deu": "Deutsch", "ger": "Deutsch",
    "it": "Italiano", "ita": "Italiano",
    "pt": "Português", "por": "Português",
    "ru": "Русский", "rus": "Русский",
    "zh": "中文", "zho": "中文", "chi": "中文",
    "ja": "日本語", "jpn": "日本語",
    "ko": "한국어", "kor": "한국어",
    "ar": "العربية", "ara": "العربية",
    "tr": "Türkçe", "tur": "Türkçe",
    "vi": "Tiếng Việt", "vie": "Tiếng Việt",
    "th": "ไทย", "tha": "ไทย",
    "id": "Bahasa Indonesia", "ind": "Bahasa Indonesia",
    "ms": "Bahasa Melayu", "msa": "Bahasa Melayu",
    "nl": "Nederlands", "nld": "Nederlands", "dut": "Nederlands",
    "pl": "Polski", "pol": "Polski",
    "sv": "Svenska", "swe": "Svenska",
    "da": "Dansk", "dan": "Dansk",
    "fi": "Suomi", "fin": "Suomi",
    "no": "Norsk", "nor": "Norsk",
    "cs": "Čeština", "ces": "Čeština", "cze": "Čeština",
    "ro": "Română", "ron": "Română", "rum": "Română",
    "hu": "Magyar", "hun": "Magyar",
    "uk": "Українська", "ukr": "Українська",
    "el": "Ελληνικά", "ell": "Ελληνικά", "gre": "Ελληνικά",
    "he": "עברית", "heb": "עברית",
    "fa": "فارسی", "fas": "فارسی", "per": "فارسی",
}


def get_language_name(code: Optional[str]) -> Optional[str]:
    """Returns the full language name for an ISO code, or the code itself if unknown."""
    if not code:
        return None
    return LANGUAGE_NAMES.get(code.lower().split("-")[0], code)


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert multilingual audio transcription summarization assistant with zero tolerance for factual errors. "
    "Your objective is to summarize transcriptions directly in their original detected language (such as Hindi, Telugu, Kannada, Spanish, German, English, etc.) without translating them to English or any other language.\n\n"
    "CRITICAL ANTI-HALLUCINATION & PRECISION GUARDRAILS:\n"
    "1. PRESERVE ORIGINAL LANGUAGE: Generate and present all summary points strictly in the same detected source language as the input transcript. DO NOT translate the summary to English unless the input transcript is originally in English.\n"
    "2. STRICT NUMERICAL & FINANCIAL ACCURACY: NEVER alter, extrapolate, round, or recalculate numbers, currency figures (Rupees, Lakhs, Crores), dates, percentages, or ages. Copy every figure exactly as stated.\n"
    "3. NO NAME OR ROLE EXTRAPOLATION: Do NOT hallucinate political designations, minister roles, or imaginary events. Do NOT guess expansions for acronyms or institutions if not defined.\n"
    "4. PRECISE LANGUAGE ACCURACY: Pay exhaustive attention to grammatical context, vocabulary, and semantics in the transcript's native language and script.\n"
    "5. ZERO-INFERENCE RULE: Base every single point strictly on explicit statements in the text. If a detail, outcome, or title is not clearly mentioned, omit it entirely.\n\n"
    "OUTPUT STRUCTURE:\n"
    "1. **Title & Overview**: A factual single-sentence overview in the transcript's original language.\n"
    "2. **Detailed Summary**: A clean, structured bulleted list of verified facts, numerical figures, and public updates in the transcript's original language.\n"
    "Format the entire response beautifully in Markdown in the transcript's original language."
)


class SummaryGenerator:
    """
    High-accuracy summary generator utilizing Google GenAI (Gemini 2.5 / 2.0 Flash).
    Applies map-reduce chunking for long texts to prevent rate-limit crashes and preserve literal precision.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        primary_model: str = DEFAULT_PRIMARY_MODEL,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ):
        raw_key = api_key or gemini_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        keys = [k.strip() for k in raw_key.split(",") if k.strip()]
        self.gemini_keys: List[str] = list(dict.fromkeys(keys))
        
        self.primary_model = primary_model
        self.fallback_models = fallback_models if fallback_models is not None else DEFAULT_FALLBACK_MODELS

        self._gemini_clients = []
        if self.gemini_keys:
            try:
                from google import genai
                self._gemini_clients = [genai.Client(api_key=k) for k in self.gemini_keys]
                logger.info(f"Initialized Gemini engine with {len(self._gemini_clients)} API key(s).")
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI client: {e}")

        self.last_token_usage = TokenUsage()
        self.total_token_usage = TokenUsage()

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 10000) -> List[str]:
        """Splits long text into manageable chunks along paragraph or sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            if current_len + len(para) + 2 > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para) + 2
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    @get_traceable_decorator(name="generate_summary", run_type="chain")
    def generate_summary(
        self,
        transcript_text: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        on_fallback_callback: Optional[callable] = None
    ) -> str:
        """
        Generates a high-accuracy executive summary for the given transcript in its original language.
        Applies map-reduce chunking for large documents (>12,000 chars).
        """
        if not self._gemini_clients:
            logger.info("No Gemini API key configured. Returning raw transcript text as summary.")
            return transcript_text

        chars_len = len(transcript_text)
        if chars_len > 12000:
            logger.info(f"Document is large ({chars_len} characters). Applying map-reduce chunking...")
            chunks = self._chunk_text(transcript_text, max_chars=10000)
            chunk_summaries = []
            
            for idx, chunk in enumerate(chunks, 1):
                partial_summary = self._run_completion_with_retries(
                    prompt_text=(
                        f"Extract key factual details, numbers, and decisions from this transcript part (Chunk {idx}/{len(chunks)}). "
                        "Keep all extracted points strictly in the same language as the transcript:\n\n"
                        f"{chunk}"
                    ),
                    system_prompt="You are an expert factual summarization assistant. Extract key points accurately in the original language of the text.",
                    temperature=temperature,
                    max_tokens=1500,
                    on_fallback_callback=on_fallback_callback
                )
                chunk_summaries.append(partial_summary)

            combined_summary_input = "\n\n".join(chunk_summaries)
            return self._run_completion_with_retries(
                prompt_text=(
                    "Combine the following chunk summaries into a single cohesive, high-accuracy Executive Summary in the transcript's original language:\n\n"
                    f"{combined_summary_input}"
                ),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_fallback_callback=on_fallback_callback
            )

        return self._run_completion_with_retries(
            prompt_text=(
                "IMPORTANT: Write the entire Executive Summary strictly in the original language of the following transcription. "
                "Do NOT translate to English unless the transcription is in English.\n\n"
                f"Transcription to summarize:\n\n{transcript_text}"
            ),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            on_fallback_callback=on_fallback_callback
        )

    @staticmethod
    def _fallback_translate_to_english(text: str, source_lang: Optional[str] = None) -> str:
        """Fast, high-reliability fallback translation to English via Google Translate API endpoint."""
        if not text or not text.strip():
            return ""
        try:
            import urllib.request
            import urllib.parse
            import json

            sl = source_lang.lower().split("-")[0] if source_lang else "auto"
            q = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl=en&dt=t&q={q}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                translated = "".join([item[0] for item in data[0] if item and item[0]])
                if translated and translated.strip():
                    return translated.strip()
        except Exception as e:
            logger.warning(f"Fallback translation error: {e}")
        return text

    @get_traceable_decorator(name="translate_text", run_type="chain")
    def translate_text(
        self,
        transcript_text: str,
        target_language: str = "English",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        on_fallback_callback: Optional[callable] = None
    ) -> str:
        """
        Translates raw transcript text into target_language using Gemini LLM or fallback translator.
        """
        if not transcript_text or not transcript_text.strip():
            return ""

        if not self._gemini_clients:
            logger.info("No Gemini API key configured. Executing auto-translation to English via fallback engine.")
            return self._fallback_translate_to_english(transcript_text)

        system_prompt = (
            f"You are an expert multilingual translation system. "
            f"Your sole objective is to translate the provided transcription into clear, natural, and highly accurate {target_language} "
            f"while strictly preserving all speaker tags (e.g. [Speaker 0], [Speaker 1]), timestamps, numbers, names, and factual details. "
            f"Output ONLY the translated {target_language} text directly without any introductory or concluding comments."
        )

        try:
            return self._run_completion_with_retries(
                prompt_text=f"Translate the following transcript into {target_language}:\n\n{transcript_text}",
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_fallback_callback=on_fallback_callback
            )
        except Exception as e:
            logger.warning(f"Gemini translation failed ({e}). Using fallback translation engine.")
            return self._fallback_translate_to_english(transcript_text)

    @staticmethod
    def _is_english(text: str) -> bool:
        """
        Helper method to verify if text is actually in English script / Latin language.
        Returns False if text contains Devanagari, Cyrillic, Arabic, CJK, Telugu, Tamil, or other non-Latin scripts.
        """
        if not text or not text.strip():
            return True
        non_latin = [c for c in text if ord(c) > 0x024F and c.isalnum()]
        return len(non_latin) == 0

    def translate_to_english(
        self,
        transcript_text: str,
        detected_language: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        on_fallback_callback: Optional[callable] = None
    ) -> str:
        """Translates transcript to English if non-English."""
        if not transcript_text or not transcript_text.strip():
            return ""

        is_eng = self._is_english(transcript_text)

        if detected_language and is_eng:
            lang_lower = detected_language.lower().strip()
            if lang_lower in ("en", "english", "en-us", "en-gb", "en-in", "eng"):
                logger.info(f"Transcript language is already English ({detected_language}). Skipping translation.")
                return transcript_text

        if not self._gemini_clients:
            logger.info(f"No Gemini API key configured. Translating chunk from '{detected_language or 'auto'}' to English via fallback engine.")
            return self._fallback_translate_to_english(transcript_text, source_lang="auto" if not is_eng else detected_language)

        try:
            res = self.translate_text(
                transcript_text=transcript_text,
                target_language="English",
                temperature=temperature,
                max_tokens=max_tokens,
                on_fallback_callback=on_fallback_callback
            )
            if res and res.strip() and (res.strip() != transcript_text.strip() or not is_eng):
                return res.strip()
        except Exception as e:
            logger.warning(f"LLM translation failed ({e}). Using fallback translation engine.")

        return self._fallback_translate_to_english(transcript_text, source_lang="auto" if not is_eng else detected_language)

    @get_traceable_decorator(name="llm_completion_with_retries", run_type="llm")
    def _run_completion_with_retries(
        self,
        prompt_text: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        on_fallback_callback: Optional[callable]
    ) -> str:
        models_to_try = [self.primary_model] + self.fallback_models
        last_error = None

        from google.genai import types

        for idx, model_name in enumerate(models_to_try):
            for client_idx, client in enumerate(self._gemini_clients):
                for attempt in range(3):
                    try:
                        logger.info(f"Sending LLM request to Gemini model '{model_name}' (Key #{client_idx+1}, attempt {attempt+1})...")
                        config = types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt_text,
                            config=config
                        )
                        content = response.text
                        if content:
                            usage = extract_token_usage(response, model_name=model_name)
                            log_token_usage(usage, context_label="LLM Execution")
                            self.last_token_usage = usage
                            self.total_token_usage.prompt_tokens += usage.prompt_tokens
                            self.total_token_usage.completion_tokens += usage.completion_tokens
                            self.total_token_usage.total_tokens += usage.total_tokens
                            self.total_token_usage.model_name = model_name
                            return content.strip()

                    except Exception as e:
                        last_error = e
                        logger.warning(f"Gemini model '{model_name}' attempt {attempt+1} failed: {e}")
                        time.sleep(2)

        if not self._gemini_clients:
            return prompt_text
        raise RuntimeError(f"All LLM summary generation attempts failed. Last error: {last_error}")