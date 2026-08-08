"""
test_langsmith_integration.py – Test suite for LangSmith tracer helper and token usage tracking.
"""

import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith_tracer import (
    TokenUsage,
    extract_token_usage,
    get_traceable_decorator,
    is_langsmith_enabled
)


class TestLangSmithTracer(unittest.TestCase):

    def test_token_usage_dataclass(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, model_name="llama-3.3-70b-versatile")
        d = usage.to_dict()
        self.assertEqual(d["prompt_tokens"], 100)
        self.assertEqual(d["completion_tokens"], 50)
        self.assertEqual(d["total_tokens"], 150)
        self.assertEqual(d["model_name"], "llama-3.3-70b-versatile")

    def test_extract_token_usage_openai_format(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 250
        mock_response.usage.completion_tokens = 120
        mock_response.usage.total_tokens = 370

        usage = extract_token_usage(mock_response, model_name="llama-3.3-70b-versatile")
        self.assertEqual(usage.prompt_tokens, 250)
        self.assertEqual(usage.completion_tokens, 120)
        self.assertEqual(usage.total_tokens, 370)
        self.assertEqual(usage.model_name, "llama-3.3-70b-versatile")

    def test_extract_token_usage_gemini(self):
        mock_response = MagicMock(spec=["usage_metadata"])
        mock_response.usage_metadata.prompt_token_count = 500
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.usage_metadata.total_token_count = 700

        usage = extract_token_usage(mock_response, model_name="gemini-2.0-flash")
        self.assertEqual(usage.prompt_tokens, 500)
        self.assertEqual(usage.completion_tokens, 200)
        self.assertEqual(usage.total_tokens, 700)

    def test_traceable_decorator_passthrough(self):
        decorator = get_traceable_decorator(name="test_func", run_type="chain")
        
        @decorator
        def add(a, b):
            return a + b

        result = add(5, 7)
        self.assertEqual(result, 12)


if __name__ == "__main__":
    unittest.main()
