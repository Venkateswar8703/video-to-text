import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt_client import ScribeSTTClient, TranscriptionResult, TranscriptionWord
from exporter import format_plain_text, save_transcript_to_file
from summary_generator import SummaryGenerator


class TestSTTPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_words_diarized = [
            TranscriptionWord(text="Hello", start=0.0, end=0.3, type="word", speaker_id="speaker_0"),
            TranscriptionWord(text=" ", start=0.3, end=0.3, type="spacing", speaker_id="speaker_0"),
            TranscriptionWord(text="there!", start=0.3, end=0.6, type="word", speaker_id="speaker_0"),
            TranscriptionWord(text="laughter", start=0.7, end=1.0, type="audio_event", speaker_id="speaker_0"),
            TranscriptionWord(text="Hi", start=1.2, end=1.5, type="word", speaker_id="speaker_1"),
            TranscriptionWord(text="!", start=1.5, end=1.6, type="word", speaker_id="speaker_1"),
        ]
        self.result_diarized = TranscriptionResult(
            raw_text="Hello there! Hi!",
            language_code="en",
            audio_duration_secs=2.0,
            words=self.sample_words_diarized
        )

    def test_parse_response_elevenlabs_style(self):
        mock_api_key = "sk_test_key_123"
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": mock_api_key}):
            with patch("elevenlabs.client.ElevenLabs"):
                client = ScribeSTTClient(elevenlabs_api_key=mock_api_key)
                mock_payload = {
                    "text": "Hello world",
                    "language_code": "en",
                    "audio_duration_secs": 1.5,
                    "words": [
                        {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word", "speaker_id": "speaker_0"},
                        {"text": "world", "start": 0.6, "end": 1.0, "type": "word", "speaker_id": "speaker_0"},
                    ]
                }
                parsed = client._parse_elevenlabs_response(mock_payload)
                self.assertEqual(parsed.raw_text, "Hello world")
                self.assertEqual(len(parsed.words), 2)

    def test_format_plain_text_with_speakers(self):
        formatted = format_plain_text(self.result_diarized, include_speakers=True)
        expected = "[speaker_0]: Hello there! (laughter)\n\n[speaker_1]: Hi!"
        self.assertEqual(formatted, expected)

    def test_format_plain_text_without_speakers(self):
        formatted = format_plain_text(self.result_diarized, include_speakers=False)
        self.assertEqual(formatted, "Hello there! Hi!")

    def test_save_transcript_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_out.txt")
            save_transcript_to_file(self.result_diarized, out_file, include_speakers=True)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("[speaker_0]: Hello there!", content)
            self.assertIn("[speaker_1]: Hi!", content)

    def test_stt_client_elevenlabs_key_initialization(self):
        env_override = {
            "ELEVENLABS_API_KEY": "sk_test_key_123"
        }
        with patch.dict(os.environ, env_override, clear=True):
            with patch("elevenlabs.client.ElevenLabs"):
                client = ScribeSTTClient()
                self.assertEqual(client.default_engine, "elevenlabs")
                self.assertEqual(len(client.elevenlabs_keys), 1)

    def test_summary_generator_gemini_initialization(self):
        mock_gemini = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "# Comprehensive Final Summary"
        mock_gemini.models.generate_content.return_value = mock_resp

        generator = SummaryGenerator(gemini_api_key="test_gemini_key", primary_model="gemini-2.5-flash", fallback_models=[])
        generator._gemini_clients = [mock_gemini]

        summary = generator.generate_summary("Some text to summarize.")
        self.assertEqual(summary, "# Comprehensive Final Summary")


if __name__ == "__main__":
    unittest.main()
