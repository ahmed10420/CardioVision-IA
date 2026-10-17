"""Tests sans réseau de la résilience Groq et des numéros de page PDF."""

import os
import unittest
from unittest.mock import Mock, call, patch

from agents.groq_client import create_groq_llm
from rag.pdf_metadata import use_one_based_page_number


class GroqClientTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "primary-model",
            "GROQ_FALLBACK_MODEL": "fallback-model",
            "GROQ_MAX_RETRIES": "3",
        },
        clear=True,
    )
    @patch("agents.groq_client.ChatGroq")
    def test_retries_and_fallback_are_configured(self, chat_groq):
        primary = Mock()
        fallback = Mock()
        runnable = Mock()
        primary.with_fallbacks.return_value = runnable
        chat_groq.side_effect = [primary, fallback]

        result = create_groq_llm(max_tokens=1200)

        common = {
            "temperature": 0,
            "reasoning_effort": "low",
            "max_tokens": 1200,
            "api_key": "test-key",
            "max_retries": 3,
        }
        self.assertEqual(
            chat_groq.call_args_list,
            [
                call(model="primary-model", **common),
                call(model="fallback-model", **common),
            ],
        )
        primary.with_fallbacks.assert_called_once_with([fallback])
        self.assertIs(result, runnable)

    @patch.dict(
        os.environ,
        {
            "GROQ_API_KEY": "test-key",
            "GROQ_MAX_RETRIES": "invalid",
        },
        clear=True,
    )
    def test_invalid_retry_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GROQ_MAX_RETRIES"):
            create_groq_llm(max_tokens=100)


class PdfMetadataTests(unittest.TestCase):
    def test_zero_based_page_is_converted_to_human_number(self):
        metadata = {"page": 17, "source": "guideline.pdf"}

        use_one_based_page_number(metadata)

        self.assertEqual(metadata["page"], 18)

    def test_missing_page_is_left_unchanged(self):
        metadata = {"source": "guideline.pdf"}

        use_one_based_page_number(metadata)

        self.assertNotIn("page", metadata)


if __name__ == "__main__":
    unittest.main()
