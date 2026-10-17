"""Tests sans réseau du comportement fail-fast de ConfidenceAgent."""

import unittest
from unittest.mock import Mock, patch

from agents.confidence_agent import ConfidenceAgent


class ConfidenceAgentTests(unittest.TestCase):
    @patch("agents.confidence_agent.OpenAI")
    @patch("agents.confidence_agent.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}, clear=False)
    def test_client_uses_short_timeout_without_retries(self, _load_dotenv, openai):
        ConfidenceAgent()

        openai.assert_called_once_with(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="test-key",
            timeout=90.0,
            max_retries=0,
        )

    def test_nvidia_error_is_returned_as_unavailable_metrics(self):
        agent = ConfidenceAgent.__new__(ConfidenceAgent)
        agent.model = "test-model"
        agent.client = Mock()
        agent.client.chat.completions.create.side_effect = TimeoutError("ReadTimeout")

        result = agent.run("Question", "Réponse")

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["entropy"])
        self.assertEqual(result["tokens_analyzed"], 0)
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["error"], "ReadTimeout")
        agent.client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
