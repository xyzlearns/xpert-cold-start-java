import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check
from xpert.indexes import ConversationIndex, ParticipantRegistry
from xpert.ingestion import load_input
from xpert.pipeline import build_baseline


class PipelineScoreTests(unittest.TestCase):
    def test_example_scores_100_percent(self):
        data = load_input("example")
        out = build_baseline(data, ParticipantRegistry.build(data), ConversationIndex.build(data))
        chats, messages, seed, answers = check.load_input("example")
        self.assertEqual(check.validate(out, chats, messages), [])
        s = check.score(out, answers, seed, messages)
        self.assertEqual(s["circles"], 1.0)
        self.assertEqual(s["organisations"], 1.0)
        self.assertEqual(s["chats"], 1.0)
        self.assertEqual(s["goals"], 1.0)
        self.assertEqual(s["loops_found"], 1.0)
        self.assertEqual(s["loops_closed"], 1.0)
        self.assertEqual(s["help"], 1.0)

    def test_dev_scores_high_if_present(self):
        if not Path("data/dev/chats.jsonl").exists():
            return
        data = load_input("data/dev")
        out = build_baseline(data, ParticipantRegistry.build(data), ConversationIndex.build(data))
        chats, messages, seed, answers = check.load_input("data/dev")
        self.assertEqual(check.validate(out, chats, messages), [])
        s = check.score(out, answers, seed, messages)
        self.assertEqual(s["circles"], 1.0)
        self.assertGreaterEqual(s["organisations"], 0.95)
        self.assertGreaterEqual(s["chats"], 0.95)
        self.assertGreaterEqual(s["loops_found"], 0.65)
        self.assertGreaterEqual(s["loops_closed"], 0.60)

    def test_llm_api_is_called_and_logged_when_configured(self):
        fake_api_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"people": [], "goals": [], "open_loops": []}
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 320, "completion_tokens": 45},
        }

        class FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(fake_api_response).encode("utf-8")

        env_vars = {
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "sk-or-v1-test-key",
            "LLM_MODEL": "openai/gpt-oss-120b",
            "XPERT_USE_CACHE": "0",
        }
        with patch.dict(os.environ, env_vars, clear=False), patch(
            "xpert.llm_client.urllib.request.urlopen",
            return_value=FakeHTTPResponse(),
        ) as mock_urlopen:
            data = load_input("example")
            out = build_baseline(data)
            self.assertEqual( mock_urlopen.call_count, 3)


if __name__ == "__main__":
    unittest.main()
